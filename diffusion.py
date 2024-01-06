
import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from inspect import isfunction
from torch.cuda.amp import autocast


from utils.diffusion.transformer_switch_decay import FlowTransformer

from vqvae import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

eps = 1e-8


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


'''
   |log(0)+log(a) |    log(a)    |log(0)+log(a) |  
a= |    log(a)    |log(0)+log(a) |log(0)+log(a) |
   |log(0)+log(a) |log(0)+log(a) |    log(a)    |


          |log(b) |log(a) |log(b) |  
maximum = |log(a) |log(b) |log(b) |
          |log(b) |log(b) |log(a) |

                 |0 |a |0 |
exp(a-maximum) = |1 |0 |0 |
                 |0 |0 |1 |

                 |0   |b/a |0   |
exp(b-maximum) = |b/a |0   |0   |
                 |0   |0   |b/a |   

             |b   |a+b |b   |    
return = log |a+b |b   |b   |
             |b   |b   |a+b |           
'''


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)



def index_to_log_multihot(input, num_classes):

    input_multihot = F.one_hot(input, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(input.size())))

    input_multihot = input_multihot.permute(permute_order)

    log_x = torch.log(input_multihot.float().clamp(min=1e-30))
    return log_x


def log_multihot_to_index(log_x):
    return log_x.argmax(1)


def alpha_schedule(time_step, N=100, C=256, att_1=0.9, att_T=0.000009, ctt_1=0.000009, ctt_T=0.9):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    bt_c = (1 - at - ct) / C
    btt_c = (1 - att - ctt) / C
    return at, bt, bt_c, ct, att, btt, btt_c, ctt



class DiffusionTransformer(nn.Module):
    def __init__(
            self,
            *,
            emb_dim=1024,
            content_classes=1024,
            coordinate_classes=128,
            content_seq_len=2048,
            condition_seq_len=2048,
            diffusion_step=100,
            alpha_init_type='cos',
            auxiliary_loss_weight=0,
            adaptive_auxiliary_loss=False,
            mask_weight=[1, 1],

            learnable_cf=False,
    ):
        super().__init__()

        # network
        self.transformer = FlowTransformer(
            condition_seq_len='dynamic',
            n_layer=17,  # Number of transformer layers
            n_embd=emb_dim,  # Dimension of input embedding
            content_classes=content_classes,  # Content classes
            n_head=16,  # Number of transformer head
            content_seq_len=content_seq_len // 4,  # Length of input sequence
            attn_pdrop=0,
            resid_pdrop=0,
            mlp_hidden_times=4,
            block_activate='GELU2',  # Activate function
            attn_type='selfcross',  # Type of attention
            condition_dim=emb_dim,  # Dimemsion of condition embedding
            coordinate_classes=coordinate_classes,  # Coordinate classes
            diffusion_step=diffusion_step,  # Diffusion time step
            timestep_type='adalayernorm',  # Type of diffusion AdaIN
            mlp_type='fc',  # Type of mlp
            checkpoint=False,  # Checkpoint
        )

        self.amp = False

        self.content_seq_len = content_seq_len
        self.content_classes = content_classes
        self.coordinate_classes = coordinate_classes
        self.num_classes = self.content_classes + 3 * self.coordinate_classes + 1

        self.loss_type = 'vb_stochastic'
        self.shape = content_seq_len
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight

        if alpha_init_type == "alpha1":
            at, bt, bt_c, ct, att, btt, btt_c, ctt = alpha_schedule(self.num_timesteps, N=self.content_classes, C=self.coordinate_classes)
        else:
            print("alpha_init_type is Wrong !! ")

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        bt_c = torch.tensor(bt_c.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_bt_c = torch.log(bt_c)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        btt_c = torch.tensor(btt_c.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_bt_c = torch.log(btt_c)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_bt_c', log_bt_c.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_bt_c', log_cumprod_bt_c.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None

        if learnable_cf:
            self.empty_text_embed = torch.nn.Parameter(
                torch.randn(size=(2048, 512), requires_grad=True, dtype=torch.float64))

        self.prior_rule = 0
        self.prior_ps = 1792
        self.prior_weight = 0

        self.update_n_sample()

        self.learnable_cf = learnable_cf

    def update_n_sample(self):
        if self.num_timesteps == 100:
            if self.prior_ps <= 10:
                self.n_sample = [1, 6] + [11, 10, 10] * 32 + [11, 15]
            else:
                self.n_sample = [1, 6] + [11, 10, 10] * 32 + [11, 15]
        elif self.num_timesteps == 50:
            self.n_sample = [10] + [21, 20] * 24 + [30]
        elif self.num_timesteps == 25:
            self.n_sample = [21] + [41] * 23 + [60]
        elif self.num_timesteps == 10:
            self.n_sample = [69] + [102] * 8 + [139]

    # 这里计算KL散度 / KL_Loss
    def multinomial_kl(self, log_prob1, log_prob2):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)  # bt
        log_bt_c = extract(self.log_bt_c, t, log_x_t.shape)  # bt_c
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :self.content_classes, :] + log_at, log_bt),

                log_add_exp(log_x_t[:, self.content_classes:self.content_classes + self.coordinate_classes, :] + log_at, log_bt_c),
                log_add_exp(log_x_t[:, self.content_classes + self.coordinate_classes:self.content_classes + 2 * self.coordinate_classes, :] + log_at, log_bt_c),
                log_add_exp(log_x_t[:, self.content_classes + 2 * self.coordinate_classes:self.content_classes + 3 * self.coordinate_classes, :] + log_at, log_bt_c),

                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    # 这里计算的是转移概率矩阵
    def q_pred(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_bt_c = extract(self.log_cumprod_bt_c, t, log_x_start.shape)  # bt_c~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :self.content_classes, :] + log_cumprod_at, log_cumprod_bt),

                log_add_exp(log_x_start[:, self.content_classes:self.content_classes + self.coordinate_classes, :] + log_cumprod_at, log_cumprod_bt_c),
                log_add_exp(log_x_start[:, self.content_classes + self.coordinate_classes:self.content_classes + 2 * self.coordinate_classes, :] + log_cumprod_at, log_cumprod_bt_c),
                log_add_exp(log_x_start[:, self.content_classes + 2 * self.coordinate_classes:self.content_classes + 3 * self.coordinate_classes, :] + log_cumprod_at, log_cumprod_bt_c),

                log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def multi_log_pred(self, logits):
        batch_size = logits.size()[0]
        log_pred_index = F.log_softmax(logits[:, :self.content_classes, :], dim=1).float()
        log_pred_x = F.log_softmax(logits[:, self.content_classes: self.content_classes + self.coordinate_classes, :], dim=1).float()
        log_pred_y = F.log_softmax(logits[:, self.content_classes + self.coordinate_classes: self.content_classes + 2 * self.coordinate_classes, :], dim=1).float()
        log_pred_z = F.log_softmax(logits[:, self.content_classes + 2 * self.coordinate_classes:, :], dim=1).float()

        index_zero_vectors = torch.zeros(batch_size, self.num_classes - self.content_classes, self.content_seq_len // 4).type_as(log_pred_index) - 70
        coordinate_zero_vectors = torch.zeros(batch_size, self.num_classes - self.coordinate_classes, self.content_seq_len // 4).type_as(log_pred_x) - 70

        log_pred_index = torch.cat([log_pred_index, index_zero_vectors], dim=1)
        log_pred_x = torch.cat([coordinate_zero_vectors[:, :self.content_classes, :], log_pred_x, coordinate_zero_vectors[:, self.content_classes:, :]], dim=1)
        log_pred_y = torch.cat([coordinate_zero_vectors[:, :self.content_classes + self.coordinate_classes, :], log_pred_y, coordinate_zero_vectors[:, self.content_classes + self.coordinate_classes:, :]], dim=1)
        log_pred_z = torch.cat([coordinate_zero_vectors, log_pred_z], dim=1)

        return torch.cat([log_pred_index, log_pred_x, log_pred_y, log_pred_z], dim=-1)

    def predict_start(self, log_x_t, cond_emb, t):  # p(x0|xt)
        x_t = log_multihot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                logits = self.transformer(x_t, cond_emb, t)
        else:
            logits = self.transformer(x_t, cond_emb, t)

        assert logits.size(0) == x_t.size(0)
        assert logits.size(1) == self.num_classes - 1

        log_pred = self.multi_log_pred(logits.double())
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_multihot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, self.content_seq_len)

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, cond_emb, t, delta=None):  # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':

            log_x_recon = self.cf_predict_start(log_x, cond_emb=cond_emb, t=t)  # condition generation

            if delta is None:
                delta = 0

            log_x0_recon = log_x_recon
            if t[0].item() >= delta:
                log_model_pred = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_x, t=t - delta)
            else:
                log_model_pred = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb=cond_emb, t=t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, t, cond_emb, delta=None, sampled=None, to_sample=None):
        model_log_prob, log_x_recon = self.p_pred(log_x=log_x, t=t, cond_emb=cond_emb, delta=delta)

        max_sample_per_step = self.prior_ps
        if t[0] > 0 and self.prior_rule > 0 and to_sample is not None:
            log_x_idx = log_multihot_to_index(log_x)

            if self.prior_rule == 1:
                score = torch.ones((log_x.shape[0], log_x.shape[2])).to(log_x.device)
            elif self.prior_rule == 2:
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                score /= (score.max(dim=1, keepdim=True).values + 1e-10)

            if self.prior_rule != 1 and self.prior_weight > 0:
                prob = ((1 + score * self.prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon

            out = self.log_sample_categorical(prob)
            out_idx = log_multihot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.num_classes - 1] = 0

            for i in range(log_x.shape[0]):
                n_sample = min(to_sample - sampled[i], max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                sel = torch.multinomial(_score[i], n_sample)
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += ((out2_idx[i] != self.num_classes - 1).sum() - (
                            log_x_idx[i] != self.num_classes - 1).sum()).item()

            out = index_to_log_multihot(out2_idx, self.num_classes)

        else:
            out = self.log_sample_categorical(model_log_prob)
            sampled = [1792] * log_x.shape[0]

        if to_sample is not None:
            return out, sampled
        else:
            return out


    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_multihot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond_emb, is_train=True):  # get the KL loss
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')

        log_x_start = index_to_log_multihot(x_start, self.num_classes)  # [B, num_classes, 512]
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)

        xt = log_multihot_to_index(log_xt)

        ############### go to p_theta function ###############

        log_x0_recon_multi = self.predict_start(log_xt, cond_emb, t=t)  # sum(P_theta(x0^i|xt,y))
        log_x0_recon = log_x0_recon_multi

        log_x0_recon_index = torch.cat((log_x0_recon_multi[:, :, :512], log_x_start[:, :, 512:]), dim=-1)
        log_x0_recon_x = torch.cat((log_x_start[:, :, :512], log_x0_recon_multi[:, :, 512: 1024], log_x_start[:, :, 1024:]), dim=-1)
        log_x0_recon_y = torch.cat((log_x_start[:, :, :1024], log_x0_recon_multi[:, :, 1024: 1536], log_x_start[:, :, 1536:]), dim=-1)
        log_x0_recon_z = torch.cat((log_x_start[:, :, :1536], log_x0_recon_multi[:, :, 1536:]), dim=-1)

        log_model_prob_index = self.q_posterior(log_x_start=log_x0_recon_index, log_x_t=log_xt, t=t)  # go through q(xt_1|xt,x0,xt_1^i)
        log_model_prob_x = self.q_posterior(log_x_start=log_x0_recon_x, log_x_t=log_xt, t=t)
        log_model_prob_y = self.q_posterior(log_x_start=log_x0_recon_y, log_x_t=log_xt, t=t)
        log_model_prob_z = self.q_posterior(log_x_start=log_x0_recon_z, log_x_t=log_xt, t=t)

        log_model_prob = torch.cat([log_model_prob_index[:, :, :512], \
                                    log_model_prob_x[:, :, 512: 1024], \
                                    log_model_prob_y[:, :, 1024: 1536], \
                                    log_model_prob_z[:, :, 1536:]], dim=-1)

        ################## compute acc list ################
        x0_recon = log_multihot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_multihot_to_index(log_model_prob)
        xt_recon = log_multihot_to_index(log_xt)

        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu() / x0_real.size()[1]
            self.diffusion_acc_list[this_t] = same_rate.item() * 0.1 + self.diffusion_acc_list[this_t] * 0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu() / xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item() * 0.1 + self.diffusion_keep_list[this_t] * 0.9

        # compute log_true_prob now
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)  # L_t-1
        kl = sum_except_batch(kl)


        decoder_nll = -log_categorical(log_x_start, log_model_prob)  # L_0
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))


        loss1 = kl_loss / pt
        vb_loss = loss1

        if self.auxiliary_loss_weight != 0 and is_train == True:
            kl_aux = self.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1 - t / self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2

        return log_model_prob, vb_loss

    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device

    def parameters(self, recurse=True, name=None):

        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}  # if p.requires_grad}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self,
            input,
            condition,
            vis,
            return_loss=True,
            return_logits=True,
            return_att_weight=False,
            is_train=True,
            **kwargs):
        if kwargs.get('autocast') == True:
            self.amp = True
        batch_size = input.shape[0]
        device = input.device

        # 1) get embeddding for condition and content     prepare input
        content_emb = input.type_as(input)
        # cont_emb = self.content_emb(sample_image)
        if condition == None:
            cond_emb = None
        else:
            cond_emb = self.transformer.condition_emb(condition, vis)

        # now we get cond_emb and sample_image
        if is_train == True:
            log_model_prob, loss = self._train_loss(content_emb, cond_emb)
            loss = loss.sum() / (content_emb.size()[0] * content_emb.size()[1])

        # 4) get output, especially loss
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss
        self.amp = False
        return out

    def cf_predict_start(self, log_x_t, cond_emb, t, guidance_scale):
        zero_vector = torch.zeros(log_x_t.shape[0], 1, self.content_seq_len).type_as(log_x_t) - 70

        log_x_recon = self.predict_start(log_x_t, cond_emb, t)[:, :-1]
        if abs(guidance_scale - 1) < 1e-3:
            return torch.cat((log_x_recon, zero_vector), dim=1)
        cf_log_x_recon = self.predict_start(log_x_t, cond_emb, t)[:, :-1]
        log_new_x_recon = cf_log_x_recon + guidance_scale * (log_x_recon - cf_log_x_recon)
        log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
        log_new_x_recon = log_new_x_recon.clamp(-70, 0)
        log_pred = torch.cat((log_new_x_recon, zero_vector), dim=1)
        return log_pred

    def predict_start_with_truncation(self, func, sample_type):

        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path

            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k=truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs

            return wrapper

        elif sample_type[-1] == 'r':

            truncation_r = float(sample_type[:-1].replace('top', ''))

            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True)
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:, 0:1, :], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:, :-1, :]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float() * out + (1 - temp4.float()) * (-70)
                probs = temp5
                return probs

            return wrapper

        else:
            print("wrong sample type")

    @torch.no_grad()
    def sample_mask(self,
                    condition,
                    Xct_pos,
                    num_samples):

        if condition == None:
            cond_emb = None
        else:
            cond_emb = self.transformer.condition_emb(condition, Xct_pos)

        # num_samples is batch size
        b = num_samples
        device = self.log_at.device
        self.shape = (b, self.content_seq_len)
        zero_logits = torch.zeros((b, self.num_classes - 1, self.content_seq_len), device=device)
        one_logits = torch.ones((b, 1, self.content_seq_len), device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)


        sample_type = "top0.85r"

        self.cf_predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        from tqdm import tqdm

        for i in tqdm((range(self.num_timesteps - 1, -1, -1)), desc="Chain timestep ", total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            sampled = [0] * log_z.shape[0]
            while min(sampled) < self.n_sample[i]:
                log_z, sampled = self.p_sample(log_z, t, cond_emb, sampled=sampled,
                                               to_sample=self.n_sample[i])

        zs = log_multihot_to_index(log_z)

        return zs

    def sample(
            self,
            condition,
            Xct_pos,
            num_samples,
            content_token=None,
            filter_ratio=0.5,
            temperature=1.0,
            return_att_weight=False,
            return_logits=False,
            content_logits=None,
            print_log=True,
            **kwargs):

        if condition == None:
            cond_emb = None
        else:
            cond_emb = self.transformer.condition_emb(condition, Xct_pos)

        batch_size = num_samples
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio)

        # get cont_emb and cond_emb
        if content_token != None:
            sample_image = input['content_token'].type_as(input['content_token'])

        if self.condition_emb is not None:  # do this
            with torch.no_grad():
                cond_emb = self.condition_emb(input['condition_token'])  # B x Ld x D   #256*1024
            cond_emb = cond_emb.float()
        else:  # share condition embeding with content
            if input.get('condition_embed_token', None) != None:
                cond_emb = input['condition_embed_token'].float()
            else:
                cond_emb = None

        if start_step == 0:
            # use full mask sample
            zero_logits = torch.zeros((batch_size, self.num_classes - 1, self.content_seq_len), device=device)
            one_logits = torch.ones((batch_size, 1, self.content_seq_len), device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z = torch.log(mask_logits)
            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    sampled = [0] * log_z.shape[0]
                    while min(sampled) < self.n_sample[diffusion_index]:
                        log_z, sampled = self.p_sample(log_z, cond_emb, t, sampled,
                                                       self.n_sample[diffusion_index])  # log_z is log_onehot

        else:
            t = torch.full((batch_size,), start_step - 1, device=device, dtype=torch.long)
            log_x_start = index_to_log_multihot(sample_image, self.num_classes)
            log_xt = self.q_sample(log_x_start=log_x_start, t=t)
            log_z = log_xt
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_z = self.p_sample(log_z, cond_emb, t)  # log_z is log_onehot

        content_token = log_multihot_to_index(log_z)

        output = {'content_token': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output

    def sample_fast(
            self,
            condition_token,
            condition_mask,
            condition_embed,
            content_token=None,
            filter_ratio=0.5,
            temperature=1.0,
            return_att_weight=False,
            return_logits=False,
            content_logits=None,
            print_log=True,
            skip_step=1,
            **kwargs):
        input = {'condition_token': condition_token,
                 'content_token': content_token,
                 'condition_mask': condition_mask,
                 'condition_embed_token': condition_embed,
                 'content_logits': content_logits,
                 }

        batch_size = input['condition_token'].shape[0]
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio)

        # get cont_emb and cond_emb
        if content_token != None:
            sample_image = input['content_token'].type_as(input['content_token'])

        if self.condition_emb is not None:
            with torch.no_grad():
                cond_emb = self.condition_emb(input['condition_token'])  # B x Ld x D   #256*1024
            cond_emb = cond_emb.float()
        else:  # share condition embeding with content
            cond_emb = input['condition_embed_token'].float()

        assert start_step == 0
        zero_logits = torch.zeros((batch_size, self.num_classes - 1, self.shape), device=device)
        one_logits = torch.ones((batch_size, 1, self.shape), device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps
        with torch.no_grad():
            # skip_step = 1
            diffusion_list = [index for index in range(start_step - 1, -1, -1 - skip_step)]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            # for diffusion_index in range(start_step-1, -1, -1):
            for diffusion_index in diffusion_list:

                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = self.cf_predict_start(log_z, cond_emb, t)
                if diffusion_index > skip_step:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t - skip_step)
                else:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)

                log_z = self.log_sample_categorical(model_log_prob)

        content_token = log_multihot_to_index(log_z)

        output = {'content_token': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output


@register_model
def class_diffusion_transformer(pretrained=False, **kwargs):
    model = DiffusionTransformer(
        content_seq_len=2048,
        diffusion_step=100,
        alpha_init_type='alpha1',
        auxiliary_loss_weight=0.0005,
        adaptive_auxiliary_loss=True,
        mask_weight=[1, 1],
    )
    model.default_cfg = _cfg()
    return model

