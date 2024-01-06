
import math
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange

from inspect import isfunction
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

def decay_schedule(time_step=100, att_1=2.0, att_T=0.000009):
    decay_weight = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1  # 线性递减

    return decay_weight

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,  # the embed dim
                 n_head,  # the number of heads
                 seq_len=None,  # the max length of sequence
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 causal=True,
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = att * mask.unsqueeze(1).unsqueeze(1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        # att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 condition_seq_len,
                 n_embd,  # the embed dim
                 condition_embd,  # condition dim
                 n_head,  # the number of heads
                 seq_len=None,  # the max length of sequence
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 causal=True,
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                 .view(1, 1, seq_len, seq_len))

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.diff_step = diffusion_step

    def forward(self, x, timestep):
        if timestep[0] >= self.diff_step:
            _emb = self.emb.weight.mean(dim=0, keepdim=True).repeat(len(timestep), 1)
            emb = self.linear(self.silu(_emb)).unsqueeze(1)
        else:
            emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale) + shift
        return x


class SwitchBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 class_type='adalayernorm',
                 class_number=1000,
                 condition_seq_len=2048,
                 n_embd=1024,
                 n_head=16,
                 seq_len=256,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 attn_type='full',
                 if_upsample=False,
                 upsample_type='bilinear',
                 upsample_pre_channel=0,
                 content_spatial_size=None,  # H , W
                 conv_attn_kernel_size=None,  # only need for dalle_conv attention
                 condition_dim=1024,
                 diffusion_step=100,
                 timestep_type='adalayernorm',
                 window_size=8,
                 mlp_type='fc',
                 ):
        super().__init__()
        self.if_upsample = if_upsample
        self.attn_type = attn_type

        if attn_type in ['selfcross', 'selfcondition', 'self']:
            if 'adalayernorm' in timestep_type:
                self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")

        else:
            self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)
        # self.if_selfcross = False
        if attn_type in ['self', 'selfcondition']:
            self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            if attn_type == 'selfcondition':
                if 'adalayernorm' in class_type:
                    self.ln2 = AdaLayerNorm(n_embd, class_number, class_type)
                else:
                    self.ln2 = AdaInsNorm(n_embd, class_number, class_type)
        elif attn_type == 'selfcross':
            self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            self.attn2 = CrossAttention(
                condition_seq_len,
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                seq_len=seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            if 'adalayernorm' in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")

        else:
            print("attn_type error")
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'conv_mlp':
            self.mlp = Conv_MLP(n_embd, mlp_hidden_times, act, resid_pdrop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

    def forward(self, x, cond, timestep, mark=0, mask=None):
        if self.attn_type == "selfcross":
            if mark % 2 == 0:
                # length switch
                x = rearrange(x, 'B (D L) C -> (B D) L C', L=512, D=4)
                mask = rearrange(mask, 'B (D L) -> (B D) L', L=512, D=4)
                a, att = self.attn1(self.ln1(x, timestep.repeat(4, 1).transpose(-1, 0).reshape(4 * timestep.shape[0])), cond, mask=mask)  ### self attention
                x = x + a

                x = rearrange(x, '(B D) L C -> B (D L) C', L=512, D=4)
                a, att = self.attn2(self.ln1_1(x, timestep), cond, mask=mask)  ### cross attention
                x = x + a
            else:
                # dimention switch
                x = rearrange(x, 'B (D L) C -> (B L) D C', L=512, D=4)
                mask = rearrange(mask, 'B (D L) -> (B L) D', L=512, D=4)
                a, att = self.attn1(self.ln1(x, timestep.repeat(512, 1).transpose(-1, 0).reshape(512*timestep.shape[0])), cond, mask=mask)  ### self attention
                x = x + a

                x = rearrange(x, '(B L) D C -> B (D L) C', L=512, D=4)
                a, att = self.attn2(self.ln1_1(x, timestep), cond, mask=mask)  ### cross attention
                x = x + a

        elif self.attn_type == "selfcondition":
            a, att = self.attn(self.ln1(x, timestep), cond, mask=mask)
            x = x + a
            x = x + self.mlp(self.ln2(x, cond.long()))  # only one really use encoder_output
            return x, att

        else:  # 'self'
            a, att = self.attn(self.ln1(x, timestep), cond, mask=mask)
            x = x + a

        x = x + self.mlp(self.ln2(x))

        return x


class Conv_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1,
                               padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1,
                               padding=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        n = x.size()[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        x = self.conv2(self.act(self.conv1(x)))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.dropout(x)


class FlowTransformer(nn.Module):
    def __init__(
            self,
            condition_seq_len=77,
            n_layer=14,
            n_embd=1024,
            content_classes=1024,
            n_head=16,
            content_seq_len=1024,
            attn_pdrop=0,
            resid_pdrop=0,
            mlp_hidden_times=4,
            block_activate=None,
            attn_type='selfcross',
            content_spatial_size=[32, 32],
            condition_dim=512,
            coordinate_classes=256,
            diffusion_step=1000,
            timestep_type='adalayernorm',
            content_emb_config=None,
            mlp_type='fc',
            checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = checkpoint

        # Embedding
        self.content_emb = Content_emb(
            content_classes=content_classes,
            coordinate_classes=coordinate_classes,
            dim=n_embd
        )

        self.condition_emb = Condition_emb(
            content_classes=content_classes - 1,
            coordinate_classes=coordinate_classes,
            dim=condition_dim
        )

        # transformer
        assert attn_type == 'selfcross'
        all_attn_type = [attn_type] * n_layer

        if content_spatial_size is None:
            s = int(math.sqrt(content_seq_len))
            assert s * s == content_seq_len
            content_spatial_size = (s, s)

        self.blocks = nn.Sequential(*[SwitchBlock(
            condition_seq_len,
            n_embd=n_embd,
            n_head=n_head,
            seq_len=content_seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activate=block_activate,
            attn_type=all_attn_type[n],
            content_spatial_size=content_spatial_size,  # H , W
            condition_dim=condition_dim,
            diffusion_step=diffusion_step,
            timestep_type=timestep_type,
            mlp_type=mlp_type,
        ) for n in range(n_layer)])

        # final prediction head
        out_cls = content_classes + 3 * coordinate_classes

        self.norm = nn.LayerNorm(n_embd)
        # self.to_logits = nn.Linear(n_embd, out_cls)
        self.to_logits_index = nn.Linear(n_embd, content_classes)
        self.to_logits_x = nn.Linear(n_embd, coordinate_classes)
        self.to_logits_y = nn.Linear(n_embd, coordinate_classes)
        self.to_logits_z = nn.Linear(n_embd, coordinate_classes)

        self.alpha = nn.Parameter(torch.FloatTensor([1.0]))
        self.beta = nn.Parameter(torch.FloatTensor([1.0]))
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.theta = nn.Parameter(torch.FloatTensor([1.0]))

        mask_weight = decay_schedule(time_step=100)
        mask_weight = torch.tensor(mask_weight.astype('float32'))
        self.register_buffer('mask_weight', mask_weight)

        self.condition_seq_len = condition_seq_len
        self.content_seq_len = content_seq_len

        # self.rezero = Rezero()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

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



    # # rezero
    def forward(
            self,
            input,
            cond_emb,
            t):
        emb, mask = self.content_emb(input)  # Content embedding

        mask_weight = (self.mask_weight[t].unsqueeze(-1)).expand_as(mask)
        mask = torch.where(mask == 0, mask.float(), mask_weight)
        mask = mask.masked_fill(mask == 0, float(1.0))

        for block_idx in range(0, len(self.blocks)-1, 2):
            emb = self.blocks[block_idx](emb, cond_emb, t.cuda(), block_idx, mask)  # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            emb = self.blocks[block_idx+1](emb, cond_emb, t.cuda(), block_idx+1, mask)

        emb = self.blocks[-1](emb, cond_emb, t.cuda(), 0, mask)

        emb = self.norm(emb)  # [B L D]
        logits = torch.cat([self.alpha*self.to_logits_index(emb[:, :512, :]), self.beta*self.to_logits_x(emb[:, 512:1024, :]), \
                            self.gamma*self.to_logits_y(emb[:, 1024:1536, ]), self.theta*self.to_logits_z(emb[:, 1536:, :])], dim=-1)

        out = rearrange(logits, 'b l c -> b c l')
        return out


# Embedding
def embed(input, basis):
    # print(input.shape, basis.shape)
    projections = torch.einsum(
        'bnd,de->bne', input, basis)  # .permute(2, 0, 1)
    # print(projections.max(), projections.min())
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings  # B x N x E


# Here use absolute position embeddings
class Condition_emb(nn.Module):
    def __init__(self,
                 content_classes=1024,
                 coordinate_classes=256,
                 r_classes=256,
                 theta_classes=180,
                 phi_classes=360,
                 dim=1024):
        super().__init__()

        self.embed = nn.Sequential(nn.Linear(48 + 3, dim))  # , nn.GELU(), Lin(128, 128))
        self.num_classes = content_classes + 3 * coordinate_classes + 1

        # self.num_classes = content_classes + r_classes + theta_classes + phi_classes + 1

        self.embedding = nn.Embedding(self.num_classes, dim)

        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

    def forward(self, input, centers):
        content_emb = self.embedding(input[:, 0, :])
        x_emb = self.embedding(input[:, 1, :])
        y_emb = self.embedding(input[:, 2, :])
        z_emb = self.embedding(input[:, 3, :])

        pos_emb = embed(centers, self.basis)
        pos_emb = self.embed(torch.cat([centers, pos_emb], dim=2))

        emb = content_emb + x_emb + y_emb + z_emb + pos_emb

        return emb


# Here use relative position embeddings
class Content_emb(nn.Module):
    def __init__(self,
                 content_classes=1024,
                 coordinate_classes=128,
                 r_classes=256,
                 theta_classes=180,
                 phi_classes=360,
                 dim=1024):
        super().__init__()

        self.position_emb = nn.Parameter(nn.Embedding(2048, dim).weight[None])
        self.num_classes = content_classes + 3 * coordinate_classes + 1


        self.embedding = nn.Embedding(self.num_classes, dim)


    def forward(self, input):

        mask = (input==self.num_classes-1).long()
        content_emb, x_emb, y_emb, z_emb = input.chunk(4, 1)

        content_emb = self.embedding(content_emb.squeeze())
        x_emb = self.embedding(x_emb.squeeze())
        y_emb = self.embedding(y_emb.squeeze())
        z_emb = self.embedding(z_emb.squeeze())

        pos_emb = self.position_emb

        emb = torch.cat([content_emb, x_emb, y_emb, z_emb], dim=-2) + pos_emb

        return emb, mask



