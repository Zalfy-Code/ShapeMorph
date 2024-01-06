import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np

from utils import utils
from utils.optimizer.lr_scheduler import ReduceLROnPlateauWithWarmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.distributed.distributed import reduce_dict
from utils.Virscan import sample_sphere, hidden_point_removal

STEP_WITH_LOSS_SCHEDULERS = (ReduceLROnPlateauWithWarmup, ReduceLROnPlateau)


def VirScan(centers, Xbd_quantized, Xbd_encoding,radius=10, C=None):


    np.random.seed(3)
    C = sample_sphere(Xbd_quantized.shape[0]) * radius
    Xct, Xct_encoding, vis = hidden_point_removal(Xbd_quantized.cpu(), Xbd_encoding.cpu(), centers.cpu().float(), C)

    return Xct, Xct_encoding, vis


def process_idx(encoding, center_quan):
    centers = center_quan.float() / 255.0 * 2 - 1
    center_quan[..., 0] = center_quan[..., 0].add(1024)
    center_quan[..., 1] = center_quan[..., 1].add(1024+256)
    center_quan[..., 2] = center_quan[..., 2].add(1024+256+256)

    coding = torch.cat((encoding[..., None], center_quan), dim=-1).permute(0, -1, -2)

    return coding, centers

def get_next_cond(c_pos_indices, z_pos_indices):
    # find next position of condition
    next_ids = torch.searchsorted(c_pos_indices[:, :, 0].contiguous(), z_pos_indices[:, :, 0].contiguous(), right=True)
    next_ids[next_ids == c_pos_indices.shape[1]] = c_pos_indices.shape[1] - 1
    next_ids = next_ids[:, :, None].expand(-1, -1, 3)

    next_cond_pos = torch.gather(c_pos_indices, dim=1, index=next_ids)

    if (z_pos_indices.shape[1] == 0):
        return z_pos_indices.clone()

    # print(next_cond_pos)
    return next_cond_pos


def get_extra_indices(Xct_pos, Xbd_pos):
    # L_c, L_z = Xct_pos.shape[1], Xbd_pos.shape[1]

    Xct_extra = Xct_pos.clone()
    Xbd_extra = get_next_cond(Xct_pos, Xbd_pos)
    extra_indices = torch.cat([Xct_extra, Xbd_extra], axis=1)
    # print("!",extra_indices)
    return extra_indices


def random_mask(quan_centers, encodings, vis, device):
    # center_end = torch.tensor([[[256, 256, 256]]])
    # latent_end = torch.tensor([[1024]])
    #
    # center_end = center_end.expand(quan_centers.shape[0], 1, -1).to(device, non_blocking=True)
    # latent_end = latent_end.expand(encodings.shape[0], 1).to(device, non_blocking=True)
    vis_empty = torch.full_like(vis, 512)
    max_num = (torch.min(torch.argmax(vis, dim=-1))-1).cpu().int()
    select_num = np.random.randint(1, max_num)
    selected_ind = np.sort(np.random.choice(max_num, select_num, replace=False))

    vis = vis[:, selected_ind]
    quan_centers = torch.gather(quan_centers, 1, vis.unsqueeze(-1).repeat(1, 1, 3))
    encodings = torch.gather(encodings, 1, vis)
    # vis_empty = torch.scatter(vis_empty, 1, vis, vis)

    assert torch.sum(quan_centers==256)==0 or torch.sum(encodings==1024)==0

    return quan_centers, encodings, vis


def sort(centers_quantized, encodings, centers):
    ind3 = torch.argsort(centers_quantized[:, :, 2], dim=1)
    centers_quantized = torch.gather(centers_quantized, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    centers = torch.gather(centers, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind3)

    _, ind2 = torch.sort(centers_quantized[:, :, 1], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    centers = torch.gather(centers, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind2)

    _, ind1 = torch.sort(centers_quantized[:, :, 0], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    centers = torch.gather(centers, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind1)
    return centers_quantized, encodings, centers



def train_batch(model, vqvae, Xbd, Xct, device):
    with torch.no_grad():
        _, _, Xct_centers_quantized, _, _, Xct_encodings, Xct_centers = vqvae.encode(Xct)
        _, _, Xbd_centers_quantized, _, _, Xbd_encodings, Xbd_centers = vqvae.encode(Xbd)

    Xct_centers_quantized, Xct_encodings, Xct_centers = sort(Xct_centers_quantized, Xct_encodings, Xct_centers)
    Xbd_centers_quantized, Xbd_encodings, Xbd_centers = sort(Xbd_centers_quantized, Xbd_encodings, Xbd_centers)

    Xct_centers_quantized, Xct_encodings, vis = VirScan(Xct_centers, Xct_centers_quantized, Xct_encodings)

    Xct_centers_quantized, Xct_encodings, vis = random_mask(Xct_centers_quantized, Xct_encodings, vis, device)

    input, _ = process_idx(Xbd_encodings, Xbd_centers_quantized)
    input = torch.cat([input[:, 0, :].squeeze(1), input[:, 1, :].squeeze(1), \
                       input[:, 2, :].squeeze(1), input[:, 3, :].squeeze(1)], dim=-1)
    condition, Xct_centers = process_idx(Xct_encodings, Xct_centers_quantized)

    output = model(input, condition, Xct_centers)
    return output


def train_one_epoch(model: torch.nn.Module, vqvae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, scheduler: object,
                    device: torch.device, epoch: int, scaler: Optional[object]=None,
                    model_ema: Optional[object] = None, clip_grad_norm: Optional[object]=None,
                    log_writer=None, start_steps=None, num_training_steps_per_epoch=None,
                    update_freq=None, args=None):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (_, _, Xbd, Xct) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # for data_iter_step, (_, _, Xbd, Xct) in enumerate(data_loader):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        Xbd = Xbd.to(device, non_blocking=True)
        Xct = Xct.to(device, non_blocking=True)

        if args.amp:
            with torch.cuda.amp.autocast():
                output = train_batch(model, vqvae, Xbd, Xct, device)
        else:
            output = train_batch(model, vqvae, Xbd, Xct, device)

        loss_value = output['loss'].item()

        loss_dict = {k: v for k, v in output.items() if ('loss' in k)}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # optimizer
        if args.amp:
            scaler.scale(output['loss'] / args.update_step).backward()
            if clip_grad_norm is not None:
                clip_grad_norm(model.parameters())
            if (step+1) % args.update_step == 0 or (step+1) == num_training_steps_per_epoch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            (output['loss'] / args.update_step).backward()
            if clip_grad_norm is not None:
                clip_grad_norm(model.parameters())
            if (step+1) % args.update_step == 0 or (step+1) == num_training_steps_per_epoch:
                optimizer.step()
                optimizer.zero_grad()

        # scheduler
        if (step+1) % args.update_step == 0 or (step+1) == num_training_steps_per_epoch:
            if isinstance(scheduler, STEP_WITH_LOSS_SCHEDULERS):
                scheduler.step(output.get('loss') / args.update_step)
            else:
                scheduler.step()

        if model_ema is not None:
            model_ema.update(iteration=it)


        torch.cuda.synchronize()

        loss_value = reduce_dict(loss_dict)['loss'].item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.state_dict()['param_groups'][0]['lr'])

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.set_step()


    metric_logger.synchronize_between_processes()
    if utils.is_main_process():
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, vqvae, device, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    model.eval()

    for batch in metric_logger.log_every(data_loader, 100, header):
        _, Xbd, Xct = batch
        Xbd = Xbd.to(device, non_blocking=True)
        Xct = Xct.to(device, non_blocking=True)

        if args.amp:
            with torch.cuda.amp.autocast():
                output = train_batch(model, vqvae, Xbd, Xct, device)
        else:
            output = train_batch(model, vqvae, Xbd, Xct, device)

        loss_value = output['loss'].item()

        metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f} '
          .format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
