import argparse
import datetime
import numpy as np
import importlib
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
import warnings

from timm.models import create_model

from utils.misc import seed_everything
from utils.distributed.launch import launch

from utils.optimizer.lr_scheduler import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from utils.ema import EMA

from engine_for_vqvae import train_one_epoch, evaluate
from utils import utils
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from utils.utils import auto_load_model_ae, save_model_ae

import vqvae

# environment variables
NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)

def get_args():
    parser = argparse.ArgumentParser('VQVAE', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # Model parameters
    parser.add_argument('--model', default='vqvae_128', type=str,
                        metavar='MODEL', help='Name of model to train')
    parser.add_argument('--point_cloud_size', default=2048, type=int,
                        help='images point cloud size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)

    # config
    parser.add_argument('--dataset_cfg', type=str, default='utils.data.build',
                        help='Dataset Module Path')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', # 1e-6
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', #
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=1.0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=80, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Model EMA
    parser.add_argument('--model_ema_decay', type=float, default=0.99, help='')
    parser.add_argument('--update_interval', type=int, default=25, help='')

    # Dataset parameters
    parser.add_argument('--data_path', default='/your/data/path', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # args for ddp
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=NODE_RANK,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default=DIST_URL,
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use. If given, only the specific gpu will be'
                             ' used, and ddp will be disabled')
    parser.add_argument('--sync_bn', action='store_true',
                        help='use sync BN layer')
    parser.add_argument('--tensorboard', action='store_true',
                        help='use tensorboard for logging')
    parser.add_argument('--timestamp', action='store_true',
                        help='use tensorboard for logging')

    # args for random
    parser.add_argument('--seed', type=int, default=114514,
                        help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', action='store_true',
                        help='set cudnn.deterministic True')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='automatic mixture of precesion')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='set as debug mode')

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    # modify args for debugging
    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0

    return args


def main():

    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


def main_worker(local_rank, args):

    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1

    print(args)
    device = torch.device(args.device)

    # dataset and dataloader
    build_data = importlib.import_module(args.dataset_cfg)

    dataloader = build_data.build_dataloader(args)

    data_loader_train = dataloader['train_loader']
    train_iters = dataloader['train_iterations']

    try:
        data_loader_val = dataloader['validation_loader']
        val_iters = dataloader['validation_iterations']
    except:
        data_loader_val = None
        val_iters = None

    # log_writer
    if args.global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    model = create_model(
        args.model,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model EMA
    if args.model_ema:
        model_ema = EMA(
            model,
            decay=args.model_ema_decay,
            update_interval=args.update_interval,
            device='cpu')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    else:
        model_ema = None

    model.to(device)

    # Model DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = dataloader['train_length'] // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % dataloader['train_length'])
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)


    # num_layers = model_without_ddp.get_num_layers()
    num_layers = 12
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    # optimizer
    optimizer = create_optimizer(
        args, model, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
    loss_scaler = NativeScaler()

    # scheduler
    print("Use step level LR scheduler!")
    lr_schedule_values = create_optimizer(args, optimizer, num_training_steps_per_epoch)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = create_optimizer(args, optimizer, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    print("criterion = %s" % str(criterion))

    # auto load model
    auto_load_model_ae(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        )

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_model_ae(
                    args=args, model=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {dataloader['val_length']} test images: {test_stats['iou']:.1f}%")
            if max_accuracy < test_stats["iou"]:
                max_accuracy = test_stats["iou"]
                if args.output_dir and args.save_ckpt:
                    save_model_ae(
                        args=args, model=model, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(test_iou=test_stats['iou'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         # **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()