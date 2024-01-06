import torch
from torch.utils.data import ConcatDataset
from utils.distributed.distributed import is_distributed
from datasets import build_shape_surface_occupancy_dataset


def build_dataloader(args=None, return_dataset=False):

    train_dataset = build_shape_surface_occupancy_dataset('train', args=args)
    if args.disable_eval:
        val_dataset = None
    else:
        val_dataset = build_shape_surface_occupancy_dataset('test', args=args)


    train_dataset = train_dataset
    if val_dataset != None:
        val_dataset = val_dataset
    
    if args is not None and args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_iters = len(train_sampler) // args.batch_size
        if val_dataset:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            val_iters = len(val_sampler) // args.batch_size
    else:
        train_sampler = None
        train_iters = len(train_dataset) // args.batch_size
        if val_dataset:
            val_sampler = None
            val_iters = len(val_dataset) // args.batch_size

    # if args is not None and not args.debug:
    #     num_workers = max(2*dataset_cfg['batch_size'], dataset_cfg['num_workers'])
    #     num_workers = min(64, num_workers)
    # else:
    #     num_workers = dataset_cfg['num_workers']
    num_workers = args.num_workers
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True,
                                               persistent_workers=True)

    if val_dataset:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False, #(val_sampler is None),
                                             num_workers=num_workers, 
                                             pin_memory=True, 
                                             sampler=val_sampler, 
                                             drop_last=True,
                                             persistent_workers=True)

    dataload_info = {
        'train_loader': train_loader,
        'train_iterations': train_iters,
        'train_length': len(train_dataset)
    }

    if val_dataset:
        dataload_info['validation_loader'] = val_loader
        dataload_info['validation_iterations'] = val_iters
        dataload_info['val_length'] = len(val_dataset)
    if return_dataset:
        dataload_info['train_dataset'] = train_dataset
        dataload_info['validation_dataset'] = val_dataset


    return dataload_info
