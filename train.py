import os
import os.path as op
import torch
import numpy as np
import random
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import datasets
from datasets.build import build_test_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
from utils.iotools import read_json
from typing import List
from datasets.bases import show_dataset_info

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_data(args):
    target_root = op.join(args.root_dir, args.target_dataset_name)
    target_dataset = datasets.create(args.target_dataset_name, target_root)
    return target_dataset

def process_caption(annos: List[dict]):
    img_paths = []
    captions = []
    for item in annos:
        for img_path, caption in item.items():
            img_paths.append(img_path)
            captions.append(caption)

    train_data = { "img_paths": img_paths, "captions": captions }
    return train_data

if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    args.output_dir = op.join(args.output_dir, args.target_dataset_name, cur_time)

    logger = setup_logger('DFLSG', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get source target dataset
    target_data = get_data(args)

    show_dataset_info(target_data, args)
    val_img_loader, val_txt_loader = build_test_dataloader(args, target_data)

    # loading generate data
    caption_path = op.join(args.root_dir + args.target_dataset_name, 'qwen1&2&deep-icfg.json')
    caption_data = read_json(caption_path)
    train_data = process_caption(caption_data)

    model = build_model(args)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    do_train(start_epoch, args, model, evaluator, optimizer, scheduler, checkpointer, train_data, caption_data)