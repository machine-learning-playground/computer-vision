import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from model import build_model
from processor import do_train
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.options import get_args
from utils.common import get_rank, synchronize


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    args = get_args()
    set_seed(1 + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{name}")
    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(",", "\n"))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info("Total params: %2.fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    start_epoch = 1

    do_train(
        start_epoch,
        args,
        model,
        train_loader,
    )
