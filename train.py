import argparse
import os
import shutil
import random
import string
import math

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import make_grid, save_image

from model import DBM
from utils import binarize, generate_id


def train(rank, args, model, optimizer, scheduler,
          dataloader, writer, epoch, detect_anomaly=False):
    model.train()
    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for step, (x, y) in enumerate(dataloader):
            v = x.to(next(model.parameters()).device)

            optimizer.zero_grad()
            loss = model(v)
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

        if rank == 0:
            writer.add_image("ground_truth/train",
                             make_grid(v[:8], 8, pad_value=1.0),
                             epoch)

            v_rec_mode, v_rec_rand = model.module.reconstruct(v)
            writer.add_image("reconstruction/train/mode",
                             make_grid(v_rec_mode[:8], 8, pad_value=1.0),
                             epoch)
            writer.add_image("reconstruction/train/rand",
                             make_grid(v_rec_rand[:8], 8, pad_value=1.0),
                             epoch)


def valid(rank, args, model, dataloader, writer, epoch):
    model.eval()
    if dataloader is not None:
        x, y = next(iter(dataloader))
        v = x.to(next(model.parameters()).device)

        v_rec_mode, v_rec_rand = model.module.reconstruct(v)

        if rank == 0:
            writer.add_image("ground_truth/test",
                             make_grid(v[:8], 8, pad_value=1.0),
                             epoch)

        if rank == 0:
            writer.add_image("reconstruction/test/mode",
                             make_grid(v_rec_mode[:8], 8, pad_value=1.0),
                             epoch)
            writer.add_image("reconstruction/test/rand",
                             make_grid(v_rec_rand[:8], 8, pad_value=1.0),
                             epoch)

    if rank == 0:
        v_mode, v_rand = model.module.sample(64)
        writer.add_image("sample/mode", make_grid(v_mode, 8, pad_value=1.0), epoch)
        writer.add_image("sample/random", make_grid(v_rand, 8, pad_value=1.0), epoch)


def get_args():
    parser = argparse.ArgumentParser(description="Deep Boltzman Machine")
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--seed', type=int, help='random seed (default: 0)', default=0)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "FashionMNIST"],
                        default="MNIST", help='dataset to be used (default: MNIST)')
    parser.add_argument("--bits", type=int, default=1, choices=[1, 8],
                        help="number of bits (default: 1)")
    parser.add_argument("--nh", type=int,
                        help="number of hidden units")
    parser.add_argument("--size", type=int, default=28,
                        help="image size (default: 28)")
    parser.add_argument("--L", type=int, default=2,
                        help="number of layers (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate (default: 1e-2)")
    parser.add_argument("--momentum", type=float, default=0,
                        help="momentum (default: 0)")
    parser.add_argument("--gamma", type=float, default=1e-3,
                        help="lr decay rate (default: 1e-3)")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="number of epochs (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="batch size (default: 1000)")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument('--device_ids', type=int, nargs='+',
                        help='list of CUDA devices (default: list(range(torch.cuda.device_count())))',
                        default=list(range(torch.cuda.device_count())))
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--detect_anomaly", action='store_true')
    parser.add_argument("--model_path", type=str, default='models')
    parser.add_argument("--distributed", action='store_true')
    args = parser.parse_args()

    return args

def set_seed(seed):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def setup(rank, args):
    port = args.port
    os.environ['MASTER_ADDR'] = 'localhost'
    while True:
        os.environ['MASTER_PORT'] = str(port)
        try:
            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=args.world_size)
        except RuntimeError:
            port += 1
        else:
            break

def cleanup():
    dist.destroy_process_group()


def run(rank, args):
    if args.distributed:
        setup(rank, args)

    if args.device_ids is None:
        args.device_ids = list(range(torch.cuda.device_count()))

    transform = [torchvision.transforms.ToTensor()]
    if args.bits == 1:
        transform.append(torchvision.transforms.Lambda(binarize))
    transform = torchvision.transforms.Compose(transform)

    # Dataset
    if args.dataset == "MNIST":
        nc = 1
        if not args.size == 28:
            raise NotImplementedError
        training_data = MNIST("dataset", download=True,
                              transform=transform)
        test_data = MNIST("dataset", train=False, download=True,
                          transform=transform)
    elif args.dataset == "FashionMNIST":
        nc = 1
        if not args.size == 28:
            raise NotImplementedError
        training_data = FashionMNIST("dataset", download=True,
                                     transform=transform)
        test_data = FashionMNIST("dataset", train=False, download=True,
                                 transform=transform)
    else:
        raise NotImplementedError

    num_workers = args.num_workers//args.world_size if args.distributed else args.num_workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    num_samples = 1000 * args.batch_size
    if args.distributed:
        num_samples //= args.world_size
    batch_size = args.batch_size//args.world_size if args.distributed else args.batch_size
    generator = torch.Generator()
    generator.manual_seed(args.seed+rank)
    train_sampler = RandomSampler(training_data, replacement=True,
                                  num_samples=num_samples, generator=generator)
    train_dataloader = DataLoader(training_data, batch_size=batch_size,
                                  sampler=train_sampler, **kwargs)

    if test_data is None:
        test_dataloader = None
    else:
        test_sampler = RandomSampler(test_data, replacement=True,
                                     num_samples=num_samples, generator=generator)
        test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                     sampler=test_sampler, **kwargs)

    # Model
    nh = nc * args.size**2 * args.bits if args.nh is None else args.nh
    model = DBM(args.size, nc, nh, args.bits, args.L).to(rank)

    model = DDP(model, device_ids=[rank]) if args.distributed \
            else nn.DataParallel(model, device_ids=args.device_ids)

    # Optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: 1 / math.sqrt(1 + args.gamma*t))

    if args.log_dir is None:
        run_id = generate_id(8)
        log_dir = f"./runs/{args.dataset}-bits:{args.bits}-L:{args.L}-nh:{nh}-lr:{args.lr}-momentum:{args.momentum}-bs:{args.batch_size}-gamma:{args.gamma}-epoch:{args.epoch}-seed:{args.seed}-{run_id}"
    else:
        log_dir = args.log_dir

    writer = None
    if rank == 0:
        # Tensorboard
        writer = SummaryWriter(log_dir)

        # make a diectory for saving models
        os.makedirs(os.path.join(log_dir, args.model_path), exist_ok=True)

    valid(rank, args, model, test_dataloader, writer, 0)

    for epoch in tqdm(range(1, args.epoch+1)):
        train(rank, args, model, optimizer, scheduler,
              train_dataloader, writer, epoch, args.detect_anomaly)
        valid(rank, args, model, test_dataloader, writer, epoch)

        # Save model
        if rank == 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, args.model_path, f"model-{epoch}.pt"))

    writer.close()

    if args.distributed:
        cleanup()


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    if args.distributed:
        mp.spawn(run,
            args=(args,),
            nprocs=args.world_size,
            join=True)
    else:
        run(0, args)
