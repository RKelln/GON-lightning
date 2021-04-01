from argparse import ArgumentParser
from pathlib import Path
import glob
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from modules import ImplicitGON
from dataset import LmdbDataset

def get_next_version(version_path):
        existing_versions = glob.glob(f"{version_path}*/")
        if len(existing_versions) == 0:
            return 0
        
        # don't just count files, in case versions have been deleted
        existing_versions.sort()
        last = existing_versions[-1].rstrip(os.path.sep)
        last_ver = last.rsplit('_v', 1) # returns ['versionpath', 'two digit number']
        if len(last_ver) == 1:
            # no version number assume no other versions
            return 1
        else: 
            return int(last_ver[1]) + 1


def train():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--experiment', type=str, default='')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--logs', type=str, default='logs')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser = ImplicitGON.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    pl.seed_everything(args.seed)

    # image transform
    # transform = transforms.Compose([transforms.ToTensor(), 
    #                                 transforms.Normalize((0.1307,), (0.3081,))])
    transform = transforms.ToTensor()

    # load datasets
    if args.dataset.lower() == 'mnist':
        dataset = MNIST('data', train=True, download=True, transform=transform)
        dataset_name = 'mnist'
    elif args.dataset.lower() == 'fashion':
        dataset = FashionMNIST('data', train=True, download=True, transform=transform)
        dataset_name = 'fashion'
    else:
        dataset_path = Path(args.dataset)
        dataset = LmdbDataset(dataset_path, transform=transform, batch_size=args.batch_size)
        dataset_name = dataset_path.stem
        # TODO: set img size and channels from dataset
        args.img_W, args.img_H = dataset.img_size # (W,H)
        args.num_channels = dataset.num_channels
        print(str(dataset))

    # NOTE: lightning does sampler for you
    data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # validate args
    assert(args.num_channels == 1 or args.num_channels == 3)
    assert(args.batch_size >= 0)
    assert(args.num_workers >= 0)

    # version
    version_str = f"{dataset_name}_{ImplicitGON.args_to_string(args)}"
    version_path = os.path.join(args.logs, args.experiment, version_str)
    version = get_next_version(version_path)
    version_str += f"_v{version:02d}"
    version_path = os.path.join(args.logs, args.experiment, version_str)
    print(args.experiment, version_str)

    # ------------
    # logger
    # ------------

    logger = TensorBoardLogger(args.logs, 
        name=args.experiment,
        version=version_str)
    
    # ------------
    # model
    # ------------
    if args.checkpoint_path is None:
        model = ImplicitGON(**(vars(args)))
    else:
        model = ImplicitGON.load_from_checkpoint(args.checkpoint_path,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size) 

    # ------------
    # callbacks
    # ------------
    callbacks = []
    # best and last checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='outer_loss',
        dirpath=version_path,
        filename='{epoch:03d}-{outer_loss:.2f}',
        save_top_k=3,
        save_last=True,
        mode='min',
    )
    callbacks.append(checkpoint_callback)
    # learning rate monitor (in some cases)
    #callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # ------------
    # training
    # ------------
    if args.checkpoint_path is None:
        args.logger = logger
        args.callbacks = callbacks
        trainer = pl.Trainer.from_argparse_args(args)
    else:
        print(f"Continuing training from {args.checkpoint_path}")
        trainer = pl.Trainer(resume_from_checkpoint=args.checkpoint_path,
            gpus=args.gpus,
            logger=logger,
            callbacks=callbacks
        )

    trainer.fit(model, data_loader)


if __name__ == '__main__':
    train()