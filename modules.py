import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from types_ import *
from adabelief_pytorch import AdaBelief

def get_rect_mgrid(h: int, w:int, dim:int=2) -> Tensor:
    """Returns a meshgrid where size = (H, W)"""
    mgrid = torch.stack(torch.meshgrid(
        torch.linspace(-1,1, steps=h),
        torch.linspace(-1,1, steps=w)
    ), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class ImplicitGON(pl.LightningModule):

    def __init__(self, 
        img_H: int = 28,
        img_W: int = 28,
        num_latent: int = 32, 
        hidden_features: int = 256, 
        num_layers: int = 4, 
        num_channels: int = 1, # RGB=3
        w0: int = 30, 
        learning_rate: float = 1e-3, 
        batch_size: int = 64,
        eps: float = 1e-16,
        weight_decay: float = 1e-4,
        weight_decouple: bool = False, 
        **kwargs): # all other kwargs are ignored

        super().__init__()
        self.save_hyperparameters()
        
        # GON model
        img_coords = 2
        # define GON architecture, for example gon_shape = [34, 256, 256, 256, 256, 1]
        dimensions = [img_coords+self.hparams.num_latent] + [self.hparams.hidden_features]*self.hparams.num_layers + [self.hparams.num_channels]
        first_layer = SirenLayer(dimensions[0], dimensions[1], w0=self.hparams.w0, is_first=True)
        other_layers = []
        for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
            other_layers.append(SirenLayer(dim0, dim1))
        final_layer = SirenLayer(dimensions[-2], dimensions[-1], w0=self.hparams.w0, is_last=True)
        self.gon_model = nn.Sequential(first_layer, *other_layers, final_layer)
        
        # coordinates
        self.register_buffer('coords',
            torch.stack([get_rect_mgrid(self.hparams.img_H, self.hparams.img_W, 2) for _ in range(self.hparams.batch_size)]),
            persistent=False # don't save in state_dict
        )

    def forward(self, z:Tensor) -> Tensor:
        # in lightning, forward defines the prediction/inference actions
        # TODO: make this a random sample?
        z_rep = z.repeat(1, self.coords.size(1), 1).float()
        model_input = torch.cat((self.coords, z_rep), dim=-1)
        return self.gon_model(model_input)

    def embedding_to_imgs(self, embedding:Tensor) -> Tensor:
        imgs = torch.clamp(embedding, 0, 1).permute(0,2,1).reshape(-1, 
            self.hparams.num_channels, self.hparams.img_H, self.hparams.img_W)
        return imgs

    def training_step(self, batch:Tensor, batch_idx:int) -> dict:
        # sample a batch of data
        x, _ = batch
        x = x.permute(0, 2, 3, 1).reshape(self.hparams.batch_size, -1, self.hparams.num_channels)

        # compute the gradients of the inner loss with respect to zeros (gradient origin)
        z = torch.zeros(self.hparams.batch_size, 1, self.hparams.num_latent, device=self.device).requires_grad_()
        g = self.forward(z)
        L_inner = ((g - x)**2).sum(1).mean()
        z = -torch.autograd.grad(L_inner, [z], create_graph=True, retain_graph=True)[0]

        # now with z as our new latent points, optimise the data fitting loss
        g = self.forward(z)
        L_outer = ((g - x)**2).sum(1).mean()

        self.log('inner_loss', L_inner)
        self.log('outer_loss', L_outer)

        return {'loss': L_outer, 'z': z.detach() }

    def training_epoch_end(self, outputs:list) -> None:
        # plot samples
        epoch_zs = [x['z'] for x in outputs]
        grid = torchvision.utils.make_grid(
            self.embedding_to_imgs(self.sample(epoch_zs)),
            nrow=int(np.sqrt(self.hparams.batch_size)))

        self.logger.experiment.add_image('generated_images', grid, self.current_epoch) 

    def configure_optimizers(self) -> Any:
        print("Learning rate: ", self.hparams.learning_rate)
        print("eps: ", self.hparams.eps)
        print("weight decay: ", self.hparams.weight_decay)
        print("weight decouple: ", self.hparams.weight_decouple)
        optimizer = torch.optim.Adam(self.gon_model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps)
        return optimizer
        # See: https://github.com/juntang-zhuang/Adabelief-Optimizer
        # optimizer = AdaBelief(self.gon_model.parameters(), 
        #     lr=self.hparams.learning_rate, 
        #     eps=self.hparams.eps, 
        #     weight_decouple = self.hparams.weight_decouple, 
        #     weight_decay=self.hparams.weight_decay,
        #     rectify = False)
        # scheduler = ReduceLROnPlateau(optimizer, patience=3)
        # scheduler = MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'outer_loss' }

    def sample(self, zs:Tensor) -> Tensor:
        zs = torch.cat(zs, dim=0).squeeze(1).detach().cpu().numpy()
        mean = np.mean(zs, axis=0)
        cov = np.cov(zs.T)
        sample = np.random.multivariate_normal(mean, cov, size=self.hparams.batch_size)
        return self.forward(torch.tensor(sample, device=self.device).unsqueeze(1))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--img_H', type=int, default=28)
        parser.add_argument('--img_W', type=int, default=28)
        parser.add_argument('--num_latent', type=int, default=32)
        parser.add_argument('--hidden_features', type=int, default=256)
        parser.add_argument('--num_layers', type=int, default=4)
        parser.add_argument('--num_channels', type=int, default=1)
        parser.add_argument('--w0', type=int, default=30)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument_group('optimizer', 'Params for the optimizer')
        parser.add_argument('--eps', type=float, default=1e-8)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--weight_decouple', action='store_true')
        return parser

    @staticmethod
    def args_to_string(args):
        return f"{args.img_H}x{args.img_W}_L{args.num_latent}_F{args.hidden_features}x{args.num_layers}"


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

