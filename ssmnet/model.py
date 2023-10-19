# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim


import pdb
import typing


class SsmNet(nn.Module):
    def __init__(self, config, step_sec):
        """
        Args:
            - config
            - step_sec
        """
        super(SsmNet, self).__init__()
        self.config = config
        

        # -------------------------------------
        layer_l = []
        layer_l.append({'nc': 32, 'conv_shape': (5,5), 'pool_shape': (2,2)}) # --- (80,40) -> (40,20)
        layer_l.append({'nc': 32, 'conv_shape': (5,5), 'pool_shape': (2,2)}) # --- (40,20) -> (20,10)
        layer_l.append({'nc': 64, 'conv_shape': (5,5), 'pool_shape': (2,2)}) # --- (20,10) -> (10,5)
        layer_l.append({'nc': 64, 'conv_shape': (5,5), 'pool_shape': (2,2)}) # --- (10,5) -> (5,2)
        layer_l.append({'nc': 128, 'conv_shape': (5,2), 'pool_shape': (5,2)})# --- (5,2) -> (1,1)

        conv_l = []
        for idx, layer in enumerate(layer_l):
            in_nc = 1 if idx == 0 else layer_l[idx-1]['nc']
            conv_l.append(nn.Sequential(
                nn.Conv2d(in_channels=in_nc, out_channels=layer['nc'], kernel_size=layer['conv_shape'], padding='same'),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=layer['pool_shape'])
                ))
        self.conv = nn.Sequential(*conv_l)

        self.resize = 128

        # -------------------------------------
        self.attention_l = []
        for idx in range(self.config['do_nb_attention']):
            self.attention_l.append(nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=128))
        self.attention = nn.Sequential(*self.attention_l)

        # -------------------------------------
        kernel_Ldemi = int(np.round(self.config['kernel_Ldemi_sec']/step_sec))
        kernel_sigma = int(np.round(self.config['kernel_sigma_sec']/step_sec))
        M = kernel_Ldemi*2+1
        self.conv_novelty = nn.Conv2d(in_channels=1, out_channels=self.config['kernel_nb'], kernel_size=(M,M), padding='same')
        self.lin_novelty = nn.Conv2d(in_channels=self.config['kernel_nb'], out_channels=1, kernel_size=(1,1), padding='same')

        if self.config['do_kernel_init_checkerboard']:
            C_m = np.zeros((self.config['kernel_nb'], 1, M, M))
            for n in range(self.config['kernel_nb']):
                C_m[n,0,:,:] = 0.05*f_checkerboard_kernel(kernel_Ldemi, kernel_sigma-n).T
            with torch.no_grad(): 
                self.conv_novelty.weight = nn.Parameter( torch.from_numpy(C_m).float() )
                if self.config['do_kernel_freeze']: 
                    self.conv_novelty.weight.requires_grad = False


    def forward(self, feat_4m):
        """
        description:
            compute embedding
        inputs:
            feat_4m (n_batch=1, T, f=80, t=40)
        outputs:
            embedding_m (T, dim_embed)
        """

        # --- feat_4m (n_batch=1, T, f=80, t=40) -> x (T, 1, f=80, t=40)
        x = feat_4m.squeeze() # remove n_batch dimension
        x = x.unsqueeze(1) # add channel dimension (dim=1)

        x = self.conv(x)
        # --- (m, C, f=1, t=1)
        x = x.view(-1, self.resize)

        x = self.attention(x)

        x = F.tanh(x)
        embedding_m = F.normalize(x, dim=1, p=2)

        return embedding_m


    def get_ssm(self, feat_4m):
        """
        description:
            compute embedding then hat_ssm
        inputs:
            feat_4m (n_batch=1, T, f=80, t=40)
        outputs:
            hat_ssm_m (T, T)
        """

        embedding_m = self.forward(feat_4m)
        hat_ssm_m = 1 - (torch.cdist(embedding_m, embedding_m)**2)/4
        return hat_ssm_m


    def get_novelty(self, feat_4m):
        """
        description:
            compute embedding then hat_ssm then hat_prob_boundary
        inputs:
            feat_4m (n_batch=1, T, f=80, t=40)
        outputs:
            hat_prob_boundary (T,)
            ssm_hat (T, T)
        """

        # --- x: (m, f=80, t=40)
        # --- ssm_hat (m, m)
        hat_ssm_m = self.get_ssm(feat_4m)
        # --- add back n_batch and channel dimension
        y = self.conv_novelty( hat_ssm_m.unsqueeze(0).unsqueeze(1) )
        y = self.lin_novelty( y )
        y = F.sigmoid( y ) 
        hat_novelty_v = torch.diagonal( y.squeeze() )

        return hat_novelty_v, hat_ssm_m


def f_checkerboard_kernel(Ldemi=10, sigma=5):
    """
    description
    """
    f_sign = lambda value: 1 if value >= 0 else -1

    C_m = np.zeros((2*Ldemi+1, 2*Ldemi+1))
    m_v = np.arange(-Ldemi, Ldemi+1)
    n_v = np.arange(-Ldemi, Ldemi+1)
    for m in m_v:
        for n in n_v:
            if m==0 or n==0:
                C_m[m+Ldemi, n+Ldemi] = 0
            else:
                C_m[m+Ldemi, n+Ldemi] = f_sign(m) * f_sign(n) * np.exp(-(m**2+n**2)/(2*sigma**2))
    return C_m