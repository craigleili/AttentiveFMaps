import os.path as osp
import sys
import math
import torch
import torch.nn as nn

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.utils import pdists
from data.utils import farthest_point_sampling


class SEBlock(nn.Module):

    def __init__(self, in_channels, reduction):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C = x.shape[:2]
        y = self.pool(x).view(B, C)
        y = self.mlp(y)
        y = x * y.view(B, C, 1)
        return y


class PointConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, use_act=True, use_se=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.use_act = use_act
        self.use_se = use_se

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm1d(out_channels) if use_norm else None
        self.act = nn.ReLU(True) if use_act else None
        self.se = SEBlock(out_channels, 4) if use_se else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_se:
            x = self.se(x)
        if self.use_act:
            x = self.act(x)
        return x


class FeatureSTN(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.mlp0 = nn.Sequential(
            PointConvBlock(in_channels, in_channels, use_norm=True, use_act=True),
            PointConvBlock(in_channels, in_channels, use_norm=True, use_act=True),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels**2),
        )

        self.register_buffer('iden', torch.unsqueeze(torch.eye(in_channels, dtype=torch.float32), dim=0))

    def forward(self, x):
        B, C = x.shape[:2]

        y = self.mlp0(x)
        y, _ = torch.max(y, dim=2)
        y = self.mlp1(y)
        y = y.view(B, C, C) + self.iden
        return y


class SpectralAttentionNet(nn.Module):

    def __init__(self, nfeatures, nsamples, spectral_dims):
        super().__init__()
        self.nfeatures = nfeatures
        self.nsamples = nsamples
        self.spectral_dims = spectral_dims

        fdim = 64
        self.mlp0 = PointConvBlock(len(spectral_dims), fdim, use_norm=True, use_act=True)
        self.fstn = FeatureSTN(fdim)
        self.mlp1 = nn.Sequential(
            PointConvBlock(fdim, nfeatures, use_norm=True, use_act=True),
            PointConvBlock(nfeatures, nfeatures, use_norm=True, use_act=False),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(nfeatures, nfeatures),
            nn.ReLU(True),
            nn.Linear(nfeatures, len(spectral_dims)),
        )

    def forward(self, xyz0, xyz1, evecs0, evecs1, evals0, evals1, mass0, mass1, feats0, feats1, fmaps01):
        B = xyz0.shape[0]

        evecs0_sub, evecs1_sub = list(), list()
        mass0_sub, mass1_sub = list(), list()
        if self.nsamples is not None and self.nsamples > 0:
            for i in range(B):
                idx0 = farthest_point_sampling(xyz0[i], self.nsamples, random_start=self.training)
                idx1 = farthest_point_sampling(xyz1[i], self.nsamples, random_start=self.training)
                evecs0_sub.append(evecs0[i, idx0, :])
                evecs1_sub.append(evecs1[i, idx1, :])
                mass0_sub.append(mass0[i, idx0])
                mass1_sub.append(mass1[i, idx1])
            evecs0_sub = torch.stack(evecs0_sub, dim=0)
            evecs1_sub = torch.stack(evecs1_sub, dim=0)
            mass0_sub = torch.stack(mass0_sub, dim=0)
            mass1_sub = torch.stack(mass1_sub, dim=0)
        else:
            evecs0_sub = evecs0
            evecs1_sub = evecs1
            mass0_sub = mass0
            mass1_sub = mass1

        residuals = list()
        for fm in fmaps01:
            k1, k0 = fm.shape[-2:]
            evecs0_fm = evecs0_sub[..., :k0] @ torch.transpose(fm.detach(), -2, -1)
            evecs1_fm = evecs1_sub[..., :k1]
            pd = pdists(evecs1_fm, evecs0_fm, squared=False)
            res, _ = torch.min(pd, dim=-1, keepdim=False)
            residuals.append(res / k1**0.5)
        residuals = torch.stack(residuals, dim=1)

        ft = self.mlp0(residuals)

        tsfm = self.fstn(ft)
        ft = torch.transpose(ft, 1, 2).contiguous() @ tsfm
        ft = torch.transpose(ft, 1, 2).contiguous()

        ft = self.mlp1(ft)
        ft = torch.sum(ft * torch.unsqueeze(mass1_sub, dim=1), dim=-1, keepdim=False) / torch.sum(
            mass1_sub, dim=-1, keepdim=True)

        attn = self.mlp2(ft)

        return attn
