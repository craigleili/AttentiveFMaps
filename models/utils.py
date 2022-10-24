import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def wlstsq(A, B, w):
    if w is None:
        return torch.linalg.lstsq(A, B).solution
    else:
        assert w.dim() + 1 == A.dim() and w.shape[-1] == A.shape[-2]
        W = torch.diag_embed(w)
        return torch.linalg.lstsq(W @ A, W @ B).solution


def pdists(x, y, squared=False, eps=1e-12):
    x2 = torch.sum(x**2, dim=-1, keepdim=True)
    y2 = torch.sum(y**2, dim=-1, keepdim=True)
    dist2 = -2.0 * torch.matmul(x, torch.transpose(y, -2, -1))
    dist2 += x2
    dist2 += torch.transpose(y2, -2, -1)
    if squared:
        return dist2
    else:
        dist2 = torch.clamp(dist2, min=eps)
        return torch.sqrt(dist2)


def validate_gradient(model):
    flag = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                flag = False
            if torch.any(torch.isinf(param.grad)):
                flag = False
    return flag


def validate_tensor(x):
    if torch.is_tensor(x):
        if x.numel() > 0:
            if torch.any(torch.isnan(x)):
                return False
            if torch.any(torch.isinf(x)):
                return False
            return True
        else:
            return False
    else:
        return False


def frobenius_loss(a, b, w=None, minval=None, maxval=None):
    assert a.dim() == b.dim() == 3
    loss = (a - b)**2
    if w is not None:
        assert w.dim() == 3
        loss = loss * w
    loss = torch.sum(loss, axis=(1, 2))
    if minval is not None:
        loss = torch.clamp(loss, min=minval)
    if maxval is not None:
        loss = torch.clamp(loss, max=maxval)
    return torch.mean(loss)


def orthogonality_s_loss(fmap01):
    B, K1, K0 = fmap01.shape

    I0 = torch.eye(K0).to(fmap01)
    W0 = torch.ones_like(I0) * (float(K0) / K0**2)
    W0.fill_diagonal_(float(K0**2 - K0) / K0**2)

    I0 = torch.unsqueeze(I0, dim=0)
    W0 = torch.unsqueeze(W0, dim=0)

    loss = frobenius_loss(torch.transpose(fmap01, 1, 2) @ fmap01, I0, w=None)
    return loss


def centered_norm(x):
    shape = x.shape
    y = np.copy(x.flatten())
    for flags in [y > 0, y < 0]:
        if np.sum(flags) > 0:
            y[flags] = y[flags] / np.amax(np.abs(y[flags]))
    y = np.reshape(y, shape)
    y = (y + 1.0) / 2.0
    return y


def fmap_to_image(fmaps, nrow):
    assert torch.is_tensor(fmaps) and fmaps.dim() == 3
    fmaps = fmaps.detach().cpu().numpy()
    images = [plt.cm.RdBu(centered_norm(fmaps[i]))[..., :3] for i in range(fmaps.shape[0])]
    images = torch.from_numpy(np.stack(images, axis=0))
    grid = make_grid(images.permute(0, 3, 1, 2), nrow=nrow, normalize=False)
    grid = grid.permute(1, 2, 0).numpy()
    return grid


def bslice(x, indices):
    assert x.dim() >= 2 and indices.dim() == 2
    assert x.shape[0] == indices.shape[0]
    out = [x[i][indices[i]] for i in range(x.shape[0])]
    out = torch.stack(out, dim=0)
    return out


def diff_zoomout(evecs0, evecs1, fmap01, fmap_sizes, nnsearcher, return_all_fmaps=False):
    assert fmap01.shape[-2] == fmap01.shape[-1] == fmap_sizes[0]
    all_fmaps = [fmap01]
    for i in range(len(fmap_sizes) - 1):
        fs = fmap_sizes[i]
        corr10_mat, corr10_indices = nnsearcher(
            evecs1[..., :fs],
            evecs0[..., :fs] @ torch.transpose(all_fmaps[i], -2, -1),
        )
        fs = fmap_sizes[i + 1]
        fmap01 = wlstsq(evecs1[..., :fs], corr10_mat @ evecs0[..., :fs], None)
        all_fmaps.append(fmap01)

    if return_all_fmaps:
        return all_fmaps
    else:
        return all_fmaps[-1]


def fmap_reg(evecs0, evecs1, evals0, evals1, mass0, mass1, feats0, feats1, **kwargs):
    reg_weight = kwargs['reg_weight']

    A = torch.transpose(evecs0, 1, 2) @ (torch.unsqueeze(mass0, 2) * feats0)
    B = torch.transpose(evecs1, 1, 2) @ (torch.unsqueeze(mass1, 2) * feats1)

    AAt = A @ torch.transpose(A, 1, 2)
    delta = (torch.unsqueeze(evals1, 2) - torch.unsqueeze(evals0, 1))**2

    C_rows = list()
    for ridx in range(evals1.size(-1)):
        lhs = AAt + torch.diag_embed(reg_weight * delta[:, ridx, :])
        rhs = A @ torch.transpose(B[:, ridx:ridx + 1, :], 1, 2)
        C_ridx = torch.linalg.inv(lhs) @ rhs
        C_rows.append(torch.transpose(C_ridx, 1, 2))
    C_est = torch.cat(C_rows, dim=1)
    return C_est


FMAP_SOLVERS = {d.__name__: d for d in [fmap_reg]}


class DiffNNSearch(nn.Module):

    def __init__(self, temp_init=1.0, temp_min=1e-4):
        super().__init__()
        self.temp_min = temp_min
        self.temp = nn.parameter.Parameter(torch.tensor(temp_init, dtype=torch.float32))

    def get_temp(self):
        return torch.clamp(self.temp**2, min=self.temp_min)

    def forward(self, feats0, feats1):
        dists = pdists(feats0, feats1, squared=True)
        dists = torch.softmax(-dists / self.get_temp(), dim=-1)
        _, indices = torch.max(dists, dim=-1, keepdim=True)
        if self.training:
            asgn_diff = dists
        else:
            asgn = torch.zeros_like(dists).scatter_(dim=-1, index=indices, value=1.0)
            asgn_diff = asgn - dists.detach() + dists
        return asgn_diff, torch.squeeze(indices, dim=-1)


class PairwiseSimilarity(nn.Module):

    def __init__(self, temp_init=1.0, temp_min=1e-4, normalize_input=True):
        super().__init__()
        self.temp_min = temp_min
        self.temp = nn.parameter.Parameter(torch.tensor(temp_init, dtype=torch.float32))
        self.normalize_input = normalize_input

    def get_temp(self):
        return torch.clamp(self.temp**2, min=self.temp_min)

    def forward(self, feats0, feats1):
        if self.normalize_input:
            feats0 = F.normalize(feats0, p=2, dim=-1)
            feats1 = F.normalize(feats1, p=2, dim=-1)
        sim = feats0 @ torch.transpose(feats1, -1, -2)
        sim = torch.softmax(sim / self.get_temp(), dim=-1)
        return sim
