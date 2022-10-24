import os.path as osp
import random
import numpy as np
import torch
import yaml
import omegaconf
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from pathlib import Path


class KNNSearch(object):
    DTYPE = np.float32
    NJOBS = 4

    def __init__(self, data):
        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):
        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, n_jobs=self.NJOBS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices

    def query_ball(self, kpt, radius):
        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, n_jobs=self.NJOBS)
        return nnindices


def validate_str(x):
    return x is not None and x != ''


def hashing(arr, M):
    assert isinstance(arr, np.ndarray) and arr.ndim == 2
    N, D = arr.shape

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        hash_vec += arr[:, d] * M**d
    return hash_vec


def omegaconf_to_dotdict(hparams):

    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if v is None:
                res[k] = v
            elif isinstance(v, omegaconf.DictConfig):
                res.update({k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()})
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
            elif isinstance(v, omegaconf.ListConfig):
                res[k] = omegaconf.OmegaConf.to_container(v, resolve=True)
            else:
                raise RuntimeError('The type of {} is not supported.'.format(type(v)))
        return res

    return _to_dot_dict(hparams)


def incrange(start, end, step):
    assert step > 0
    res = [start]
    if start <= end:
        while res[-1] + step <= end:
            res.append(res[-1] + step)
    else:
        while res[-1] - step >= end:
            res.append(res[-1] - step)
    return res


def seeding(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def run_trainer(trainer_cls):
    cfg_cli = OmegaConf.from_cli()

    assert cfg_cli.run_mode is not None
    if cfg_cli.run_mode == 'train':
        assert cfg_cli.run_cfg is not None
        cfg = OmegaConf.merge(
            OmegaConf.load(cfg_cli.run_cfg),
            cfg_cli,
        )
        OmegaConf.resolve(cfg)
        cfg = omegaconf_to_dotdict(cfg)
        seeding(cfg['seed'])
        trainer = trainer_cls(cfg)
        trainer.train()
        trainer.test()
    elif cfg_cli.run_mode == 'test':
        assert cfg_cli.run_ckpt is not None
        log_dir = str(Path(cfg_cli.run_ckpt).parent)
        cfg = OmegaConf.merge(
            OmegaConf.load(osp.join(log_dir, 'config.yml')),
            cfg_cli,
        )
        OmegaConf.resolve(cfg)
        cfg = omegaconf_to_dotdict(cfg)
        cfg['test_ckpt'] = cfg_cli.run_ckpt
        seeding(cfg['seed'])
        trainer = trainer_cls(cfg)
        trainer.test()
    else:
        raise RuntimeError(f'Mode {cfg_cli.run_mode} is not supported.')
