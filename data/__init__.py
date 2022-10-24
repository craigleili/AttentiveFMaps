import sys
import os.path as osp
import numpy as np
import torch
from collections import defaultdict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_DIRS = {
    'faust': 'FAUST_r',
    'scape': 'SCAPE_r',
    'smalr': 'SMAL_r',
    'shrec19': 'SHREC_r',
    'dt4dintra': 'DT4D_r',
    'dt4dinter': 'DT4D_r',
}


def get_data_dirs(root, name, mode):
    prefix = osp.join(root, DATA_DIRS[name])
    shape_dir = osp.join(prefix, 'shapes')
    cache_dir = osp.join(prefix, 'cache_dzo')
    corr_dir = osp.join(prefix, 'correspondences')
    return shape_dir, cache_dir, corr_dir


def collate_default(data_list):
    data_dict = defaultdict(list)
    for pair_dict in data_list:
        for k, v in pair_dict.items():
            data_dict[k].append(v)
    for k in data_dict.keys():
        if k.startswith('fmap') or k.startswith('evals') or k.endswith('_sub'):
            data_dict[k] = np.stack(data_dict[k], axis=0)
    batch_size = len(data_list)
    for k, v in data_dict.items():
        assert len(v) == batch_size

    return data_dict


def prepare_batch(data_dict, device):
    for k in data_dict.keys():
        if isinstance(data_dict[k], np.ndarray):
            data_dict[k] = torch.from_numpy(data_dict[k]).to(device)
        else:
            if k.startswith('gradX') or \
               k.startswith('gradY') or \
               k.startswith('L'):
                from diffusion_net.utils import sparse_np_to_torch
                tmp_list = [sparse_np_to_torch(st).to(device) for st in data_dict[k]]
                if len(data_dict[k]) == 1:
                    data_dict[k] = torch.stack(tmp_list, dim=0)
                else:
                    data_dict[k] = tmp_list
            else:
                if isinstance(data_dict[k][0], np.ndarray):
                    tmp_list = [torch.from_numpy(st).to(device) for st in data_dict[k]]
                    if len(data_dict[k]) == 1:
                        data_dict[k] = torch.stack(tmp_list, dim=0).to(device)
                    else:
                        data_dict[k] = tmp_list

    return data_dict
