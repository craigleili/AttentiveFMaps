import os.path as osp
import sys
import numpy as np
import itertools
from pathlib import Path
from collections import defaultdict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.faust import ShapeDataset as FaustShapeDataset
from data.faust import ShapePairDataset as FaustShapePairDataset
from utils.io import read_lines

IGNORED_CATEGORIES = ['pumpkinhulk']


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = None
    TEST_IDX = None

    def _get_file_list(self):
        if self.mode.startswith('train'):
            file_list = read_lines(osp.join(self.shape_dir, '..', 'train.txt'))
        elif self.mode.startswith('test'):
            file_list = read_lines(osp.join(self.shape_dir, '..', 'test.txt'))
        else:
            raise RuntimeError(f'Mode {self.mode} is not supported.')
        shape_list = [fn + '.obj' for fn in file_list]
        return shape_list


class ShapePairDataset(FaustShapePairDataset):

    def _init(self):
        self.name_id_map = self.shape_data.get_name_id_map()
        categories = defaultdict(list)
        for sname in self.name_id_map.keys():
            categories[sname.split('/')[0]].append(sname)
        self.pair_indices = list()
        for cname, clist in categories.items():
            if cname in IGNORED_CATEGORIES:
                continue
            for pname in itertools.combinations(clist, 2):
                self.pair_indices.append((self.name_id_map[pname[0]], self.name_id_map[pname[1]]))

    def _load_corr_gt(self, sdict0, sdict1):
        corr0 = self._load_corr_file(sdict0['name'])
        corr1 = self._load_corr_file(sdict1['name'])
        corr_gt = np.stack((corr0, corr1), axis=1)
        return corr_gt

    def _load_corr_file(self, sname):
        corr_path = osp.join(self.corr_dir, f'{sname}.vts')
        corr = np.loadtxt(corr_path, dtype=np.int32)
        return corr - 1
