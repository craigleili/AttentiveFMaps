import os.path as osp
import sys
import numpy as np
import itertools
from pathlib import Path
from collections import defaultdict

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.dt4dintra import IGNORED_CATEGORIES
from data.dt4dintra import ShapeDataset
from data.faust import ShapePairDataset as FaustShapePairDataset
from utils.io import list_files


class ShapePairDataset(FaustShapePairDataset):

    def _init(self):
        self.name_id_map = self.shape_data.get_name_id_map()
        categories = defaultdict(list)
        for sname in self.name_id_map.keys():
            categories[sname.split('/')[0]].append(sname)
        self.pair_indices = list()
        for filename in list_files(osp.join(self.corr_dir, 'cross_category_corres'), '*.vts', alphanum_sort=False):
            cname0, cname1 = filename[:-4].split('_')
            if cname0 in IGNORED_CATEGORIES or cname1 in IGNORED_CATEGORIES:
                continue
            for sname0 in categories[cname0]:
                for sname1 in categories[cname1]:
                    self.pair_indices.append((self.name_id_map[sname0], self.name_id_map[sname1]))

    def _load_corr_gt(self, sdict0, sdict1):
        sname0 = sdict0['name']
        sname1 = sdict1['name']
        cname0 = sname0.split('/')[0]
        cname1 = sname1.split('/')[0]
        assert cname0 != cname1
        lmk01 = self._load_corr_file(f'cross_category_corres/{cname0}_{cname1}')
        corr0 = self._load_corr_file(sname0)
        corr1 = self._load_corr_file(sname1)
        corr_gt = np.stack((corr0, corr1[lmk01]), axis=1)
        return corr_gt

    def _load_corr_file(self, sname):
        corr_path = osp.join(self.corr_dir, f'{sname}.vts')
        corr = np.loadtxt(corr_path, dtype=np.int32)
        return corr - 1
