import os.path as osp
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.faust import ShapeDataset as FaustShapeDataset
from data.faust import ShapePairDataset as FaustShapePairDataset
from utils.io import list_files


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = None
    TEST_IDX = np.arange(44)


class ShapePairDataset(FaustShapePairDataset):

    def _init(self):
        assert self.mode.startswith('test')

        self.name_id_map = self.shape_data.get_name_id_map()
        self.pair_indices = list()
        for corr_filename in list_files(self.corr_dir, '*.map', alphanum_sort=True):
            sname0, sname1 = corr_filename[:-4].split('_')
            if sname0 == '40' or sname1 == '40':
                continue
            self.pair_indices.append((self.name_id_map[sname1], self.name_id_map[sname0]))

    def _load_corr_gt(self, sdict0, sdict1):
        pmap10 = self._load_corr_file(sdict1['name'], sdict0['name'])
        corr_gt = np.stack((pmap10, np.arange(len(pmap10))), axis=1)
        return corr_gt

    def _load_corr_file(self, sname0, sname1):
        corr_path = osp.join(self.corr_dir, f'{sname0}_{sname1}.map')
        corr = np.loadtxt(corr_path, dtype=np.int32)
        return corr - 1
