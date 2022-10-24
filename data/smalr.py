import os.path as osp
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data.faust import ShapeDataset as FaustShapeDataset
from data.faust import ShapePairDataset
from utils.io import list_files, read_lines


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = None
    TEST_IDX = None

    def _get_file_list(self):
        if self.mode.startswith('train'):
            categories = ['cow', 'dog', 'fox', 'lion', 'wolf']
        elif self.mode.startswith('test'):
            categories = ['cougar', 'hippo', 'horse']
        else:
            raise RuntimeError(f'Mode {self.mode} is not supported.')

        file_list = list_files(self.shape_dir, '*.obj', alphanum_sort=True)
        shape_list = [fn for fn in file_list if fn.split('_')[0] in categories]
        return shape_list
