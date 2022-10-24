import argparse
import os
import os.path as osp
import sys
import numpy as np
import pickle
import multiprocessing
import importlib
from joblib import Parallel, delayed
from scipy.io import loadmat, savemat
from pathlib import Path

ROOT_DIR = osp.abspath(osp.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data import get_data_dirs, DATA_DIRS
from data.utils import load_geodist
from utils.io import list_folders, may_create_folder


def load_corr_preds(filepath):
    data = loadmat(filepath)
    pmap10_ref = np.squeeze(np.asarray(data['pmap10_ref'], dtype=np.int32))
    return pmap10_ref


def run_exp(cfg, test_root, out_root):
    exp_name = Path(test_root).name
    data_type = exp_name.split('_')[1]
    assert data_type in DATA_DIRS.keys()
    mode = 'test'

    if Path(out_root).is_dir() and not Path(out_root + '.pkl').is_file():
        return

    if Path(out_root + '.pkl').is_file():
        print(f'[*] {exp_name} already evaluated: load from pkl...')
        with open(out_root + '.pkl', 'rb') as fh:
            saved = pickle.load(fh)
            all_pair_ids = saved['pair_ids']
            all_geoerrs_ref = saved['geoerrs_ref']
    else:
        shape_cls = getattr(importlib.import_module(f'data.{data_type}'), 'ShapeDataset')
        pair_cls = getattr(importlib.import_module(f'data.{data_type}'), 'ShapePairDataset')
        shape_dir, cache_dir, corr_dir = get_data_dirs(cfg.data_root, data_type, mode)
        dset = shape_cls(shape_dir=shape_dir,
                         cache_dir=cache_dir,
                         mode=mode,
                         aug_noise_type=None,
                         aug_noise_args=None,
                         aug_rotation_type=None,
                         aug_rotation_args=None,
                         aug_scaling=False,
                         aug_scaling_args=None,
                         laplacian_type=cfg.laplacian_type,
                         feature_type=None)
        dset = pair_cls(corr_dir=corr_dir,
                        mode=mode,
                        num_corrs=cfg.num_corrs,
                        use_geodists=False,
                        fmap_sizes=[10],
                        shape_data=dset,
                        corr_loader=None)

        may_create_folder(out_root)

        all_pair_ids = list()
        all_geoerrs_ref = list()
        for pid in range(len(dset)):
            pair_dict = dset[pid]
            id0, id1 = pair_dict['name0'], pair_dict['name1']
            pair_filename = f'{id0}-{id1}.mat'
            evecs0 = pair_dict['evecs0']
            evecs1 = pair_dict['evecs1']
            num_verts0 = evecs0.shape[0]
            num_verts1 = evecs1.shape[0]

            geodist0, sqrt_area0 = load_geodist(osp.join(shape_dir, '..', 'geodist', '{}.mat'.format(id0)))

            pmap10_ref = load_corr_preds(osp.join(test_root, pair_filename))

            corr0 = pair_dict['corr_gt'][:, 0]
            corr1 = pair_dict['corr_gt'][:, 1]

            match010_ref = np.stack([corr0, pmap10_ref[corr1]], axis=-1)
            match010_ref = np.ravel_multi_index(match010_ref.T, dims=[num_verts0, num_verts0])
            geoerrs_ref = np.take(geodist0, match010_ref) / sqrt_area0
            geoerrs_ref = np.squeeze(geoerrs_ref)
            all_geoerrs_ref.append(geoerrs_ref)

            all_pair_ids.append((id0, id1))

            to_save = {'pmap10_ref': np.asarray(pmap10_ref, dtype=np.int32)}
            matpath = osp.join(out_root, '{}.mat'.format(pair_filename[:-4]))
            may_create_folder(str(Path(matpath).parent))
            savemat(matpath, to_save)

        with open(out_root + '.pkl', 'wb') as fh:
            to_save = {
                'pair_ids': all_pair_ids,
                'geoerrs_ref': all_geoerrs_ref,
            }
            pickle.dump(to_save, fh)

    all_geoerrs_ref = np.concatenate(all_geoerrs_ref)

    with open(out_root + '.csv', 'w') as fh:
        fh.write('MeanGeoErrRef,{:.4f}\n'.format(np.mean(all_geoerrs_ref)))


def run_model(cfg, model_root):
    if not Path(model_root).is_dir():
        return

    print(f'Evaluating {Path(model_root).name}')

    for folder_name in list_folders(model_root):
        if not folder_name.startswith('test_'):
            continue
        if folder_name.endswith('_eval'):
            continue
        test_root = osp.join(model_root, folder_name)
        out_root = test_root + '_eval'
        run_exp(cfg, test_root, out_root)

    print(f'Finished {Path(model_root).name}')


def run(cfg):
    num_threads = min(len(cfg.test_roots), 3)
    Parallel(n_jobs=num_threads)(delayed(run_model)(cfg, test_root) for test_root in cfg.test_roots)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_roots', nargs='+')
    parser.add_argument('--data_root', type=str, default='exp/data')
    parser.add_argument('--laplacian_type', type=str, default='mesh')
    parser.add_argument('--num_corrs', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_arguments()
    run(cfg)
