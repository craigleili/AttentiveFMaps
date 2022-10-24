import os
import os.path as osp
import sys
import math
import random
import numpy as np
import torch
import wandb
import time
import glob
import shutil
import yaml
import importlib
from torch.utils.data import DataLoader, ConcatDataset
from scipy.io import savemat
from tqdm import tqdm
from pathlib import Path

ROOT_DIR = osp.abspath(osp.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.attnfmaps import SpectralAttentionNet
from models.utils import DiffNNSearch, to_numpy, validate_gradient, validate_tensor, fmap_to_image, bslice
from models.utils import FMAP_SOLVERS, frobenius_loss, diff_zoomout
from diffusion_net.layers import DiffusionNet
from data import get_data_dirs, collate_default, prepare_batch
from data.utils import farthest_point_sampling
from utils.fmap import FM_to_p2p
from utils.io import may_create_folder
from utils.misc import incrange, validate_str, run_trainer


class Trainer(object):
    STATE_KEYS = ['feat_model', 'san_model', 'nnsearcher', 'optimizer', 'scheduler']
    STATE_KEY_VERSION = 1

    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device(f'cuda:{cfg["gpu"]}' if torch.cuda.is_available() else 'cpu')

        self.spectral_dims = incrange(cfg['loss.spectral_dim'], cfg['loss.max_spectral_dim'], cfg['loss.spectral_step_size'])

        self.feat_model = DiffusionNet(C_in=cfg['feat_model.in_channels'],
                                       C_out=cfg['feat_model.out_channels'],
                                       C_width=cfg['feat_model.block_width'],
                                       N_block=cfg['feat_model.num_blocks'],
                                       dropout=cfg['feat_model.dropout'],
                                       num_eigenbasis=cfg['feat_model.num_eigenbasis'])
        self.san_model = self._get_san_model()
        self.nnsearcher = DiffNNSearch()

        self.feat_model = self.feat_model.to(self.device)
        self.san_model = self.san_model.to(self.device)
        self.nnsearcher = self.nnsearcher.to(self.device)

        self.fmap_solver = FMAP_SOLVERS[cfg['loss.fmap_type']]

        self.dataloaders = dict()

    def _get_san_model(self):
        cfg = self.cfg
        return SpectralAttentionNet(nfeatures=cfg['attn_model.nfeatures'],
                                    nsamples=cfg['attn_model.nsamples'],
                                    spectral_dims=self.spectral_dims)

    def _init_train(self, phase='train'):
        cfg = self.cfg

        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        cfg['log_dir'] = cfg['log_dir'] + f'_{exp_time}'
        may_create_folder(cfg['log_dir'])

        self.start_epoch = 1

        parameters = list(self.feat_model.parameters())
        parameters += list(self.san_model.parameters())
        parameters += list(self.nnsearcher.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=cfg['optim.lr'], betas=(0.9, 0.99))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=cfg['optim.decay_step'],
                                                         gamma=cfg['optim.decay_gamma'])

        dsets = list()
        for data_name in cfg['data.train.types']:
            shape_cls = getattr(importlib.import_module(f'data.{data_name}'), 'ShapeDataset')
            pair_cls = getattr(importlib.import_module(f'data.{data_name}'), 'ShapePairDataset')
            shape_dir, cache_dir, corr_dir = get_data_dirs(cfg['data.root'], data_name, phase)
            dset = shape_cls(shape_dir=shape_dir,
                             cache_dir=cache_dir,
                             mode=phase,
                             aug_noise_type=cfg['data.train.noise_type'],
                             aug_noise_args=cfg['data.train.noise_args'],
                             aug_rotation_type=cfg['data.train.rotation_type'],
                             aug_rotation_args=cfg['data.train.rotation_args'],
                             aug_scaling=cfg['data.train.scaling'],
                             aug_scaling_args=cfg['data.train.scaling_args'],
                             laplacian_type=cfg['data.laplacian_type'],
                             feature_type=cfg['data.feature_type'])
            dset = pair_cls(corr_dir=corr_dir,
                            mode=phase,
                            num_corrs=cfg['data.num_corrs'],
                            use_geodists=False,
                            fmap_sizes=self.spectral_dims,
                            shape_data=dset,
                            corr_loader=None)
            dsets.append(dset)
        dsets = ConcatDataset(dsets)
        dloader = DataLoader(dsets,
                             collate_fn=collate_default,
                             batch_size=cfg['data.train.batch_size'],
                             shuffle=True,
                             num_workers=cfg['data.num_workers'],
                             pin_memory=False,
                             drop_last=False)
        self.dataloaders[phase] = dloader

        wandb.init(project=cfg['project'],
                   dir=cfg['log_dir'],
                   group=cfg['group'],
                   notes=exp_time,
                   tags=[exp_time[:8]] + cfg['data.train.types'],
                   settings=wandb.Settings(start_method='fork'))
        wandb.config.update(cfg)

        with open(osp.join(cfg['log_dir'], 'config.yml'), 'w') as fh:
            yaml.dump(cfg, fh)
        with open(osp.join(cfg['log_dir'], 'model.txt'), 'w') as fh:
            fh.write(str(self.feat_model))
            fh.write('\n')
            fh.write(str(self.san_model))
            fh.write('\n')
            fh.write(str(self.nnsearcher))

        code_backup_dir = osp.join(wandb.run.dir, 'code_src')
        for fileext in ['.py', '.yml', '.sh', 'Dockerfile']:
            for filepath in glob.glob('**/*{}'.format(fileext), recursive=True):
                may_create_folder(osp.join(code_backup_dir, Path(filepath).parent))
                shutil.copy(osp.join(ROOT_DIR, filepath), osp.join(code_backup_dir, filepath))

        if validate_str(cfg['train_ckpt']):
            self._load_ckpt(cfg['train_ckpt'])

    def _init_test(self, phase='test'):
        cfg = self.cfg

        assert validate_str(cfg['test_ckpt'])
        cfg['log_dir'] = str(Path(cfg['test_ckpt']).parent)
        may_create_folder(cfg['log_dir'])

        for data_name in cfg['data.test.types']:
            shape_cls = getattr(importlib.import_module(f'data.{data_name}'), 'ShapeDataset')
            pair_cls = getattr(importlib.import_module(f'data.{data_name}'), 'ShapePairDataset')
            shape_dir, cache_dir, corr_dir = get_data_dirs(cfg['data.root'], data_name, phase)
            dset = shape_cls(shape_dir=shape_dir,
                             cache_dir=cache_dir,
                             mode=phase,
                             aug_noise_type=None,
                             aug_noise_args=None,
                             aug_rotation_type=None,
                             aug_rotation_args=None,
                             aug_scaling=False,
                             aug_scaling_args=None,
                             laplacian_type=cfg['data.laplacian_type'],
                             feature_type=cfg['data.feature_type'])
            dset = pair_cls(corr_dir=corr_dir,
                            mode=phase,
                            num_corrs=cfg['data.num_corrs'],
                            use_geodists=False,
                            fmap_sizes=self.spectral_dims,
                            shape_data=dset,
                            corr_loader=None)
            dloader = DataLoader(
                dset,
                collate_fn=collate_default,
                batch_size=cfg['data.test.batch_size'],
                shuffle=False,
                num_workers=cfg['data.num_workers'],
                pin_memory=False,
                drop_last=False,
            )
            self.dataloaders[f'{phase}_{data_name}'] = dloader

        self._load_ckpt(cfg['test_ckpt'])

    def _train_epoch(self, epoch, phase='train'):
        cfg = self.cfg
        spectral_dims = self.spectral_dims
        nsamples = cfg['loss.refine_nsamples']
        fmit = cfg['feat_model.in_type']

        get_weight = lambda x: (spectral_dims[-1] / float(x))**2

        num_iters = len(self.dataloaders[phase])
        loader_iter = iter(self.dataloaders[phase])

        self.feat_model.train()
        self.san_model.train()
        self.nnsearcher.train()
        self.optimizer.zero_grad()
        for iter_idx in tqdm(range(num_iters), miniters=int(num_iters / 100), desc=f'Epoch: {epoch} {phase}'):
            global_step = (epoch - 1) * num_iters + (iter_idx + 1)
            log_dict = {'global_step': global_step}

            batch_data = next(loader_iter)
            batch_data = prepare_batch(batch_data, self.device)

            evecs0_dzo = batch_data['evecs0'].float()
            evecs1_dzo = batch_data['evecs1'].float()
            if nsamples > 0:
                sindices0 = farthest_point_sampling(batch_data['vertices0'].float(), nsamples, True)
                sindices1 = farthest_point_sampling(batch_data['vertices1'].float(), nsamples, True)
                evecs0_dzo = bslice(evecs0_dzo, sindices0)
                evecs1_dzo = bslice(evecs1_dzo, sindices1)

            all_feats = list()
            for pidx in range(2):
                feats = self.feat_model(x_in=batch_data[f'{fmit}{pidx}'].float(),
                                        mass=batch_data[f'mass{pidx}'].float(),
                                        L=None,
                                        evals=batch_data[f'evals{pidx}'].float(),
                                        evecs=batch_data[f'evecs{pidx}'].float(),
                                        gradX=batch_data[f'gradX{pidx}'],
                                        gradY=batch_data[f'gradY{pidx}'])
                all_feats.append(feats)
            feats0, feats1 = all_feats

            fmaps01_init = list()
            if cfg['loss.fmaps_fast_solve']:
                SD = spectral_dims[-1]
                fsd = self.fmap_solver(
                    evecs0=batch_data['evecs0'][..., :SD].float(),
                    evecs1=batch_data['evecs1'][..., :SD].float(),
                    evals0=batch_data['evals0'][..., :SD].float(),
                    evals1=batch_data['evals1'][..., :SD].float(),
                    mass0=batch_data['mass0'].float(),
                    mass1=batch_data['mass1'].float(),
                    feats0=feats0,
                    feats1=feats1,
                    reg_weight=cfg['loss.fmap_reg'],
                )
                for sidx, SD in enumerate(spectral_dims):
                    fmaps01_init.append(fsd[..., :SD, :SD])
            else:
                for sidx, SD in enumerate(spectral_dims):
                    fsd = self.fmap_solver(
                        evecs0=batch_data['evecs0'][..., :SD].float(),
                        evecs1=batch_data['evecs1'][..., :SD].float(),
                        evals0=batch_data['evals0'][..., :SD].float(),
                        evals1=batch_data['evals1'][..., :SD].float(),
                        mass0=batch_data['mass0'].float(),
                        mass1=batch_data['mass1'].float(),
                        feats0=feats0,
                        feats1=feats1,
                        reg_weight=cfg['loss.fmap_reg'],
                    )
                    fmaps01_init.append(fsd)

            attns_logits = self.san_model(xyz0=batch_data['vertices0'].float(),
                                          xyz1=batch_data['vertices1'].float(),
                                          evecs0=batch_data['evecs0'].float(),
                                          evecs1=batch_data['evecs1'].float(),
                                          evals0=batch_data['evals0'].float(),
                                          evals1=batch_data['evals1'].float(),
                                          mass0=batch_data['mass0'].float(),
                                          mass1=batch_data['mass1'].float(),
                                          feats0=feats0,
                                          feats1=feats1,
                                          fmaps01=fmaps01_init)
            attns = torch.softmax(attns_logits, dim=1)

            fmaps01_dzo = list()
            for sidx, SD in enumerate(spectral_dims):
                fsd = diff_zoomout(evecs0=evecs0_dzo,
                                   evecs1=evecs1_dzo,
                                   fmap01=fmaps01_init[sidx],
                                   fmap_sizes=[SD, spectral_dims[-1]],
                                   nnsearcher=self.nnsearcher,
                                   return_all_fmaps=False)
                fmaps01_dzo.append(fsd)

            fmap01_final = torch.sum(attns.view(attns.shape[0], attns.shape[1], 1, 1) * torch.stack(fmaps01_dzo, dim=1), dim=1)

            loss_interim = 0
            for sidx, SD in enumerate(spectral_dims):
                loss_interim = loss_interim + get_weight(SD) * frobenius_loss(
                    fmaps01_init[sidx], batch_data[f'fmap01_{SD}_gt'].float(), maxval=cfg['loss.fmap_maxval'])
            loss_interim /= len(spectral_dims)

            loss_final = frobenius_loss(fmap01_final,
                                        batch_data[f'fmap01_{spectral_dims[-1]}_gt'].float(),
                                        maxval=cfg['loss.fmap_maxval'])

            loss = cfg['loss.final_weight'] * loss_final + cfg['loss.interim_weight'] * loss_interim

            if (iter_idx + 1) % cfg['log_step'] == 0 or (iter_idx + 1) == num_iters:
                log_dict['loss'] = loss.item()
                log_dict['loss_final'] = loss_final.item()
                log_dict['loss_interim'] = loss_interim.item()
                log_dict['temp'] = self.nnsearcher.get_temp().item()
                log_dict['feats0'] = wandb.Histogram(to_numpy(feats0))
                log_dict['feats1'] = wandb.Histogram(to_numpy(feats1))

                attns_tab = np.stack((np.asarray(spectral_dims), np.mean(to_numpy(attns), axis=0)), axis=1)
                attns_tab = wandb.Table(data=attns_tab, columns=['dim', 'attn'])
                log_dict['attns'] = wandb.plot.line(attns_tab, 'dim', 'attn')

                nrow = int(math.ceil(math.sqrt(cfg['data.train.batch_size'])))
                for sidx, SD in enumerate(spectral_dims):
                    if validate_tensor(fmaps01_init[sidx]):
                        log_dict[f'fmap01_pred_{SD}'] = wandb.Image(fmap_to_image(fmaps01_init[sidx], nrow))
                    if validate_tensor(fmaps01_dzo[sidx]):
                        log_dict[f'fmap01_pred_{SD}_dzo'] = wandb.Image(fmap_to_image(fmaps01_dzo[sidx], nrow))
                    if validate_tensor(batch_data[f'fmap01_{SD}_gt']):
                        log_dict[f'fmap01_gt_{SD}'] = wandb.Image(fmap_to_image(batch_data[f'fmap01_{SD}_gt'], nrow))
                if validate_tensor(fmap01_final):
                    log_dict['fmap01_pred_final'] = wandb.Image(fmap_to_image(fmap01_final, nrow))
                wandb.log(log_dict)

            loss /= float(cfg['optim.accum_step'])
            loss.backward()

            if (iter_idx + 1) % cfg['optim.accum_step'] == 0 or (iter_idx + 1) == num_iters:
                if validate_gradient(self.feat_model) and \
                   validate_gradient(self.san_model) and \
                   validate_gradient(self.nnsearcher):
                    torch.nn.utils.clip_grad_value_(self.feat_model.parameters(), cfg['optim.grad_clip'])
                    torch.nn.utils.clip_grad_value_(self.san_model.parameters(), cfg['optim.grad_clip'])
                    torch.nn.utils.clip_grad_value_(self.nnsearcher.parameters(), cfg['optim.grad_clip'])
                    self.optimizer.step()
                else:
                    print('[!] Invalid gradients')
                self.optimizer.zero_grad()

    def _test_epoch(self, epoch=None, phase='test'):
        cfg = self.cfg
        spectral_dims = self.spectral_dims
        nsamples = cfg['loss.refine_nsamples']
        fmit = cfg['feat_model.in_type']

        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        out_root = osp.join(cfg['log_dir'], f'{phase}_{exp_time}')
        may_create_folder(out_root)

        num_iters = len(self.dataloaders[phase])
        loader_iter = iter(self.dataloaders[phase])

        if cfg['eval_mode']:
            self.feat_model.eval()
            self.san_model.eval()
        else:
            self.feat_model.train()
            self.san_model.train()
        self.nnsearcher.eval()
        for iter_idx in tqdm(range(num_iters), miniters=int(num_iters / 100), desc=phase):
            global_step = iter_idx + 1

            batch_data = next(loader_iter)
            batch_data = prepare_batch(batch_data, self.device)

            evecs0_dzo = batch_data['evecs0'].float()
            evecs1_dzo = batch_data['evecs1'].float()
            if nsamples > 0:
                sindices0 = farthest_point_sampling(batch_data['vertices0'].float(), nsamples, False)
                sindices1 = farthest_point_sampling(batch_data['vertices1'].float(), nsamples, False)
                evecs0_dzo = bslice(evecs0_dzo, sindices0)
                evecs1_dzo = bslice(evecs1_dzo, sindices1)

            with torch.no_grad():
                all_feats = list()
                for pidx in range(2):
                    feats = self.feat_model(x_in=batch_data[f'{fmit}{pidx}'].float(),
                                            mass=batch_data[f'mass{pidx}'].float(),
                                            L=None,
                                            evals=batch_data[f'evals{pidx}'].float(),
                                            evecs=batch_data[f'evecs{pidx}'].float(),
                                            gradX=batch_data[f'gradX{pidx}'],
                                            gradY=batch_data[f'gradY{pidx}'])
                    all_feats.append(feats)
                feats0, feats1 = all_feats

                fmaps01_init = list()
                if cfg['loss.fmaps_fast_solve']:
                    SD = spectral_dims[-1]
                    fsd = self.fmap_solver(
                        evecs0=batch_data['evecs0'][..., :SD].float(),
                        evecs1=batch_data['evecs1'][..., :SD].float(),
                        evals0=batch_data['evals0'][..., :SD].float(),
                        evals1=batch_data['evals1'][..., :SD].float(),
                        mass0=batch_data['mass0'].float(),
                        mass1=batch_data['mass1'].float(),
                        feats0=feats0,
                        feats1=feats1,
                        reg_weight=cfg['loss.fmap_reg'],
                    )
                    for sidx, SD in enumerate(spectral_dims):
                        fmaps01_init.append(fsd[..., :SD, :SD])
                else:
                    for sidx, SD in enumerate(spectral_dims):
                        fsd = self.fmap_solver(
                            evecs0=batch_data['evecs0'][..., :SD].float(),
                            evecs1=batch_data['evecs1'][..., :SD].float(),
                            evals0=batch_data['evals0'][..., :SD].float(),
                            evals1=batch_data['evals1'][..., :SD].float(),
                            mass0=batch_data['mass0'].float(),
                            mass1=batch_data['mass1'].float(),
                            feats0=feats0,
                            feats1=feats1,
                            reg_weight=cfg['loss.fmap_reg'],
                        )
                        fmaps01_init.append(fsd)

                attns_logits = self.san_model(xyz0=batch_data['vertices0'].float(),
                                              xyz1=batch_data['vertices1'].float(),
                                              evecs0=batch_data['evecs0'].float(),
                                              evecs1=batch_data['evecs1'].float(),
                                              evals0=batch_data['evals0'].float(),
                                              evals1=batch_data['evals1'].float(),
                                              mass0=batch_data['mass0'].float(),
                                              mass1=batch_data['mass1'].float(),
                                              feats0=feats0,
                                              feats1=feats1,
                                              fmaps01=fmaps01_init)
                attns = torch.softmax(attns_logits, dim=1)
                sidx_opt = torch.argmax(attns, dim=1, keepdim=False)
                assert sidx_opt.shape[0] == 1
                sidx_opt = sidx_opt[0].item()

                fmaps01_dzo = list()
                for sidx, SD in enumerate(spectral_dims):
                    fsd = diff_zoomout(evecs0=evecs0_dzo,
                                       evecs1=evecs1_dzo,
                                       fmap01=fmaps01_init[sidx],
                                       fmap_sizes=[SD, spectral_dims[-1]],
                                       nnsearcher=self.nnsearcher,
                                       return_all_fmaps=False)
                    fmaps01_dzo.append(fsd)

                fmap01_final = torch.sum(attns.view(attns.shape[0], attns.shape[1], 1, 1) * torch.stack(fmaps01_dzo, dim=1),
                                         dim=1)

            name0 = batch_data['name0'][0]
            name1 = batch_data['name1'][0]
            fmap01_ref = to_numpy(torch.squeeze(fmap01_final, 0))
            evecs0 = to_numpy(torch.squeeze(batch_data['evecs0'], 0))
            evecs1 = to_numpy(torch.squeeze(batch_data['evecs1'], 0))

            pmap10_ref = FM_to_p2p(fmap01_ref, evecs0, evecs1)

            to_save = {
                'id0': name0,
                'id1': name1,
                'pmap10_ref': pmap10_ref,
            }
            matpath = osp.join(out_root, f'{name0}-{name1}.mat')
            may_create_folder(str(Path(matpath).parent))
            savemat(matpath, to_save)

    def train(self):
        cfg = self.cfg

        print('Start training')
        self._init_train()

        for epoch in range(self.start_epoch, cfg['data.train.epochs'] + 1):
            print(f'Epoch: {epoch}, LR = {self.scheduler.get_last_lr()}')
            self._train_epoch(epoch)
            self.scheduler.step()
            latest_ckpt_path = self._save_ckpt(epoch, 'latest')

        cfg['test_ckpt'] = latest_ckpt_path
        print('Training finished')

    def test(self):
        cfg = self.cfg

        print('Start testing')
        self._init_test()

        for mode in self.dataloaders.keys():
            if mode.startswith('test'):
                self._test_epoch(phase=mode)

        print('Testing finished')

    def _save_ckpt(self, epoch, name=None):
        cfg = self.cfg

        state = {'epoch': epoch, 'version': self.STATE_KEY_VERSION}
        for k in self.STATE_KEYS:
            if hasattr(self, k):
                state[k] = getattr(self, k).state_dict()
        if name is None:
            filepath = osp.join(cfg['log_dir'], f'ckpt_epoch_{epoch}.pth')
        else:
            filepath = osp.join(cfg['log_dir'], f'ckpt_{name}.pth')
        torch.save(state, filepath)
        print(f'Saved checkpoint to {filepath}')

        return filepath

    def _load_ckpt(self, filepath, keys=None):
        if keys is None:
            keys = self.STATE_KEYS
        if Path(filepath).is_file():
            state = torch.load(filepath)
            if not 'version' in state or state['version'] != self.STATE_KEY_VERSION:
                raise RuntimeError(f'State version in checkpoint {filepath} does not match!')
            used_keys = list()
            for k in keys:
                if hasattr(self, k):
                    getattr(self, k).load_state_dict(state[k])
                    used_keys.append(k)
            if len(used_keys) == 0:
                raise RuntimeError(f'No state is loaded from checkpoint {filepath}!')
            print(f'Loaded checkpoint from {filepath} with keys {used_keys}')
        else:
            raise RuntimeError(f'Checkpoint {filepath} does not exist!')


if __name__ == '__main__':
    run_trainer(Trainer)
