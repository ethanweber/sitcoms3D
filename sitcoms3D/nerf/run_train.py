import os

import os
from sitcoms3D.nerf.src.opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader

from tqdm import tqdm

# models
from sitcoms3D.nerf.models.nerf import (
    PosEmbedding,
    NeRF
)
from sitcoms3D.nerf.models.rendering import (
    render_rays
)
from sitcoms3D.nerf.models.pose import LearnFocal, LearnPose

# optimizer, scheduler, visualization
from sitcoms3D.nerf.utils import (
    get_parameters,
    get_optimizer,
    get_scheduler,
    get_learning_rate
)

from sitcoms3D.nerf.datasets.sitcom3D import Sitcom3DDataset

import numpy as np

from sitcoms3D.nerf.src.metrics import psnr

from sitcoms3D.nerf.utils.visualization import get_image_summary_from_vis_data

# losses
from sitcoms3D.nerf.src.losses import loss_dict

import datetime

import cv2

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger
from einops import repeat

from sitcoms3D.nerf.datasets.ray_utils import get_ray_directions, get_rays, get_rays_batch_version
from sitcoms3D.nerf.utils import load_ckpt

from typing import Optional

# https://github.com/PyTorchLightning/pytorch-lightning/discussions/6441
# https://github.com/PyTorchLightning/pytorch-lightning/issues/3238#issuecomment-695421373

from sitcoms3D.utils.io import make_dir


class DistributedSamplerWrapper(torch.utils.data.distributed.DistributedSampler):
    def __init__(
            self, sampler,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            sampler.dataset, num_replicas, rank, shuffle)
        # typo:
        # self.sampler = Sampler
        self.sampler = sampler

    def __iter__(self):
        indices = list(self.sampler)
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return len(self.sampler)


class NeRFSystem(LightningModule):
    def __init__(self, hparams, eval_only=False):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz - 1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir - 1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                in_channels_dir=6 * hparams.N_emb_dir + 3,
                                use_view_dirs=hparams.use_view_dirs)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                                  in_channels_dir=6 * hparams.N_emb_dir + 3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min,
                                  use_view_dirs=hparams.use_view_dirs)
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]
        self.models_mm_to_train = []
        self.eval_only = eval_only
        self.init_datasets()

        self.near_min = 0.1
        self.appearance_id = None
        self.height = None

    def load_from_ckpt_path(self, ckpt_path):
        """TODO(ethan): move this elsewhere
        """
        load_ckpt(self.embedding_a, ckpt_path, model_name="embedding_a")
        load_ckpt(self.embedding_t, ckpt_path, model_name="embedding_t")
        load_ckpt(self.nerf_coarse, ckpt_path, model_name="nerf_coarse")
        load_ckpt(self.nerf_fine, ckpt_path, model_name="nerf_fine")
        load_ckpt(self.learn_f, ckpt_path, model_name="learn_f")
        load_ckpt(self.learn_p, ckpt_path, model_name="learn_p")

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    @torch.no_grad()
    def forward_rays(self,
                     rays_o,
                     rays_d,
                     a=None,
                     t=None,
                     id_=None,
                     N_samples=128,
                     N_importance=128,
                     near_min=0.1):
        """
        rays_o: (H, W, 3)
        rays_d: (H, W, 3)
        a: ()
        t: ()

        near_min is helpful to avoid any artifacts.
        however, probably don't need this for test time if during training near_min
        other than 0 is used
        """

        if isinstance(a, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                a = torch.zeros_like(self.embedding_a(torch.tensor(id_).to(rays_o.device)))
            else:
                a = self.embedding_a(torch.tensor(id_).to(rays_o.device))
        if isinstance(t, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                t = torch.zeros_like(self.embedding_t(torch.tensor(id_).to(rays_o.device)))
            else:
                t = self.embedding_t(torch.tensor(id_).to(rays_o.device))

        H, W, _ = rays_o.shape
        assert rays_o.shape == rays_d.shape
        rays_o_flat = rays_o.view(-1, 3)
        rays_d_flat = rays_d.view(-1, 3)
        nears, fars, ray_mask = self.train_dataset.get_nears_fars_from_rays_or_cam(rays_o_flat, rays_d_flat, c2w=None)
        nears[nears < near_min] = near_min

        rays = torch.cat([rays_o_flat,
                          rays_d_flat,
                          nears,
                          fars],
                         1)

        B = rays.shape[0]
        results = defaultdict(list)
        for i in tqdm(range(0, B, self.hparams.chunk)):
            kwargs = {}
            kwargs["a_embedded"] = repeat(a, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            kwargs["t_embedded"] = repeat(t, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            rendered_ray_chunks = \
                render_rays(models=self.models,
                            embeddings=self.embeddings,
                            rays=rays[i:i + self.hparams.chunk],
                            ts=None,  # ts[i:i + self.hparams.chunk],
                            N_samples=N_samples,
                            use_disp=False,
                            perturb=self.hparams.perturb,
                            noise_std=False,
                            N_importance=N_importance,
                            chunk=self.hparams.chunk,  # chunk size is effective in val mode
                            white_back=True,
                            test_time=True,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        results['img_wh'] = [W, H]
        return results

    @torch.no_grad()
    def forward_pose_K_a_t(self, c2w, K,
                           a=None,
                           t=None,
                           id_=None,
                           #    N_samples=128,
                           #    N_importance=128,
                           N_samples=128,
                           #    N_importance=64,
                           N_importance=128,
                           near_min=0.1):
        """
        pose: (3, 4)
        K: (3, 3)
        a: ()
        t: ()

        near_min is helpful to avoid any artifacts.
        however, probably don't need this for test time if during training near_min
        other than 0 is used
        """

        if isinstance(a, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                a = torch.zeros_like(self.embedding_a(torch.tensor(id_).to(c2w.device)))
            else:
                a = self.embedding_a(torch.tensor(id_).to(c2w.device))
        if isinstance(t, type(None)):
            if id_ is None:
                id_ = self.train_dataset.img_ids[0]
                t = torch.zeros_like(self.embedding_t(torch.tensor(id_).to(c2w.device)))
            else:
                t = self.embedding_t(torch.tensor(id_).to(c2w.device))

        assert c2w.shape == (3, 4)

        H, W = round(K[1, 2].item() * 2.0), round(K[0, 2].item() * 2.0)  # using "round" bc of floating precision

        directions = get_ray_directions(H, W, K)
        rays_o, rays_d = get_rays(directions, c2w)
        nears, fars, ray_mask = self.train_dataset.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=c2w)

        # print((nears < near_min).float().sum())
        nears[nears < near_min] = near_min

        # nears += 0.05 # TODO(ethan): decide to keep this or not
        rays = torch.cat([rays_o, rays_d,
                          nears,
                          fars],
                         1)

        B = rays.shape[0]
        results = defaultdict(list)
        for i in tqdm(range(0, B, self.hparams.chunk)):
            kwargs = {}
            kwargs["a_embedded"] = repeat(a, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            kwargs["t_embedded"] = repeat(t, 'c -> n c', n=len(rays[i:i + self.hparams.chunk]))
            rendered_ray_chunks = \
                render_rays(models=self.models,
                            embeddings=self.embeddings,
                            rays=rays[i:i + self.hparams.chunk],
                            ts=None,  # ts[i:i + self.hparams.chunk],
                            N_samples=N_samples,
                            use_disp=False,
                            perturb=self.hparams.perturb,
                            noise_std=False,
                            N_importance=N_importance,
                            chunk=self.hparams.chunk,  # chunk size is effective in val mode
                            white_back=True,
                            test_time=True,
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v.cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        results['img_wh'] = [W, H]
        return results

    def forward(self, rays, ts, version=None):
        """Do batched inference on rays using chunk."""
        assert version is not None
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hparams.chunk],
                            ts[i:i + self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb if version == "train" else 0,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            validation_version=version == "val")

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def init_datasets(self):
        dataset = Sitcom3DDataset
        kwargs = {'environment_dir': self.hparams.environment_dir,
                  'near_far_version': self.hparams.near_far_version}
        # kwargs['img_downscale'] = self.hparams.img_downscale
        kwargs['val_num'] = self.hparams.num_gpus
        kwargs['use_cache'] = self.hparams.use_cache
        self.train_dataset = dataset(split='train' if not self.eval_only else 'val',
                                     img_downscale=self.hparams.img_downscale, **kwargs)
        self.val_dataset = dataset(split='val', img_downscale=self.hparams.img_downscale_val, **kwargs)

        # if self.is_learning_pose():
        # NOTE(ethan): self.train_dataset.poses is all the poses, even those in the val dataset
        train_poses = torch.FloatTensor(self.train_dataset.poses)  # (N, 3, 4)
        train_poses = torch.cat([train_poses, torch.zeros_like(train_poses[:, 0:1, :])], dim=1)
        train_poses[:, 3, 3] = 1.0
        self.learn_f = LearnFocal(len(train_poses), self.hparams.learn_f).cuda()
        self.learn_p = LearnPose(len(train_poses), self.hparams.learn_r, self.hparams.learn_t,
                                 init_c2w=train_poses).cuda()
        self.models_mm = {}
        self.models_mm["learn_f"] = self.learn_f
        self.models_mm["learn_p"] = self.learn_p
        self.models_mm_to_train += [self.learn_f]
        self.models_mm_to_train += [self.learn_p]

    def setup(self, stage):
        pass
        # TODO(ethan): handle optimizer parameters here

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        self.optimizer.add_param_group({"params": get_parameters(self.models_mm_to_train), "lr": self.hparams.pose_lr})
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def is_learning_pose(self):
        return (self.hparams.learn_f or self.hparams.learn_r or self.hparams.learn_t)

    def rays_from_batch(self, batch):
        if self.is_learning_pose():
            directions = batch['directions'].squeeze()
            ts2 = batch['ts2'].squeeze()
            f = self.models_mm["learn_f"]()[ts2]
            cameras, c2w_delta = self.models_mm["learn_p"]()
            c2w = cameras[ts2]
            directions_new = directions / f
            rays_o, rays_d = get_rays_batch_version(directions_new, c2w)
            # NOTE(ethan): nerfmm won't work when near_far_version == "cam" due to ray near/far computation
            nears, fars, ray_mask = self.train_dataset.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=None)
            rays_t = batch['ts'].squeeze().unsqueeze(-1)  # hacky way to get [N, 1]
            rays = torch.cat([rays_o, rays_d, nears, fars, rays_t], 1)
            ray_mask = ray_mask.squeeze()
        else:
            rays = batch['rays'].squeeze()
            ray_mask = batch['ray_mask'].squeeze()
        return rays, ray_mask

    def training_step(self, batch, batch_nb):
        rays, ray_mask = self.rays_from_batch(batch)
        rgbs, ts = batch['rgbs'], batch['ts']
        results = self.forward(rays, ts, version="train")
        loss_d = self.loss(results, rgbs, ray_mask)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, ray_mask = self.rays_from_batch(batch)
        rgbs, ts = batch['rgbs'], batch['ts']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        results = self.forward(rays, ts, version="val")
        loss_d = self.loss(results, rgbs, ray_mask)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            if self.hparams.dataset_name == 'sitcom3D':
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams.img_wh
            vis_data = {}
            vis_data.update(results)
            vis_data["rgbs"] = rgbs
            vis_data["img_wh"] = [W, H]
            image = get_image_summary_from_vis_data(vis_data)
            self.logger.experiment.add_image('val/GT_pred_depth', image, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Run inference on an image. Uses batches
        from TrajectoryDataset.
        """

        batch_size = batch["c2w"].shape[0]
        for b in range(batch_size):
            trajectory_name = batch["trajectory_name"][b]
            folder = batch["folder"][b]
            image_idx = batch["idx"][b]
            near_min = batch["near_min"][b].float()
            filename = os.path.join(folder, trajectory_name, "images/{:06d}.png".format(image_idx))
            if os.path.exists(filename):
                print(f"skipping {filename}")
                continue
            else:
                print(f"NOT skipping {filename}")

            c2w = batch["c2w"][b][:3, :].float()
            K = batch["K"][b].float()

            H, W = round(K[1, 2].item() * 2.0), round(K[0, 2].item() * 2.0)
            if self.height is not None:
                scalar = (self.height / (K[1, 2] * 2.0))
                K[:2] *= scalar

            id_ = self.train_dataset.img_ids[0]
            if self.appearance_id is None:
                a_embedded = torch.zeros_like(self.embedding_a(torch.tensor(id_).to(c2w.device)))
            else:
                a_embedded = self.embedding_a(torch.tensor(self.appearance_id).to(c2w.device))
            t_embedded = torch.zeros_like(self.embedding_t(torch.tensor(id_).to(c2w.device)))

            if near_min == -1:
                near_min = self.near_min
            results = self.forward_pose_K_a_t(c2w, K, a_embedded, t_embedded, near_min=near_min)

            img = results["rgb_fine_static"]
            img_w, img_h = results["img_wh"]
            image = img.view(img_h, img_w, 3).cpu().numpy()
            image = (image * 255.0).astype(np.uint8)
            depth_mean = results['depth_fine_static_exp'].view(img_h, img_w).cpu().numpy()
            depth_median = results['depth_fine_static_med'].view(img_h, img_w).cpu().numpy()
            if self.height is not None:
                image = cv2.resize(image, (W, H))
                depth_mean = cv2.resize(depth_mean, (W, H), interpolation=cv2.INTER_NEAREST)
                depth_median = cv2.resize(depth_median, (W, H), interpolation=cv2.INTER_NEAREST)

            # filename = goat.pjoin(folder, trajectory_name, "images/{:06d}.png".format(image_idx))
            filename = make_dir(filename)
            cv2.imwrite(filename, image[:, :, ::-1])
            filename = os.path.join(folder, trajectory_name, "depth_mean/{:06d}.npy".format(image_idx))
            filename = make_dir(filename)
            np.save(filename, depth_mean)

            filename = os.path.join(folder, trajectory_name, "depth_median/{:06d}.npy".format(image_idx))
            filename = make_dir(filename)
            np.save(filename, depth_median)


def main(hparams):
    save_dir = os.path.join(hparams.environment_dir, "runs")

    if hparams.resume_name:
        timedatestring = hparams.resume_name
    else:
        timedatestring = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

    # print(hparams.exp_name)

    # following pytorch lightning convention here

    logger = TestTubeLogger(save_dir=save_dir,
                            name=timedatestring,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)
    version = logger.experiment.version
    if not isinstance(version, int):
        version = 0

    if hparams.resume_name:
        assert hparams.ckpt_path is not None

    # following pytorch lightning convention here
    dir_path = os.path.join(save_dir, timedatestring, f"version_{version}")
    system = NeRFSystem(hparams)

    checkpoint_filepath = os.path.join(f'{dir_path}/ckpts')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_filepath,
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=-1)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      val_check_interval=int(1000),  # run val every int(X) batches
                      benchmark=True,
                      #   profiler="simple" if hparams.num_gpus == 1 else None
                      )

    trainer.fit(system)


if __name__ == '__main__':

    hparams = get_opts()
    main(hparams)
