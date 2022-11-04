import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

import cv2
import kornia
import copy

from sitcoms3D.utils.io import load_from_json


def width_height_from_intr(K):
    cx, cy = K[0, 2], K[1, 2]
    W = int(2 * cx)
    H = int(2 * cy)
    return W, H


def near_far_from_points(xyz_world_h, w2c):
    """
    """
    xyz_cam = (xyz_world_h @ w2c.T)[:, :3]  # xyz in the ith cam coordinate
    xyz_cam = xyz_cam[xyz_cam[:, 2] > 0]  # filter out points that lie behind the cam
    near = np.percentile(xyz_cam[:, 2], 0.1)
    far = np.percentile(xyz_cam[:, 2], 99.9)
    return near, far


def torch_ray_intersect_aabb(rays_o, rays_d, aabb):
    """
    Args:
        rays_o (torch.tensor): (batch_size, 3)
        rays_d (torch.tensor): (batch_size, 3)
        aabb (torch.tensor): (2, 3)
            This is [min point (x,y,z), max point (x,y,z)]
    """

    # avoid divide by zero
    dir_fraction = 1.0 / (rays_d + 1e-6)

    # x
    t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

    nears = torch.max(torch.cat([
        torch.minimum(t1, t2),
        torch.minimum(t3, t4),
        torch.minimum(t5, t6)], dim=1),
        dim=1).values
    fars = torch.min(torch.cat([
        torch.maximum(t1, t2),
        torch.maximum(t3, t4),
        torch.maximum(t5, t6)], dim=1),
        dim=1).values

    # TODO(ethan): handle two cases
    # fars < 0: means the ray is behind the camera
    # nears > fars: means no intersection
    # currently going to assert the valid cases
    # assert torch.all(fars > 0)
    # assert torch.all(fars > nears)
    # if not torch.all(fars > 0) or not torch.all(fars > nears):
    #     print("OUT OF BOUNDS!\n\n")

    mask = (fars > nears).float() * (fars > 0).float()
    # nears, fars = nears[mask], fars[mask]
    # set nears to be zero
    nears[nears < 0.0] = 0.0

    nears = nears.unsqueeze(-1)
    fars = fars.unsqueeze(-1)
    mask = mask.unsqueeze(-1)
    return nears, fars, mask


class RenderDataset(Dataset):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__()
        pass

    def get_bbox_pointcloudT(self):
        """Get the bounding box and the point cloud transformation.
        Here we want to bound the scene with a tight bbox.
        The bbox_params.json file comes from three.js editor online.
        """
        filename = os.path.join(self.environment_dir, 'threejs.json')
        assert os.path.exists(filename)

        data = load_from_json(filename)

        # point cloud transformation
        # pointcloudT = np.array(data['object']['children'][0]['children'][0]["matrix"]).reshape(4, 4).T
        pointcloudT = np.array(data['object']['children'][0]["matrix"]).reshape(4, 4).T
        assert pointcloudT[3, 3] == 1.0

        # bbox transformation
        bbox_T = np.array(data['object']['children'][1]["matrix"]).reshape(4, 4).T
        w, h, d = data["geometries"][1]["width"], data["geometries"][1]["height"], data["geometries"][1]["depth"]
        temp = np.array([w, h, d]) / 2.0
        bbox = np.array([-temp, temp])
        bbox = np.concatenate([bbox, np.ones_like(bbox[:, 0:1])], axis=1)
        bbox = (bbox_T @ bbox.T).T[:, 0:3]

        return bbox, pointcloudT

    def get_img(self, id_, img_downscale=None):
        img = Image.open(os.path.join(self.environment_dir, 'images',
                                      self.id_to_image_path[id_])).convert('RGB')
        img_w, img_h = img.size
        if not img_downscale:
            img_downscale = self.img_downscale
        if img_downscale > 1:
            img_w = img_w // img_downscale
            img_h = img_h // img_downscale
            img = img.resize((img_w, img_h), Image.LANCZOS)
        img = np.array(img)
        return img

    def get_pose(self, id_, homogeneous=False):
        pose = copy.deepcopy(self.poses_dict[id_])
        if homogeneous:
            pose = np.concatenate([pose, np.zeros_like(pose[:1])], -2)
            pose[3, 3] = 1
        return pose

    def get_HW(self, id_):
        K = copy.deepcopy(self.Ks[id_])
        H, W = round(K[1, 2] * 2.0), round(K[0, 2] * 2.0)
        return H, W

    def get_K(self, id_):
        K = copy.deepcopy(self.Ks[id_])
        return K

    def get_mask(self, id_):
        # human mask
        # TODO(ethan): confirm that this works!
        panoptic = np.array(Image.open(os.path.join(self.environment_dir, 'segmentations', 'thing',
                                                '%s.png' % self.id_to_image_path[id_][:-4])))
        mask = np.zeros_like(panoptic)
        mask[panoptic == 1] = 1 # 1 is the person class
        K = self.get_K(id_)
        img_w, img_h = width_height_from_intr(K)
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def get_image_paths(self):
        """
        """

        imgs = []
        imdata = read_images_binary(os.path.join(self.environment_dir, 'colmap', 'images.bin'))
        image_paths = set([v.name for v in imdata.values()])
        filter_image_paths = set(self.image_filenames)
        if len(filter_image_paths) > 0:
            image_paths = list(image_paths.intersection(filter_image_paths))
            assert len(image_paths) == len(filter_image_paths)

        for img_path in image_paths:
            temp = os.path.join(self.environment_dir, 'images', img_path)
            assert os.path.exists(temp)
        image_paths = list(image_paths)
        image_paths = sorted(image_paths)

        return image_paths

    def get_fl(self, id_):
        """Returns the focal length.
        """
        K = self.get_K(id_)
        assert K[0, 0] == K[1, 1]
        return float(K[0, 0])

    def get_fov(self, id_):
        fl = self.get_fl(id_)
        H, W = self.get_HW(id_)
        fov = 2 * np.arctan(float(H) / (2.0 * fl))
        fov *= 180.0 / np.pi
        return fov

    def get_aa(self, id_):
        """Return the angle axis (aa) representation for the c2w.
        """
        c2w = self.get_pose(id_)
        rotation_matrix = torch.from_numpy(c2w[:3, :3]).reshape(1, 3, 3).clone()
        aa = kornia.rotation_matrix_to_angle_axis(rotation_matrix)[0].numpy()
        return aa

    def get_viewdir(self, id_):
        """
        """
        c2w = self.get_pose(id_)
        viewdir = np.array((0, 0, -1)) @ c2w[:3, :3].T
        return viewdir

    def get_xyz(self, id_):
        c2w = self.get_pose(id_)
        xyz = c2w[:3, 3]
        return xyz

    def get_ids(self, version="filtered"):
        """Choose the version of ids to get.
        """
        image_paths = self.dataset_paths[version]
        ids = [self.image_path_to_id[image_path] for image_path in image_paths]
        return ids


class Sitcom3DDataset(RenderDataset):
    def __init__(self, environment_dir, split='train', img_downscale=1,
                 val_num=1, use_cache=False, near_far_version="cam", read_points=True):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        super().__init__()

        self.environment_dir = environment_dir
        self.image_filenames = os.listdir(os.path.join(self.environment_dir, 'images'))

        self.read_points = read_points

        # make nerf folder if it doesn't exist
        self.cache_dir = os.path.join(self.environment_dir, f'cache/{near_far_version}/')

        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
#        if split == 'val': # image downscale=1 will cause OOM in val mode
#            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num)  # at least 1
        self.use_cache = use_cache
        self.near_far_version = near_far_version
        self.define_transforms()

        # print(f"Using near_far_version: {self.near_far_version}")

        self.read_meta()
        self.white_back = False

    def get_nears_fars_from_rays_or_cam(self, rays_o, rays_d, c2w=None):
        """
        """
        if self.near_far_version == "box":
            nears, fars, ray_mask = torch_ray_intersect_aabb(rays_o, rays_d, aabb=self.bbox)
        elif self.near_far_version == "cam":
            assert not isinstance(c2w, type(None))
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            c2w_h = np.concatenate([c2w.cpu().numpy(), np.zeros((1, 4))], -2)
            c2w_h[3, 3] = 1
            c2w_h[:, 1:3] *= -1
            w2c = np.linalg.inv(c2w_h)
            near, far = near_far_from_points(xyz_world_h, w2c)
            nears = near * torch.ones_like(rays_o[:, 0:1])
            fars = far * torch.ones_like(rays_o[:, 0:1])
            ray_mask = torch.ones_like(fars)
        else:
            raise NotImplementedError("")
        return nears, fars, ray_mask

    def read_meta(self):

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.cache_dir, 'img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.cache_dir, 'id_to_image_path.pkl'), 'rb') as f:
                self.id_to_image_path = pickle.load(f)
        else:
            image_paths = self.get_image_paths()

            imdata = read_images_binary(os.path.join(self.environment_dir, 'colmap', 'images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.id_to_image_path = {}  # {id: filename}
            for image_path in image_paths:
                id_ = img_path_to_id[image_path]
                self.id_to_image_path[id_] = image_path
                self.img_ids += [id_]

        self.image_path_to_id = {}
        for id_, image_path in self.id_to_image_path.items():
            self.image_path_to_id[image_path] = id_

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.cache_dir, f'Ks.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
                for id_ in self.Ks.keys():
                    self.Ks[id_][:2] /= self.img_downscale
        else:
            self.Ks = {}  # {id: K}
            camdata = read_cameras_binary(os.path.join(self.environment_dir, 'colmap', 'cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                assert len(cam.params) == 3
                img_w, img_h = int(cam.params[1] * 2), int(cam.params[2] * 2)
                img_w_, img_h_ = img_w // self.img_downscale, img_h // self.img_downscale
                K[0, 0] = cam.params[0] * img_w_ / img_w  # fx
                K[1, 1] = cam.params[0] * img_h_ / img_h  # fy
                K[0, 2] = cam.params[1] * img_w_ / img_w  # cx
                K[1, 2] = cam.params[2] * img_h_ / img_h  # cy
                K[2, 2] = 1
                assert K[0, 0] == K[1, 1], "maybe check if img_downscale is too small?"
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.cache_dir, 'poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0)  # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3]  # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            # self.xyz_world = np.load(os.path.join(self.cache_dir, 'xyz_world.npy'))
            # self.rgb_world = np.load(os.path.join(self.cache_dir, 'rgb_world.npy'))
            self.scale_factor = float(np.load(os.path.join(self.cache_dir, 'scale_factor.npy')))
            self.bbox = np.load(os.path.join(self.cache_dir, 'bbox.npy'))
            self.pointcloudT = np.load(os.path.join(self.cache_dir, 'pointcloudT.npy'))
        else:
            if self.read_points:
                pts3d = read_points3d_binary(os.path.join(self.environment_dir, 'colmap', 'points3D.bin'))
                self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
                self.rgb_world = np.array([pts3d[p_id].rgb for p_id in pts3d])
                xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # max_num_points = 10000
            # subsample_steps = len(self.xyz_world) // max_num_points
            # self.xyz_world = self.xyz_world[::subsample_steps]
            # self.rgb_world = self.rgb_world[::subsample_steps]
            # print(self.xyz_world.shape)

            if self.near_far_version == "box":
                self.bbox, self.pointcloudT = self.get_bbox_pointcloudT()
                self.poses = np.concatenate([self.poses, np.zeros_like(self.poses[:, 0:1, :])], axis=1)
                self.poses[:, 3, 3] = 1.0  # (N_images, 4, 4)
                self.poses = self.pointcloudT @ self.poses
                self.poses = self.poses[:, :3]
                if self.read_points:
                    self.xyz_world = (xyz_world_h @ self.pointcloudT.T)[:, :3]
                temp = self.bbox[0] - self.bbox[1]
                max_far = np.sqrt(np.dot(temp.T, temp))
            elif self.near_far_version == "cam":
                camera_translations = self.poses[:, :3, 3]
                all_points = np.concatenate([self.xyz_world, camera_translations], axis=0)
                vol_point_min = np.min(all_points, axis=0)  # volume point minimum
                vol_point_max = np.max(all_points, axis=0)  # volume point maximum
                self.bbox = np.array([vol_point_min, vol_point_max])
                # Compute near and far bounds for each image individually
                nears, fars = {}, {}  # {id_: distance}
                for i, id_ in enumerate(self.img_ids):
                    xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3]  # xyz in the ith cam coordinate
                    xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]  # filter out points that lie behind the cam
                    nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                    fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)
                max_far = np.fromiter(fars.values(), np.float32).max()

            print("max_far")
            print(max_far)
            self.scale_factor = max_far / 5  # so that the max far is scaled to 5
            self.poses[..., 3] /= self.scale_factor
            self.xyz_world /= self.scale_factor
            self.bbox /= self.scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}

        self.id_to_idx = {}
        for idx, id_ in enumerate(self.img_ids):
            self.id_to_idx[id_] = idx

        if self.split == 'train':  # create buffer of all rays and rgb data
            if self.use_cache:
                raise ValueError("Not setup to use cache in training mode.")
                # all_rays = np.load(os.path.join(self.cache_dir, f'rays{self.img_downscale}.npy'))
                # self.all_rays = torch.from_numpy(all_rays)
                # all_directions = np.load(os.path.join(self.cache_dir, f'directions{self.img_downscale}.npy'))
                # self.all_directions = torch.from_numpy(all_directions)
                # all_rgbs = np.load(os.path.join(self.cache_dir, f'rgbs{self.img_downscale}.npy'))
                # self.all_rgbs = torch.from_numpy(all_rgbs)
            else:
                self.all_rays = []
                self.all_directions = []
                self.all_rgbs = []
                self.all_masks = []
                self.all_ray_mask = []
                for id_ in tqdm(self.img_ids):
                    c2w = torch.FloatTensor(self.get_pose(id_))
                    img = self.get_img(id_)
                    img_h, img_w, _ = img.shape
                    mask = self.get_mask(id_)
                    img = self.transform(img)  # (3, h, w)
                    img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                    mask = self.transform(mask)
                    mask = mask.view(-1)

                    directions = get_ray_directions(img_h, img_w, self.get_K(id_))
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)
                    rays_t2 = self.id_to_idx[id_] * torch.ones(len(rays_o), 1)

                    nears, fars, ray_mask = self.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=c2w)

                    self.all_rgbs += [img]
                    self.all_masks += [mask]
                    temp = torch.cat([rays_o, rays_d,
                                      nears,
                                      fars,
                                      rays_t,
                                      ray_mask],
                                     1)
                    self.all_rays += [temp]
                    self.all_directions += [torch.cat([directions.view(-1, 3),
                                                       nears,
                                                       fars,
                                                       rays_t,
                                                       rays_t2,
                                                       ray_mask],
                                                      1)]
                    self.all_ray_mask += [ray_mask]

                self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
                self.all_directions = torch.cat(self.all_directions, 0)  # ((N_images-1)*h*w, 5)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
                self.all_masks = torch.cat(self.all_masks, 0)
                self.all_ray_mask = torch.cat(self.all_ray_mask, 0)

                # throw away the pixels that belong to people
                valid_ray_all_masks = self.all_masks < 1
                # throw away the pixels that are outside the bounding box
                valid_rays_all_ray_mask = self.all_ray_mask.squeeze() > 0
                valid_rays = valid_ray_all_masks & valid_rays_all_ray_mask

                self.all_rays = self.all_rays[valid_rays]
                self.all_directions = self.all_directions[valid_rays]
                self.all_rgbs = self.all_rgbs[valid_rays]
                self.all_masks = self.all_masks[valid_rays]
                self.all_ray_mask = self.all_ray_mask[valid_rays]

        if self.split in ['val', 'test_train']:  # use the first image as val image (also in train)
            self.val_id = self.image_path_to_id[self.image_filenames[0]]
        else:  # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return len(self.img_ids)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx_or_id):
        if self.split == 'train':  # use data in the buffers
            idx = idx_or_id
            sample = {'rays': self.all_rays[idx, :8],
                      'directions': self.all_directions[idx, :3],
                      'ts': self.all_rays[idx, 8].long(),
                      'ts2': self.all_directions[idx, 6].long(),
                      'rgbs': self.all_rgbs[idx],
                      'ray_mask': self.all_rays[idx, 9]}
        elif self.split in ['val', 'test_train']:
            sample = {}
            if self.split == 'val':
                id_ = self.val_id
            else:
                id_ = idx_or_id
            sample['c2w'] = c2w = torch.FloatTensor(self.get_pose(id_))

            img = self.get_img(id_)
            img_h, img_w, _ = img.shape
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            sample['rgbs'] = img

            mask = self.get_mask(id_)
            # sample['masks'] = self.transform(mask).view(-1)
            sample['human_mask'] = self.transform(mask).view(-1) < 1.0

            directions = get_ray_directions(img_h, img_w, self.get_K(id_))
            rays_o, rays_d = get_rays(directions, c2w)
            nears, fars, ray_mask = self.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=c2w)
            rays = torch.cat([rays_o, rays_d,
                              nears,
                              fars],
                             1)  # (h*w, 8)
            sample['rays'] = rays
            sample['ray_mask'] = ray_mask.squeeze()
            sample['directions'] = directions.view(-1, 3)
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['ts2'] = self.id_to_idx[id_] * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])
        else:
            idx = idx_or_id
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx][:3])
            K = self.Ks_test[idx]
            H, W = round(K[1, 2] * 2.0), round(K[0, 2] * 2.0)  # using "round" bc of floating precision
            directions = get_ray_directions(H, W, K)
            rays_o, rays_d = get_rays(directions, c2w)
            # near, far = 0, 5
            nears, fars, ray_mask = self.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=c2w)
            rays = torch.cat([rays_o, rays_d,
                              nears,
                              fars],
                             1)
            id_ = self.appearance_test[idx]
            sample['rays'] = rays
            sample['ray_mask'] = ray_mask.squeeze()
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['ts2'] = self.id_to_idx[id_] * torch.ones(len(rays), dtype=torch.long)

            img = self.get_img(id_)
            img_h, img_w, _ = img.shape
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            sample['rgbs'] = img

            mask = self.get_mask(id_)
            sample['human_mask'] = self.transform(mask).view(-1) < 1.0

            sample['img_wh'] = torch.LongTensor([W, H])

        return sample
