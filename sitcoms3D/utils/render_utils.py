import copy

import kornia
import numpy as np

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender

import torch
from torch import nn
import cv2
from sitcoms3D.utils.io import get_absolute_path
import numpy as np

# nerf
from sitcoms3D.nerf.src.opt import get_opts_from_args_str
from sitcoms3D.nerf.run_train import NeRFSystem

def image_depth_from_results(results):
    img = results["rgb_fine_static"]
    # img = results["rgb_fine"]
    img_w, img_h = results["img_wh"]
    image = img.view(img_h, img_w, 3).cpu().numpy()
    image = (image * 255.0).astype(np.uint8)
    depth = results['depth_fine_static_med'].view(img_h, img_w).cpu().numpy()
    return image, depth


def Z_to_raydepth(depth, intrinsics):
    """ray depths (like what we get from nerf) from a z value depth map
    """

    # first, backproject the points
    # print(depth.shape)
    H, W = depth.shape
    points = kornia.create_meshgrid(H, W, normalized_coordinates=False)[0].reshape(-1, 2).cpu().numpy()  # x,y
    points = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=1)
    points = np.transpose(np.linalg.inv(intrinsics) @ np.transpose(points))
    points = np.reshape(points, (H, W, 3))
    points = points * depth[:, :, None]
    raydepth = np.sqrt(np.sum(points**2, axis=-1))
    return raydepth


def raydepth_to_Z(depth, intrinsics):
    """z buffer from ray depths (what we get from nerf)
    """
    H, W = depth.shape
    points = kornia.create_meshgrid(H, W, normalized_coordinates=False)[0].reshape(-1, 2).cpu().numpy()  # x,y
    points = np.concatenate([points, np.ones_like(points[:, 0:1])], axis=1)
    points = np.transpose(np.linalg.inv(intrinsics) @ np.transpose(points))
    denominator = np.sum(points**2.0, axis=1) ** 0.5
    Z = depth / np.reshape(denominator, (H, W))
    return Z


def render_human(obj_mesh_data, pose_in, K, alphaMode="OPAQUE", baseColorFactors=None):
    """
    obj_meshes - obj_mesh or list of obj_meshes
    """
    pose = copy.deepcopy(pose_in)
    if pose.shape == (3, 4):
        pose = np.concatenate([pose, np.zeros_like(pose[0:1])], axis=0)
        pose[3, 3] = 1.0

    # bg_color is important for the alpha mask to work properly
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5),
                           bg_color=(1.0, 1.0, 1.0, 0.0))

    if not isinstance(obj_mesh_data, list):
        obj_mesh_data = [obj_mesh_data]

    for i in range(len(obj_mesh_data)):
        obj_mesh = obj_mesh_data[i]
        if baseColorFactors is None:
            baseColorFactor = (0.5, 0.5, 0.5, 1.0)
        else:
            baseColorFactor = baseColorFactors[i]
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode=alphaMode,
            baseColorFactor=baseColorFactor)

        mesh = pyrender.Mesh.from_trimesh(obj_mesh, material=material)
        scene.add(mesh)

    camera = pyrender.camera.IntrinsicsCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2]
    )

    scene.add(camera, pose=pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = copy.deepcopy(pose)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    W, H = round(K[0, 2] * 2), round(K[1, 2] * 2)

    r = pyrender.OffscreenRenderer(W, H)
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    depth = Z_to_raydepth(depth, K)
    alpha = color[:, :, 3].astype(np.float32) / 255.0
    image = color[:, :, :3]

    r.delete()
    return image, depth, alpha

class NeRFWrapper(nn.Module):

    def __init__(self, environment_dir, ckpt_path, use_cache=True):
        super().__init__()
        config = get_absolute_path("sitcoms3D/nerf/configs/default.txt")
        self.args = get_opts_from_args_str(f"-c {config} --environment_dir {environment_dir} --use_cache {use_cache} --chunk 50000")
        system = NeRFSystem(self.args, eval_only=True)
        system.load_from_ckpt_path(ckpt_path)
        system.cuda()
        system.eval()
        self.system = system

    def forward(self, pose, Kin, height=None, id_=None, near_min=0.1):
        c2w = torch.from_numpy(pose[:3]).float().to(self.system.device)
        K = torch.from_numpy(Kin).float().to(self.system.device)

        H, W = round(K[1, 2].item() * 2.0), round(K[0, 2].item() * 2.0)

        if height is not None:
            scalar = (height / K[0, 2]) / 2
            K[:2] *= scalar

        # resize after the downscaling
        results = self.system.forward_pose_K_a_t(c2w, K, id_=id_, near_min=near_min)
        image, depth = image_depth_from_results(results)
        image = cv2.resize(image, (W, H))
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        return image, depth