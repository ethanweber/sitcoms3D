import copy

import kornia
import numpy as np

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh


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
