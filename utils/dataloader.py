import copy
import os

import numpy as np

from .colmap_utils import (read_cameras_binary, read_images_binary,
                           read_pointsTD_binary)


def human_to_nerf_space(obj_mesh_in, point_cloud_transform, scale_factor, colmap_rescale):
    """
    scale_factor - used by nerf so max far is 5
    """
    obj_mesh = copy.deepcopy(obj_mesh_in)
    verts = obj_mesh.vertices

    verts /= colmap_rescale

    verts = np.concatenate([verts, np.ones_like(verts[:, 0:1])], axis=1)
    verts = (verts @ point_cloud_transform.T)[:, :3]

    obj_mesh.vertices = verts
    obj_mesh.vertices /= scale_factor
    return obj_mesh


def load_colmap_cameras_from_sitcom_location(basedir):

    # --- image filenames ----
    images_data = read_images_binary(os.path.join(basedir, "colmap", "images.bin"))

    image_path_to_image_id = {}
    image_id_to_image_path = {}
    image_paths = []
    for v in images_data.values():
        image_path_to_image_id[v.name] = v.id
        image_id_to_image_path[v.id] = v.name
        image_paths.append(v.name)

    # --- intrinsics ---
    cameras_data = read_cameras_binary(os.path.join(basedir, "colmap", "cameras.bin"))
    intrinsics = []
    for image_path in image_paths:
        cam = cameras_data[image_path_to_image_id[image_path]]
        assert len(cam.params) == 3
        focal_length = cam.params[0]  # f (fx and fy)
        cx = cam.params[1]  # cx
        cy = cam.params[2]  # cy
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics.append(K)

    # --- camera_to_world (extrinsics) ---
    camera_to_world = []
    bottom_row = np.array([0, 0, 0, 1.0]).reshape(1, 4)
    for image_path in image_paths:
        image_data = images_data[image_path_to_image_id[image_path]]
        rot = image_data.qvec2rotmat()
        trans = image_data.tvec.reshape(3, 1)
        c2w = np.concatenate([np.concatenate([rot, trans], 1), bottom_row], 0)
        c2w = np.linalg.inv(c2w)
        camera_to_world.append(c2w)

    # dictionary to populate
    image_name_to_info = {}
    for image_path, intrin, c2w in zip(image_paths, intrinsics, camera_to_world):
        image_name_to_info[image_path] = {
            "intrinsics": intrin,
            "camtoworld": c2w,
        }

    return image_name_to_info
