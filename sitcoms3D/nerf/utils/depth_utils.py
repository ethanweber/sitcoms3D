import kornia
import numpy as np

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
