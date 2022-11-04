import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch
import copy
import plotly.graph_objects as go
from sitcoms3D.nerf.utils.depth_utils import Z_to_raydepth

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh


def np_visualize_depth(depth, normalization=None, cmap=cv2.COLORMAP_JET):
    """
    """
    x = np.nan_to_num(depth)  # change nan to 0
    mask = x == 0
    # try:
    if not normalization:
        mi = np.min(x[x != 0.0])  # get minimum depth
        ma = np.max(x[x != 0.0])
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    else:
        x /= normalization
    # except:
    #     pass
    x = (255 * x).astype("uint8")
    x_ = cv2.applyColorMap(x, cmap)
    x_[mask] = 0
    assert x_.dtype == "uint8"
    x_ = Image.fromarray(x_)
    return x_


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x_ = np_visualize_depth(x, cmap=cmap)
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def image_depth_from_results(results):
    """
    """
    img = results["rgb_fine_static"]
    # img = results["rgb_fine"]
    img_w, img_h = results["img_wh"]
    image = img.view(img_h, img_w, 3).cpu().numpy()
    image = (image * 255.0).astype(np.uint8)
    depth = results['depth_fine_static_med'].view(img_h, img_w).cpu().numpy()
    return image, depth


def get_image_summary_from_vis_data(vis_data):
    """Returns an image summary.
    """
    W, H = vis_data["img_wh"]
    img_gt = vis_data["rgbs"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    img_st = vis_data['rgb_fine_static'].view(H, W, 3).permute(2, 0, 1).cpu()
    img_tr = vis_data['rgb_fine_transient'].view(H, W, 3).permute(2, 0, 1).cpu()
    img = vis_data['rgb_fine'].view(H, W, 3).permute(2, 0, 1).cpu()

    beta = visualize_depth(vis_data['beta'].view(H, W))
    # TODO(ethan): find a better spot beta

    # print(vis_data.keys())

    depth_st = visualize_depth(vis_data['depth_fine_static_med'].view(H, W))
    depth_tr = visualize_depth(vis_data['depth_fine_transient'].view(H, W))
    depth = visualize_depth(vis_data['depth_fine'].view(H, W))

    # build the image
    zeros = torch.zeros_like(img_gt)
    row1 = torch.cat([img_gt, img, img_st, img_tr], dim=2)
    row2 = torch.cat([beta, depth, depth_st, depth_tr], dim=2)
    image = torch.cat([row1, row2], dim=1)

    return image


def composite(foreground_image,
              foreground_depth,
              background_image,
              background_depth,
              foreground_alpha=None,
              erode=None):
    """Composite a foreground image with a background image.
    Args:
        foreground_image (H, W, 3)
        foreground_depth (H, W)
        background_image (H, W, 3)
        background_depth (H, W)
        foreground_alpha (H, W)
    """
    import mediapy as media
    a = foreground_alpha[:, :, None]
    foreground_image_on_background = (foreground_image * a + background_image * (1 - a)).astype("uint8")
    human_mask = ((background_depth == 0.0) | ((foreground_depth != 0.0) *
                  (foreground_depth < background_depth))).astype("uint8")
    # media.show_image(human_mask)
    if erode:
        kernel = np.ones((3, 3), np.uint8)
        human_mask = cv2.dilate(human_mask, kernel, iterations=erode)
        # media.show_image(human_mask)
        human_mask = cv2.erode(human_mask, kernel, iterations=erode)
        # media.show_image(human_mask)
    background_mask = (1 - human_mask)
    image = foreground_image_on_background * (1 - background_mask[:, :, None]) + background_image * background_mask[:, :, None]
    return image


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

    # color = color.astype(np.float32) / 255.0
    # valid_mask = (depth > 0)[:,:,None]
    # color = (color[:, :, :3] * valid_mask + (1 - valid_mask) * color)
    # color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.ALL_WIREFRAME)
    r.delete()
    return image, depth, alpha


def render_bunny(obj_mesh_data, pose_in, K, alphaMode="OPAQUE", baseColorFactors=None):
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

    # color = color.astype(np.float32) / 255.0
    # valid_mask = (depth > 0)[:,:,None]
    # color = (color[:, :, :3] * valid_mask + (1 - valid_mask) * color)
    # color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.ALL_WIREFRAME)
    r.delete()
    return image, depth, alpha


def render_scenemesh(obj_mesh_data, pose, K):
    """
    obj_meshes - obj_mesh or list of obj_meshes
    """
    scene = pyrender.Scene()

    if not isinstance(obj_mesh_data, list):
        obj_mesh_data = [obj_mesh_data]

    for obj_mesh in obj_mesh_data:
        mesh = pyrender.Mesh.from_trimesh(obj_mesh)
        scene.add(mesh)

    camera = pyrender.camera.IntrinsicsCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2]
    )

    scene.add(camera, pose=pose)

    W, H = round(K[0, 2] * 2), round(K[1, 2] * 2)

    r = pyrender.OffscreenRenderer(W, H)
    # https://github.com/mmatl/pyrender/issues/51#issuecomment-577266913
    # flat shading
    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    depth = Z_to_raydepth(depth, K)
    r.delete()
    return color, depth

# def get_image_summary_from_vis_data_with_obj(vis_data, obj_mesh_data, pose, K, W, H):
#     """Composite obj into the nerf rendering.
#     Human compositing.
#     """

#     # convert things to numpy
#     img = (vis_data['rgb_fine'].view(H, W, 3).cpu().numpy() * 255.0).astype(np.uint8)
#     img_st = (vis_data['rgb_fine_static'].view(H, W, 3).cpu().numpy() * 255.0).astype(np.uint8)
#     depth = vis_data['depth_fine'].view(H, W).cpu().numpy()
#     depth_st = vis_data['depth_fine_static'].view(H, W).cpu().numpy()

#     color_h, depth_h = render_human(obj_mesh_data, pose, K)

#     mask = ((depth_h != 0.0) * (depth_h < depth))[:, :, None] * 1.0
#     mask_st = ((depth_h != 0.0) * (depth_h < depth_st))[:, :, None] * 1.0

#     img_with_obj = (img * (1 - mask) + color_h * mask).astype(np.uint8)
#     img_with_obj_st = (img_st * (1 - mask_st) + color_h * mask_st).astype(np.uint8)

#     depthvis = (visualize_depth(vis_data['depth_fine'].view(H, W)).permute(
#         1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     depthvis_st = (visualize_depth(vis_data['depth_fine_static'].view(H, W)
#                                    ).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     depthvis_with_obj = (depthvis * (1 - mask) + color_h * mask).astype(np.uint8)
#     depthvis_with_obj_st = (depthvis_st * (1 - mask_st) + color_h * mask_st).astype(np.uint8)

#     image = np.vstack([
#         np.hstack([img_with_obj, img_with_obj_st]),
#         np.hstack([depthvis_with_obj, depthvis_with_obj_st])
#     ])
#     return image

# # plotly helper functions


def data_point_cloud(dataset, point_size=1, skip=1, opacity=1.0):
    """The initial point cloud.
    """
    points = dataset.xyz_world
    c = dataset.rgb_world / 255.0
    colors = goat.plotly_utils.c_to_colors(c)
    return goat.plotly_utils.get_scatter3d(
        x=points[:, 0][::skip],
        y=points[:, 1][::skip],
        z=points[:, 2][::skip],
        colors=colors[::skip],
        name="colmap point cloud",
        size=point_size,
        opacity=opacity
    )


def data_cameras(dataset, ids=None, size=7.5, skip=1, alpha=1, color=(1, 0, 0), viewdir_scalar=0.1, include_lines=True, name="cameas"):
    """For ploting
    """
    if ids is None:
        ids = dataset.img_ids
    poses = np.array([dataset.get_pose(id_) for id_ in ids])
    viewdirs = np.array([dataset.get_viewdir(id_) for id_ in ids]) * viewdir_scalar

    lines = np.stack([poses[:, 0:3, 3], poses[:, 0:3, 3] + viewdirs], axis=1)

    c = goat.plotly_utils.color_str(color)
    data = []
    if include_lines:
        data += goat.plotly_utils.get_line_segments_from_lines(lines[::skip], color=c, marker_color=c, marker_size=size)
    data += [go.Scatter3d(
        x=poses[:, 0, 3][::skip],
        y=poses[:, 1, 3][::skip],
        z=poses[:, 2, 3][::skip],
        mode='markers',
        name=name,
        marker=dict(color='rgba(0, 0, 0, 1)',
                    size=size)
    )]
    return data


def get_density_figure(dataset, xyz, mask, mask_transient, vertices, triangles, voxel_size):
    data = []

    SKIP = 3

    # scene points
    data.append(
        data_point_cloud(dataset, point_size=1, skip=3)
    )

    # density field
    DENSITY_SKIP = 5
    data.append(
        go.Scatter3d(
            x=xyz[mask, 0][::DENSITY_SKIP],
            y=xyz[mask, 1][::DENSITY_SKIP],
            z=xyz[mask, 2][::DENSITY_SKIP],
            mode='markers',
            name='static density',
            marker=dict(size=voxel_size, color="rgba(0.0, 0.0, 0.0, 1.0)", opacity=0.5)
        )
    )

    # density field
    DENSITY_SKIP = 5
    data.append(
        go.Scatter3d(
            x=xyz[mask_transient, 0][::DENSITY_SKIP],
            y=xyz[mask_transient, 1][::DENSITY_SKIP],
            z=xyz[mask_transient, 2][::DENSITY_SKIP],
            mode='markers',
            name='transient density',
            marker=dict(size=voxel_size, color="rgba(0.0, 0.0, 0.0, 1.0)", opacity=0.5)
        )
    )

    # # marching cubes
    # DENSITY_SKIP = 5
    # data.append(
    #     go.Mesh3d(
    #         x=vertices[:,0],
    #         y=vertices[:,1],
    #         z=vertices[:,2],
    #         # i, j and k give the vertices of triangles
    #         i=triangles[:,0],
    #         j=triangles[:,1],
    #         k=triangles[:,2],
    #         name='marching cubes',
    #         showlegend=True
    #     )
    # )

    layout = go.Layout(
        autosize=False,
        width=1000,
        height=1000,
        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        scene=go.layout.Scene(
            aspectmode="data",
            camera={
                "up": {"x": 0, "y": -1, "z": 0},
                "eye": {"x": -1.25, "y": -1.25, "z": -1.25}
            }
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig
