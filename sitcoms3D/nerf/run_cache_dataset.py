import sys
import os

import argparse
from sitcoms3D.nerf.datasets.sitcom3D import Sitcom3DDataset
import numpy as np
import os
import pickle
from sitcoms3D.nerf.src.opt import get_opts
import pprint
from multiprocessing import Pool

from sitcoms3D.utils.io import make_dir, get_absolute_path

def cache_environment_dir(args, environment_dir):
    print(f'Preparing cache for scale {args.img_downscale} and {environment_dir}...')

    # NOTE(ethan): notice the split here is no longer 'train'
    dataset = Sitcom3DDataset(environment_dir, 'test', args.img_downscale, near_far_version=args.near_far_version)
    

    path = dataset.cache_dir
    path = make_dir(path)

    # save img ids   
    with open(os.path.join(path, 'img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    # save img paths
    with open(os.path.join(path, 'id_to_image_path.pkl'), 'wb') as f:
        pickle.dump(dataset.id_to_image_path, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(path, f'Ks.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)
    # save scene points
#     np.save(os.path.join(path, 'xyz_world.npy'), dataset.xyz_world)
#     np.save(os.path.join(path, 'rgb_world.npy'),  dataset.rgb_world)
    np.save(os.path.join(path, 'scale_factor.npy'), dataset.scale_factor)
    np.save(os.path.join(path, 'bbox.npy'), dataset.bbox)
    np.save(os.path.join(path, 'pointcloudT.npy'), dataset.pointcloudT)
    
    # save poses
    np.save(os.path.join(path, 'poses.npy'),
            dataset.poses)
    # save rays and rgbs
#     np.save(os.path.join(path, f'rays{args.img_downscale}.npy'),
#             dataset.all_rays.numpy())
#     np.save(os.path.join(path, f'directions{args.img_downscale}.npy'),
#             dataset.all_directions.numpy())
#     np.save(os.path.join(path, f'rgbs{args.img_downscale}.npy'),
#             dataset.all_rgbs.numpy())
    print(f"Data cache saved to {path}!")


if __name__ == '__main__':
    args = get_opts()
    environment_dirs = os.listdir(get_absolute_path("data/sparse_reconstruction_and_nerf_data"))
    pprint.pprint(environment_dirs)
    arguments = [(args, os.path.join("data/sparse_reconstruction_and_nerf_data", x)) for x in environment_dirs]
    with Pool(16) as p:
        _ = p.starmap(cache_environment_dir, arguments)