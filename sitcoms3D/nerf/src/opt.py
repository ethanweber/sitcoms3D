import os
import numpy as np
import torch
import random
import configargparse
import glob

from sitcoms3D.utils.io import load_from_json, make_dir, get_absolute_path

def str2bool(v):
    assert type(v) is str
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')


def select_gpus(gpus_arg):
    # so that default gpu is one of the selected, instead of 0
    if len(gpus_arg) > 0:
        gpus_arg_str = gpus_arg
        if gpus_arg_str[-1] == ",":  # removing trailing comma ,
            gpus_arg_str = gpus_arg_str[:-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg_str
        gpus = list(range(len(gpus_arg_str.split(','))))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        gpus = []
    print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    return gpus


def get_ckpt_path_dir_path_from_environment_dir(environment_dir):
    choices = load_from_json(os.path.join(environment_dir, "choices.json"))
    if choices["ckpt_path"] == "":
        latest_run_folder = sorted(glob.glob(os.path.join(environment_dir, "runs", "*", "version_0")))[-1]
        ckpt_paths = glob.glob(os.path.join(latest_run_folder, "ckpts", "*.ckpt"))
        ckpt_paths_steps = [int(x[x.find("step=") + 5:x.find(".ckpt")]) for x in ckpt_paths]
        ckpt_path = ckpt_paths[np.argsort(np.array(ckpt_paths_steps))[-1]]
    else:
        ckpt_path = choices["ckpt_path"]
    if choices["directory"] == "":
        # TODO(ethan): change the folder name back to "outputs"
        dir_path = make_dir(os.path.join(environment_dir, "outputs/"))
    else:
        dir_path = make_dir(choices["directory"])
    return ckpt_path, dir_path


def get_parser():
    parser = configargparse.ArgumentParser()

    parser.add('-c', '--config', default="sitcoms3D/nerf/configs/default.txt", is_config_file=True, help='config file path')

    parser.add_argument('--environment_dir', type=str, required=False, help='directory to the environment')
    parser.add_argument('--dataset_name', type=str, default='sitcom3D',
                        choices=['blender', 'sitcom3D'],
                        help='which dataset to train/val')
    # for blender
    parser.add_argument('--data_perturb', nargs="+", type=str, default=[],
                        help='''what perturbation to add to data.
                                Available choices: [], ["color"], ["occ"] or ["color", "occ"]
                             ''')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    # for sitcom3D
    parser.add_argument('--img_downscale', type=int, default=8,
                        help='how much to downscale the images for sitcom3D dataset')
    parser.add_argument('--img_downscale_val', type=int, default=4,
                        help='how much to downscale the images for sitcom3D dataset')
    parser.add_argument('--use_cache', type=str2bool, default="True",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    # original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=1500,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', type=str2bool, default="True",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', type=str2bool, default="True",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=16,
                        help='number of embeddings for transient objects')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='minimum color variance for each ray')
    parser.add_argument('--N_cams', type=int, default=159,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')

    # swap out the model
    parser.add_argument('--backbone', type=str, default="nerf", help="choose model arch to use")

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for train and val datasets')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='directory to resme from')
    parser.add_argument('--dir_path', type=str, default=None, help='')
    parser.add_argument('--log_folder', type=str, default=None,
                        help='where to store logging information during training or evaluation')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    # params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--refresh_every', type=int, default=1,
                        help='print the progress bar every X steps')

    ###########################
    parser.add_argument('--seed', default=0, type=int, help='the random seed for reproducibility')

    # some helper commands
    parser.add_argument('--gpus', default=None, type=str,
                        help='What gpus to use, E.g., \"0,1,2\" (same as CUDA_VISIBLE_DEVICES)')

    # coefficients for the extra losses
    parser.add_argument('--nerfw_loss', type=float, default=1.0, help='nerfw loss coeff')
    parser.add_argument('--depth_loss', type=float, default=0.0, help='depth loss coeff')
    parser.add_argument('--sparse_loss', type=float, default=0.0, help='sparsity loss coeff')
    parser.add_argument('--sparse_ray_loss', type=float, default=0.0, help='sparse ray loss coeff')
    # TODO(ethan): add support for a different version w/ proper logging
    parser.add_argument('--sparse_ray_version', type=str, default="soft",
                        choices=["soft", "hard"], help='which version of the sparse ray loss to use')

    # for ray sampling
    parser.add_argument('--near_far_version', type=str, default="box",
                        choices=["box", "cam"], help='either use box or camera and distance from it')

    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--step', type=str, default=None)

    # add parameters to use the viewing direction or not
    # TODO(ethan): add support for this flag
    parser.add_argument('--use_view_dirs', type=str2bool, default='True',
                        help='whether to condition on the viewing direction or not')

    # arguments for learning pose (nerf-- style)
    parser.add_argument('--learn_f', default=False, action="store_true",
                        help='whether to learn residual focal length')
    parser.add_argument('--learn_r', default=False, action="store_true",
                        help='whether to learn residual rotation')
    parser.add_argument('--learn_t', default=False, action="store_true",
                        help='whether to learn residual translation')
    # NOTE(ethan): this will be how much to update the f, r, t parameters (lr for pose)
    parser.add_argument('--pose_lr', type=float, default=5e-4, help='pose parameters learning rate')

    ########################
    # params for eval.py
#     parser.add_argument('--scene_name', type=str, default='test',
#                         help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'])
    parser.add_argument('--video_format', type=str, default='gif',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')
    parser.add_argument('--traj_name', type=str, default=None,
                        help='specified trajectory')
    parser.add_argument('--render_point_cloud', default=False, action='store_true',
                        help='this will skip nerf and render the point cloud')

    parser.add_argument('--ids', type=str, action=None, help='list of ids to move between')

    return parser


def get_opts():
    parser = get_parser()
    args = parser.parse_args()
    if args.environment_dir:
        args.environment_dir = get_absolute_path(args.environment_dir)
    if args.gpus:  # set gpus if specified
        gpus = select_gpus(args.gpus)
        args.num_gpus = len(gpus)

    # for reproducability
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    return args


def get_opts_from_args_str(args_str):
    parser = get_parser()
    args = args_str.split(" ")
    args = parser.parse_args(args)
    if args.environment_dir:
        args.environment_dir = get_absolute_path(args.environment_dir)
    if args.gpus:  # set gpus if specified
        gpus = select_gpus(args.gpus)
        args.num_gpus = len(gpus)
    return args
