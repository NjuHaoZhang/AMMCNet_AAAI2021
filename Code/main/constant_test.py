import os,time
import argparse
import configparser
#
import torch
#
from ..utils import utils
from .params.const_params import ConstConfig as const_params
from ..dataset import test_dataset


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        # if name in self.__dict__:
        #     raise self.ConstError("Can't change const.{}".format(name))
        # # if not name.isupper():
        # #     raise self.ConstCaseError('const name {} is not all uppercase'.format(name))

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            print(name, value)
            _str += '\t{}\t{}\n'.format(name, value)

        return _str


def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')

    # device setting
    parser.add_argument('--node', type=str, default='admin-node',
                        help='the node of gpu.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='the device id of gpu.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='set the batch size, default is 1.')
    parser.add_argument('-n', '--num_workers', type=int, default=16,
                        help='set the num_workers, default is 16.')

    # exp_setting
    parser.add_argument('--mode', type=str,
                        default='testing',
                        help='exp mode: training or testing')
    parser.add_argument('--exp_tag', type=str,
                        default='',
                        help='identify a exp for train and test_load')
    parser.add_argument('--helper_tag', type=str,
                        default='test_single',
                        help='helper_tag')
    parser.add_argument('--net_tag', type=str,
                        default='',
                        help='net_tag')
    parser.add_argument('--data_type', type=str,
                        default='rgb_op',
                        help='rgb or op or two_stream')
    parser.add_argument('-d', '--dataset_name', type=str,
                        help='the name of dataset.', default=None)
    parser.add_argument('--data_dir', type=str,
                        default='/p300/dataset',
                        help='data_dir')
    parser.add_argument('--which_ds', type=str,
                        default='normal',
                        help='which_ds')
    parser.add_argument('--num_his', type=int, # for clip_length
                        default=4,
                        help='num_his')

    # for testing
    parser.add_argument('--evaluate', type=str, default='compute_auc',
                        help='the evaluation metric, default is compute_auc')

    parser.add_argument('--loss_name', type=str, default='psnr',
                        help='the evaluation metric, default is compute_auc')

    parser.add_argument('--metric_name', type=str, default='img_pred_fea_comm_rgb_auc',
                        help='the evaluation metric, default is compute_auc')

    parser.add_argument('--normalize', type=bool,
                        default=True,
                        help='normalize each sub_video psnr or mse')

    parser.add_argument('--ckptfile', type=str,
                        default=None,
                        help='')

    parser.add_argument('--start_step', type=int,
                        default=0,
                        help='start_step')


    return parser.parse_args()


# ========================================================================================== #
const = Const() 

args = parser_args()
config_const = const_params
root_dir = config_const.root_dir
dataset_dir = config_const.dataset_dir
save_root = config_const.save_root
ped2_net_params = config_const.ped2_net_params
avenue_net_params = config_const.avenue_net_params
shanghaitech_net_params = config_const.shanghaitech_net_params
ped2_ckpt = config_const.ped2_ckpt
avenue_ckpt = config_const.avenue_ckpt
shanghaitech_ckpt = config_const.shanghaitech_ckpt
# ========================================================================================== #
# common setting
const.config_const = config_const
# log
const.test_log_config_path = config_const.test_log_config_path
#
exp_tag_log_save = config_const.exp_tag_log_save
# net_init_params
net_params_map = config_const.net_params_map
# ds_init_params
ds_params_map = config_const.ds_params_map
#
const.exp_tag_log_save = exp_tag_log_save
const.exp_tag = args.exp_tag
# ================================================================================ #
#
# load form train_save_dir and args
# (1) for testing dataset init
# params = utils.load_params(ds_params_map, const.exp_tag)
const.data_dir = dataset_dir #
const.which_ds = "normal"  # normal, lmdb
args.data_type = "rgb_op"

# please set your params
if args.dataset_name:
    const.dataset_name = args.dataset_name #
if args.data_type:
    const.data_type = args.data_type #
if args.which_ds:
    const.which_ds = args.which_ds #
if args.num_his:
    const.num_his = args.num_his #
if args.data_dir:
    const.data_dir = args.data_dir

tmp_map = {
    "avenue": os.path.join(save_root, "avenue"),
    "ped2": os.path.join(save_root, "ped2"),
    "shanghaitech": os.path.join(save_root, "shanghaitech"),
}
test_save_dir = tmp_map[const.dataset_name]


# const.net_params_map = net_params_map # net_init 
# init_params = utils.load_params(net_params_map, const.exp_tag)
tmp_map = {
    "ped2": ped2_net_params,
    "avenue": avenue_net_params,
    "shanghaitech": shanghaitech_net_params,
}
const.net_params_map = tmp_map[const.dataset_name]
init_params = utils.load_params_test(tmp_map[const.dataset_name])
# print("init_params: ", init_params)
const.net_tag = init_params.net_tag
const.data_type = init_params.data_type
const.in_channel = init_params.in_channel
const.out_channel = init_params.out_channel
const.embed_dim = init_params.embed_dim
const.n_embed = init_params.n_embed
const.k = init_params.k
# const.ds_params_map = ds_params_map # ds_init 
#
const.helper_tag = args.helper_tag
const.mode = args.mode
const.start_step = int(args.start_step) # 

if const.which_ds == "normal":
    path_rgb = os.path.join(const.data_dir,
        "{}/{}/frames".format(const.dataset_name, const.mode))  #
    path_op = os.path.join(const.data_dir,
        "{}/optical_flow/{}/frames/flow".format(const.dataset_name,
        const.mode))  #
elif const.which_ds == "lmdb":
    path_rgb = os.path.join(const.data_dir, "lmdb_final", const.dataset_name,
                            'rgb', const.mode)  #
    path_op = os.path.join(const.data_dir, "lmdb_final", const.dataset_name,
                           'op', const.mode)  #
else:
    print("# for dataset setting error")
if const.data_type == "rgb_op":
    const.video_folder = (path_rgb, path_op)
    # const.len_clip = (const.num_his + 1, const.num_his)
    const.clip_length = (const.num_his + 1, const.num_his)
    
else:
    tmp_mapp = {"rgb": path_rgb, "op": path_op,} #
    const.video_folder = tmp_mapp[const.data_type]
    if const.data_type == "rgb":
        # const.len_clip = const.num_his + 1
        const.clip_length = const.num_his + 1
    if const.data_type == "op": # 少一帧
        # const.len_clip = const.num_his
        const.clip_length = const.num_his
const.dataset_fn = test_dataset

if args.ckptfile:
    const.test_load_ckpt = args.ckptfile
else:
    tmp_map = {
        "ped2": ped2_ckpt,
        "avenue": avenue_ckpt,
        "shanghaitech": shanghaitech_ckpt,
    }
    const.test_load_ckpt = tmp_map[const.dataset_name]   

# (4) for testing
const.data_dir_gt = config_const.data_dir_gt
const.normalize = args.normalize
const.loss_name = args.loss_name
const.metric_name = args.metric_name
const.evaluate = args.evaluate
# ======================================================================================== #
# data loading
#
# hardward related
const.node = args.node #
const.gpu_idx = args.gpu
# const.num_workers = args.num_workers # 16
# const.batch_size = args.batch_size
#
# #  to get video_folder
# if const.which_ds == "normal":
#     path_rgb = os.path.join(const.data_dir,
#                             "{}/{}/frames".format(const.dataset_name, const.mode))  #
#     path_op = os.path.join(const.data_dir,
#                            "{}/optical_flow/{}/frames/flow".format(const.dataset_name,
#                                                                    const.mode))  #
# if const.which_ds == "lmdb":
#     path_rgb = os.path.join(const.data_dir, "lmdb_final", const.dataset_name,
#                             'rgb', const.mode)  #
#     path_op = os.path.join(const.data_dir, "lmdb_final", const.dataset_name,
#                            'op', const.mode)  #
# #
# if const.data_type == "rgb_op":
#     const.video_folder = (path_rgb, path_op)
#     const.clip_length = (const.num_his[0] + 1, const.num_his[1] + 1)
# else:
#     tmp_mapp = {"rgb": path_rgb, "op": path_op, } #
#     const.video_folder = tmp_mapp[const.data_type]
#     const.clip_length = const.num_his + 1
#
# =============================================================================== #
#
const.test_save_dir = utils.get_dir(test_save_dir) #
#
const.log_save_root = utils.get_dir(os.path.join(const.test_save_dir,
        "log_dir"))
const.logger = utils.get_logger(const.test_log_config_path,
        const.log_save_root, "test")
const.summary_dir = utils.get_dir(os.path.join(const.test_save_dir, 'summary'))
const.vis_sample_size = config_const.vis_sample_size #

