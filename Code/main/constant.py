import sys
# sys.path.append('..')
import os
import argparse
import configparser

from ..utils import utils

class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        if not name.isupper():
            raise self.ConstCaseError('const name {} is not all uppercase'.format(name))

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            print(name, value)
            _str += '\t{}\t{}\n'.format(name, value)

        return _str

def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the device id of gpu.')
    parser.add_argument('-n', '--node', type=str, default='admin-node',
                        help='the node of gpu.')
    parser.add_argument('-i', '--iters', type=int, default=80000,
                        help='set the number of iterations, default is 80000')
    parser.add_argument('-b', '--batch', type=int, default=4,
                        help='set the batch size, default is 4.')
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='set the num_workers, default is 4.')

    # parser.add_argument('-e', '--epoch_num', type=int, default=200,
    #                     help='set the epoch_num, default is 200.')
    parser.add_argument('--num_his', type=int, default=9,
                        help='set the time steps, default is 9.')
    parser.add_argument('-d', '--dataset_name', type=str,
                        help='the name of dataset.', default='')
    parser.add_argument('--train_folder', type=str, default='',
                        help='set the training folder path.')
    parser.add_argument('--test_folder', type=str, default='',
                        help='set the testing folder path.')

    # parser.add_argument('--config', type=str, default='',
    #                     help='the path of params, default is params/const_params.py')

    # parser.add_argument('--snapshot_dir', type=str, default=None,
    #                     help='if it is folder, then it is the directory to save models, '
    #                          'if it is a specific model.ckpt-xxx, then the system will load it for testing.')
    parser.add_argument('--summary_dir', type=str, default=None, help='the directory to save summaries.')
    # parser.add_argument('--psnr_dir', type=bool, default=True, help='save psnrs results in testing.')
    parser.add_argument('--evaluate', type=str, default='compute_auc',
                        help='the evaluation metric, default is compute_auc')

    # parser.add_argument('--eval_epoch', type=int, default=10,
    #                     help='the evaluation timestep, default is 1')

    parser.add_argument('--num_net_layes', type=int, default=4,
                        help='the num_net_layes, default is 4')
    parser.add_argument('--unet_feature_root', type=int,
                        default=64,
                        help='model_params')
    #
    parser.add_argument('--mode', type=str,
                        default='training',
                        help='exp mode: training or testing')

    parser.add_argument('--data_channel', type=int,
                        default=3,
                        help='data_channel')

    parser.add_argument('--data_type', type=str,
                        default="rgb",
                        help='rgb or op or two_stream')

    parser.add_argument('--exp_tag', type=str,
                        default='baseline_v1',
                        help='identify a exp for train and test_load')
    parser.add_argument('--net_tag', type=str,
                        default='vqvae',
                        help='net_tag')

    # parser.add_argument('--embed_dim', type=int,
    #                     default=64,
    #                     help='embed_dim')
    #
    # parser.add_argument('--n_embed', type=int,
    #                     default=512,
    #                     help='n_embed')

    parser.add_argument('--discriminator_num_filters', type=list,
                        default=[128,256,512,512],
                        help='discriminator_num_filters')

    parser.add_argument('--ckptfile', type=str,
                        default=None,
                        help='ckptfile')

    parser.add_argument('--loss_func', type=str,
                        default="PSNRLoss",
                        help='loss_func')
    parser.add_argument('--normalize', type=bool,
                        default=True,
                        help='normalize each sub_video psnr or mse')

    parser.add_argument('--data_dir', type=str,
                        default="/p300/dataset",
                        help='data_dir')

    parser.add_argument('--pretrain', type=bool,
                        default=False,
                        help='pretrain for training')
    return parser.parse_args()

const = Const() #
args = parser_args()

# ===================================================================== #
const.PROJ_ROOT = utils.get_dir('/p300/model_run_result/'
            'zk8')
#
# input image
const.HEIGHT = 256
const.WIDTH = 256
# flownet
const.FLOWNET_CHECKPOINT = ""
const.FLOW_HEIGHT = 384
const.FLOW_WIDTH = 512
const.NUM_CHANNEL = 3
const.SAMPLE_SIZE = 8

# for training config
CODEROOT = ""
const.CONFIG = os.path.join(CODEROOT,
                            "params/const_params.py")
const.LOG_CONFIG_TRAIN_PATH = os.path.join(CODEROOT,
                              "logger_config/train_log_config.yaml")
# for testing
const.LOG_CONFIG_TEST_TPATH = os.path.join(CODEROOT,
                              "logger_config/test_log_config.yaml")
# train and test
const.EXP_TAG_LOG_SAVE =os.path.join(const.PROJ_ROOT, "exp_tag_log.json")


config = configparser.ConfigParser()
assert config.read(const.CONFIG)

const.MODE = args.mode
# hardward related
const.GPU = args.gpu
const.BATCH_SIZE = args.batch
const.NUM_WORKERS = args.num_workers
const.NUM_HIS = args.num_his
const.ITERATIONS = args.iters
# const.EPOCH_NUM = args.epoch_num
# inputs data
const.DATA_TYPE = args.data_type
const.DATA_DIR = args.data_dir #
const.DATASET = args.dataset

# UNet
const.NUM_UNET_LAYES = args.num_net_layes
const.UNET_FEATURE_ROOT = args.unet_feature_root
# discriminator
const.D_NUM_FILTERS = args.discriminator_num_filters
# Memory params (important)
const.EMBED_DIM = config.getint(const.DATASET, 'EMBED_DIM')
const.N_EMBED = config.getint(const.DATASET, 'N_EMBED')
const.K = config.getint(const.DATASET, 'K')
#
const.EXP_TAG = args.exp_tag # for training and testing


#
if const.MODE == "testing":
    const.TEST_FOLDER = os.path.join(const.DATA_DIR, args.test_folder)
    const.EVALUATE = args.evaluate
    const.LOSS_FUNC = args.loss_func
    const.NORMALIZE = args.normalize
    # const.DECIDABLE_IDX = 4

    const.SAVE_DIR = utils.load_json(const.EXP_TAG_LOG_SAVE, const.EXP_TAG) #
    print("save_dir: ", const.SAVE_DIR)
    const.TEST_LOAD_CKPT = os.path.join(const.PROJ_ROOT,
                            const.SAVE_DIR, "training/checkpoints/generator") #
    if args.ckptfile:
        const.TEST_LOAD_CKPT = os.path.join(const.TEST_LOAD_CKPT, args.ckptfile)
    # test_res_save
    const.TEST_RES_SAVE = os.path.join(const.PROJ_ROOT,const.SAVE_DIR, 'testing')
    # other params
    # const.PSNRS_DIR = utils.get_dir(os.path.join(const.TEST_RES_SAVE, "psnrs"))
    # const.RES_DICT_FILE = os.path.join(const.TEST_RES_SAVE, "test_result.json")

path_rgb = os.path.join(const.DATA_DIR,
    "{}/{}/frames".format(const.DATASET, const.MODE))  #
path_optical_flow = os.path.join(const.DATA_DIR,
    "{}/optical_flow/{}/frames/flow".format(const.DATASET, const.MODE))  #
# print(path_rgb)
const.VIDEO_FOLDER = {"rgb": path_rgb, "op": path_optical_flow, }

const.LOG_SAVE_ROOT = utils.get_dir(os.path.join(const.PROJ_ROOT, const.SAVE_DIR, "log_dir"))

if args.summary_dir:
    const.SUMMARY_DIR = utils.get_dir(args.summary_dir)
else:
    const.SUMMARY_DIR = utils.get_dir(os.path.join(const.PROJ_ROOT, const.SAVE_DIR, 'summary'))

