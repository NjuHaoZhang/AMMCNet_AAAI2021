import os,time
import argparse
import configparser
#
import torch
#
from ..utils import utils
from .params.const_params import ConstConfig as const_params
'''
为了保持一致性：本文件一次性加入全部参数，
然后模型里具体用什么只取部分即可
这里尽可能处理好code 实参接受的所有的直接和间接参数 (code内部动态生成的 tmp vriable 就不需要)
'''

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

# code启动，argparse会【自动】收集 命令行参数
def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')

    # device config
    parser.add_argument('--node', type=str, default='48',
                        help='the node of gpu.')
    parser.add_argument('--gpu', type=str, default='3',
                        help='the device id of gpu.')
    parser.add_argument('-i', '--iters', type=int, default=80000,
                        help='set the number of iterations, default is 80000')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='set the batch size, default is 32.')
    parser.add_argument('-n', '--num_workers', type=int, default=16,
                        help='set the num_workers, default is 16.')

    # exp_setting
    parser.add_argument('--mode', type=str,
                        default='training',
                        help='exp mode: training or testing')
    parser.add_argument('--exp_tag', type=str,
                        default='',
                        help='identify a exp for train and test_load')
    parser.add_argument('--helper_tag', type=str,
                        default='',
                        help='net_tag')
    parser.add_argument('--net_tag', type=str,
                        default='',
                        help='net_tag')
    parser.add_argument('--loss_tag', type=str,
                        default='',
                        help='loss_tag')


    # data config
    parser.add_argument('--data_type', type=str,
                        default="",
                        help='rgb or op or two_stream')
    parser.add_argument('-d', '--dataset_name', type=str,
                        help='the name of dataset.', default='')
    parser.add_argument('--data_dir', type=str,
                        default="/p300/dataset",
                        help='data_dir')
    parser.add_argument('--which_ds', type=str,
                        default="normal",
                        help='data_dir')

    # for optimization
    parser.add_argument('--pretrain', type=bool,
                        default=False,
                        help='pretrain for training')  # 用于断点后继续train

    parser.add_argument('--use_fixed_params', type=bool,
                        default=False,
                        help='transfer learning')  # 用于断点后继续train

    parser.add_argument('--use_adv', type=bool,
                        default=True,
                        help='adv learning')  # GAN
    # for config

    return parser.parse_args()

# ================================================================================= #
# 参数收集器
const = Const() # const 这个变量在code收集 【在命令行、文件中设置的参数】并在其他code中使用

#
# (1) 从 命令行中配置 (运行时参数，也会经常修改，但不是超参)
args = parser_args() # 从命令行设置：一些辅助参数

# (2) 读入常规配置 (一般不需要 tune, 但是偶尔会手动修改)
config_const = const_params  # 相对是 const params

# (3) 读入 超参文件 (会经常修改)
config_tune = configparser.ConfigParser()  # hyper_params to tune
assert config_tune.read(config_const.tune_params_path)

# ========================================================================================= #
#
# log
const.log_config_path = config_const.log_config_path
# hyper_params to tune
const.config_tune = config_tune
const.config_const = config_const
# common setting
const.net_tag = args.net_tag
const.loss_tag = args.loss_tag
const.helper_tag = args.helper_tag
const.mode = args.mode # 区分 train and test mode,
# for data_loading, network_init, train-test
# ========================================================================= #
# Data

# hardward related
const.node = args.node
const.gpu_idx = args.gpu
# const.device = torch.device("cuda:{}".format(const.gpu))
const.num_workers = args.num_workers # 16
const.batch_size = args.batch_size

# dataset
const.data_dir = args.data_dir #
const.data_type = args.data_type # 区分 rgb, op and rgb_op (op暂时不处理)
const.dataset_name = args.dataset_name
#
const.height = config_const.img_height
const.width = config_const.img_width
const.num_channel = config_const.channel_dict[const.data_type] # 3-rgb or 2-op
const.num_his = config_const.his_dict[const.data_type]
#
# for dataset setting (没必要也很难解耦出去)
const.which_ds = args.which_ds # normal or lmdb, 必须设置，默认是 normal
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
#
# TODO 其实下面的还是写复杂了：默认 rgb 比 op 多一帧即可 消除所有的 tuple (即先验)
if const.data_type == "rgb_op":
    const.video_folder = (path_rgb, path_op)
    const.clip_length = (const.num_his[0] + 1, const.num_his[1] + 1)
else:
    tmp_mapp = {"rgb": path_rgb, "op": path_op, } #
    const.video_folder = tmp_mapp[const.data_type]
    const.clip_length = const.num_his + 1

# ============================================================================== #
# Model

# for generator
#
# (1) 设置统一的一组参数，但是初始化为: "unuse" (为了大一统的不得已)
# 所以 xxx_params.ini 就只需要关注本次实验需要修改的参数，其他无关的参数会下面自动化清0
# xxx_params.ini  仅仅是 方便修改的，真正的 modify 在后面(2)中
const.in_channel = "unuse"
const.out_channel = "unuse"
const.embed_dim = "unuse"
const.n_embed = "unuse"
const.k = "unuse"
const.use_fixed_params = "unuse"
#
# (2) 根据 具体的generator init 需要什么参数特异性设置， 1个网络1个 if branch
if const.net_tag == "vqvae" or \
        const.net_tag == "vqvae_res":
    const.in_channel = int(const.num_his * const.num_channel)  # 9*3
    const.out_channel = const.num_channel  # 1*3
    # Memory params (important), 其实是需要调节的参数，写到 tune.yaml file 去
    const.embed_dim = config_tune.getint(const.dataset_name, 'embed_dim')
    const.n_embed = config_tune.getint(const.dataset_name, 'n_embed')
if const.net_tag == "vqvae_topk" or \
        const.net_tag == "vqvae_topk_res":
    # 无须调的参，直接写死，待会儿写到文件中，方便修改 (代码里不要出现hardcode)
    const.in_channel = int(const.num_his * const.num_channel)  # 9*3
    const.out_channel = const.num_channel  # 1*3
    # Memory params (important), 其实是需要调节的参数，写到 tune.yaml file 去
    const.embed_dim = config_tune.getint(const.dataset_name, 'embed_dim')
    const.n_embed = config_tune.getint(const.dataset_name, 'n_embed')
    const.k = config_tune.getint(const.dataset_name, 'k')
if const.net_tag == "vqvae_twostream":
    const.in_channel = ( int(const.num_his[0] * const.num_channel[0]),
                        int(const.num_his[1] * const.num_channel[1]))  # 9*3
    const.out_channel = const.num_channel  # 1*3
    # Memory params (important)
    const.embed_dim = config_tune.getint(const.dataset_name, 'embed_dim')
    const.n_embed = config_tune.getint(const.dataset_name, 'n_embed')
    const.k = config_tune.getint(const.dataset_name, 'k')
    const.use_fixed_params = args.use_fixed_params
#
#
if const.net_tag == "unet":  # 暂时啥都不做，做 ablation study再说
    const.in_channel = int(const.num_his * const.num_channel)  # 4*3
    const.out_channel = const.num_channel  # 1*3
    const.num_unet_layers = config_tune.getint(const.dataset_name, 'num_unet_layers')
    const.unet_feature_root = config_tune.getint(const.dataset_name, 'unet_feature_root')
if const.net_tag == "unet_vq" or \
        const.net_tag == "unet_vq_res":
    const.in_channel = int(const.num_his * const.num_channel)  # 4*3
    const.out_channel = const.num_channel  # 1*3
    const.num_unet_layers = config_tune.getint(const.dataset_name, 'num_unet_layers')
    const.unet_feature_root = config_tune.getint(const.dataset_name, 'unet_feature_root')
    const.embed_dim = config_tune.getint(const.dataset_name, 'embed_dim')
    const.n_embed = config_tune.getint(const.dataset_name, 'n_embed')
if const.net_tag == "unet_vq_topk" or \
        const.net_tag == "unet_vq_topk_res":  # 暂时仅做 rgb
    const.in_channel = int(const.num_his * const.num_channel)  # 4*3
    const.out_channel = const.num_channel  # 1*3
    const.num_unet_layers = config_tune.getint(const.dataset_name, 'num_unet_layers')
    const.unet_feature_root = config_tune.getint(const.dataset_name, 'unet_feature_root')
    const.embed_dim = config_tune.getint(const.dataset_name, 'embed_dim')
    const.n_embed = config_tune.getint(const.dataset_name, 'n_embed')
    const.k = config_tune.getint(const.dataset_name, 'k')
if const.net_tag == "unet_vq_twostream":
    const.in_channel = (int(const.num_his[0] * const.num_channel[0]),
                        int(const.num_his[1] * const.num_channel[1]))  # 9*3
    const.out_channel = const.num_channel  # 1*3
    const.num_unet_layers = config_tune.getint(const.dataset_name, 'num_unet_layers')
    const.unet_feature_root = config_tune.getint(const.dataset_name, 'unet_feature_root')
    const.embed_dim = config_tune.getint(const.dataset_name, 'embed_dim')
    const.n_embed = config_tune.getint(const.dataset_name, 'n_embed')
    const.k = config_tune.getint(const.dataset_name, 'k')
    const.use_fixed_params = args.use_fixed_params
#
const.pretrain = args.pretrain  # load ckpt for traininng (for generator in rag branch)

# for discriminator
const.d_num_filters = config_const.d_num_filters

# for flownet
const.flow_model_path = config_const.flow_model_path # for flow_loss in rgb branch

# ===================================================================================#
# Training (loss, lam)

const.iterations = args.iters # 不要变，最多调 lr_decay

# for lam_loss (optimization), lr, lr_decay, 根据不同 loss 调度不同的 if branch
# (1) 清0
const.l_num = "unuse"
const.alpha_num = "unuse"
const.lam_adv = "unuse"
const.lam_lp = "unuse"
const.lam_gdl = "unuse"
const.lam_flow = "unuse"
const.lam_latent = "unuse"
const.lam_lp_op = "unuse"
const.lam_adv_op = "unuse"
#
# (2) 为不同 loss 修改不同参数
# for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
if const.loss_tag == "rgb_int_gdl_flow_adv":
    const.l_num = config_tune.getint(const.dataset_name, 'l_num')
    # the power to which each gradient term is raised in GDL loss
    const.alpha_num = config_tune.getint(const.dataset_name, 'alpha_num')
    # the percentage of the adversarial loss to use in the combined loss
    const.lam_adv = config_tune.getfloat(const.dataset_name, 'lam_adv')
    # the percentage of the lp loss to use in the combined loss
    const.lam_lp = config_tune.getfloat(const.dataset_name, 'lam_lp')
    # the percentage of the GDL loss to use in the combined loss
    const.lam_gdl = config_tune.getfloat(const.dataset_name, 'lam_gdl')
    # the percentage of the different frame loss
    const.lam_flow = config_tune.getfloat(const.dataset_name, 'lam_flow')
if const.loss_tag == "op_int_adv":
    # the percentage of the lp loss to use in the combined loss
    const.lam_lp_op = config_tune.getfloat(const.dataset_name, 'lam_lp_op')
    const.lam_adv_op = config_tune.getfloat(const.dataset_name, 'lam_adv_op')
if const.loss_tag == "twostream":
    const.l_num = config_tune.getint(const.dataset_name, 'l_num')
    # the power to which each gradient term is raised in GDL loss
    const.alpha_num = config_tune.getint(const.dataset_name, 'alpha_num')
    # the percentage of the adversarial loss to use in the combined loss
    const.lam_adv = config_tune.getfloat(const.dataset_name, 'lam_adv')
    # the percentage of the lp loss to use in the combined loss
    const.lam_lp = config_tune.getfloat(const.dataset_name, 'lam_lp')
    # the percentage of the GDL loss to use in the combined loss
    const.lam_gdl = config_tune.getfloat(const.dataset_name, 'lam_gdl')
    # the percentage of the different frame loss
    const.lam_flow = config_tune.getfloat(const.dataset_name, 'lam_flow')
    #
    const.lam_lp_op = config_tune.getfloat(const.dataset_name, 'lam_lp_op')
if const.loss_tag == "rgb_int_gdl_flow_adv_vq":
    const.l_num = config_tune.getint(const.dataset_name, 'l_num')
    # the power to which each gradient term is raised in GDL loss
    const.alpha_num = config_tune.getint(const.dataset_name, 'alpha_num')
    # the percentage of the adversarial loss to use in the combined loss
    const.lam_adv = config_tune.getfloat(const.dataset_name, 'lam_adv')
    # the percentage of the lp loss to use in the combined loss
    const.lam_lp = config_tune.getfloat(const.dataset_name, 'lam_lp')
    # the percentage of the GDL loss to use in the combined loss
    const.lam_gdl = config_tune.getfloat(const.dataset_name, 'lam_adv')
    # the percentage of the different frame loss
    const.lam_flow = config_tune.getfloat(const.dataset_name, 'lam_flow')
    #
    const.lam_latent = config_tune.getfloat(const.dataset_name, 'lam_latent')
if const.loss_tag == "op_int_adv_vq":
    # the percentage of the lp loss to use in the combined loss
    const.lam_lp_op = config_tune.getfloat(const.dataset_name, 'lam_lp_op')
    const.lam_adv_op = config_tune.getfloat(const.dataset_name, 'lam_adv_op')
    #
    const.lam_latent = config_tune.getfloat(const.dataset_name, 'lam_latent')
if const.loss_tag == "twostream_vq":
    const.l_num = config_tune.getint(const.dataset_name, 'l_num')
    # the power to which each gradient term is raised in GDL loss
    const.alpha_num = config_tune.getint(const.dataset_name, 'alpha_num')
    # the percentage of the adversarial loss to use in the combined loss
    const.lam_adv = config_tune.getfloat(const.dataset_name, 'lam_adv')
    # the percentage of the lp loss to use in the combined loss
    const.lam_lp = config_tune.getfloat(const.dataset_name, 'lam_lp')
    # the percentage of the GDL loss to use in the combined loss
    const.lam_gdl = config_tune.getfloat(const.dataset_name, 'lam_adv')
    # the percentage of the different frame loss
    const.lam_flow = config_tune.getfloat(const.dataset_name, 'lam_flow')
    #
    const.lam_latent = config_tune.getfloat(const.dataset_name, 'lam_latent')
    #
    const.lam_lp_op = config_tune.getfloat(const.dataset_name, 'lam_lp_op')

# for lr, lr_decay (optimization)
# Learning rate of generator
const.lr_g = config_tune.getfloat(const.dataset_name, 'lr_g')
# const.LRATE_G_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_G_BOUNDARIES'))
const.step_decay_g = eval(config_tune.get(const.dataset_name, 'step_lr_g_decay'))
#
const.use_adv = args.use_adv # 默认 False (rgb 需要，op 不需要？)
const.lr_d = "unuse"
const.step_decay_d = "unuse"
if const.use_adv:
    # Learning rate of discriminator
    const.lr_d = config_tune.getfloat(const.dataset_name, 'lr_d')
    # const.LRATE_D_BOUNDARIES = eval(config.get(const.DATASET, 'LRATE_D_BOUNDARIES'))
    const.step_decay_d = eval(config_tune.get(const.dataset_name, 'step_lr_d_decay'))

# ============================================================================== #
# Result save (ckpt, summ_dir, )

# -----------------------------------------------------------------------#
# 联系 train and test (非常重要)
const.exp_tag = args.exp_tag
#
# train_save_dir 写入exp_tag_log_save，
# 并且exp 作为 它的key (所以 test mode 可以直接拿到 train_save_dir)
const.exp_tag_log_save = config_const.exp_tag_log_save
#
# save net_init_params (这两个的 train_setting 写入文件 由 net_init,ds_init
# 的code 实现，即 get_model(), get_dataset(), )
const.net_params_pickle_save = config_const.net_params_pickle_save
const.net_params_map = config_const.net_params_map
const.ds_params_pickle_save = config_const.ds_params_pickle_save
const.ds_params_map = config_const.ds_params_map

# 以network为核心
const.proj_root = utils.get_dir(os.path.join(config_const.cur_goal_tmp, const.net_tag))
# 将 exp_tag => save_dir 写入文件，等待 testing 读
# timestamp确保不重复
save_path = "{}-{}-{}-{}".format(
    const.net_tag, const.dataset_name, const.data_type, str(round(time.time())))
const.save_dir = os.path.join(const.proj_root, save_path)
tmp = (const.exp_tag, const.save_dir)  # 关联 train and test 的核心-这个 json
utils.save_json(const.exp_tag_log_save, tmp)  # 多process 同时写 会出错，
# so要加锁(process-level)
# ------------------------------------------------------------------------------#
#
# ckpt, summ save
const.train_save_ckpt = utils.get_dir(os.path.join(const.save_dir,
                                                   'training/checkpoints'))  # the path of save model_ckpt
const.rgb_model_path = config_const.rgb_model_path[const.dataset_name]
const.op_model_path = config_const.op_model_path[const.dataset_name]
#
# the path of save model_ckpt in disk)
#
const.log_save_root = utils.get_dir(os.path.join(const.save_dir, "log_dir"))
const.summary_dir = utils.get_dir(os.path.join(const.save_dir, 'summary'))
const.vis_sample_size = config_const.vis_sample_size  # for summ visulization
const.step_log = config_const.step_log
const.step_summ = config_const.step_summ
const.step_save_ckpt = config_const.step_save_ckpt
#
const.logger = utils.get_logger(const.log_config_path, const.log_save_root, "train")
# ==================================================================================== #


# TODO: 统计不同 model_params 的 best_metric 用另一个统计code
# 尽可能避免人工去copy-paste 然后计算比较，都交给code自动化