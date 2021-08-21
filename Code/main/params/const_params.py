
from ...utils import utils
import os,time

class ConstConfig(object):
    # please setting 
    root_dir = "/p300/model_run_result/ammcnet_os"
    dataset_dir = "/p300/dataset"
    data_dir_gt = "/p300/dataset"

    log_config_path = os.path.join(root_dir,
                                   "logger_config/train_log_config.yaml")
    test_log_config_path = os.path.join(root_dir,
                                   "logger_config/test_log_config.yaml")
    
    img_height = 256
    img_width = 256
    channel_dict = {
        "rgb": 3,
        "op": 2,
        "rgb_op": (3, 2)
    }
    his_dict = {
        "rgb": 4,
        "op": 3,
        "rgb_op": (4, 3)
    }
    #
    vis_sample_size = 4
    step_log = 10 # 10
    step_summ = 100 # 100
    step_save_ckpt = 1000 # 1000,
    #
    d_num_filters = [128, 256, 512, 512]
    #
    # 
    save_root = os.path.join(root_dir, "model_result_save")
    ped2_net_params = os.path.join(root_dir, "net_params/ped2_net_params.pkl")
    avenue_net_params = os.path.join(root_dir, "net_params/avenue_net_params.pkl")
    shanghaitech_net_params = os.path.join(root_dir, "net_params/shanghaitech_net_params.pkl")
    #
    ped2_ckpt = os.path.join(root_dir, "load_model_ckpt/ped2.pth")
    avenue_ckpt = os.path.join(root_dir, "load_model_ckpt/avenue.pth")
    shanghaitech_ckpt = os.path.join(root_dir, "load_model_ckpt/shanghaitech.pth")
    #
    flow_model_path = '/p300/pretain_model/' \
                      'pretrain4FLowNet2_pytorch/' \
                      'FlowNet2-SD_checkpoint.pth.tar'

    # #
    cur_goal_tmp = utils.get_dir('/p300/model_run_result/ammcnet_os/log')  #
    tmp = utils.get_dir(os.path.join(cur_goal_tmp, "train_config"))
    exp_tag_log_save = os.path.join(tmp, "exp_tag_log.json")
    # net_init_params
    net_params_pickle_save = os.path.join(tmp, 'net-{}.pkl'.format(
        str(round(time.time()))))  # save_model as pickle,
    net_params_map = os.path.join(tmp, "net_params_map.json") #
    # ds_init_params
    ds_params_pickle_save = os.path.join(tmp, 'ds-{}.pkl'.format(
        str(round(time.time()))))  # save_model as pickle,
    ds_params_map = os.path.join(tmp, "ds_params_map.json")

