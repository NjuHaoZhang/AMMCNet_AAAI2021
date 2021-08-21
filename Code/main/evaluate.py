import sys
sys.path.append('..')
#
import os
import time
import numpy as np
import pickle,json
import glob
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from Dataset import two_stream_dataset
from models.unet import get_model
from losses import *
#
import eval_metric
from constant import const
import util
import utils

# device_idx = const.GPU
# device = torch.device("cuda:{}".format(device_idx))

dataset_name = const.DATASET
data_dir = const.DATA_DIR
test_folder = const.TEST_FOLDER
proj_root = const.PROJ_ROOT
log_config_path = const.LOG_CONFIG_TEST_TPATH
log_save_root = const.LOG_SAVE_ROOT
logger = util.get_logger(log_config_path, log_save_root, "test")

num_unet_layers = const.NUM_UNET_LAYES
embed_dim = const.EMBED_DIM
n_embed = const.N_EMBED
k = const.K
features_root = const.UNET_FEATURE_ROOT

num_his = const.NUM_HIS
frame_num = num_his + 1
input_channels = (3 * num_his, 2*(num_his-1))
output_channels = (3,2)

evaluate_name = const.EVALUATE
loss_func = const.LOSS_FUNC


def gen_loss_file(frame_num, model, dataset, pickle_path, writer,):

    time_stamp = time.time()
    total = 0  # count total num of processed frame
    #
    psnr_records = []
    mse_records = []
    ssim_records = []

    all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
    for video_id, sub_video_name in enumerate(all_sub_video_name_list):
        #
        len_all_frame = dataset.videos["rgb"][sub_video_name]["length"]
        psnrs = np.empty(shape=(len_all_frame,), dtype=np.float32)  # 先填充 num_clip为 num_all_frame
        mses = np.empty(shape=(len_all_frame,), dtype=np.float32)
        ssims = np.empty(shape=(len_all_frame,), dtype=np.float32)
        #
        # padding部分frame在真正计算 metric时再舍弃
        dataset.test(sub_video_name)
        for idx in range(len(dataset)):
            sample = dataset[idx] # (t,c,h,w), since no batch
            rgb, op = sample["rgb"].to(device), sample["op"].to(device)  # (b,t,c,h,w)
            rgb_input, op_input = rgb[:-1, :, :, :], op[:-1, :, :, :]
            rgb_target, op_target = rgb[-1, :, :, :], op[-1, :, :, :]
            rgb_input_last = rgb[-1, :, :, :]
            # as (1,t*c,h,w)
            rgb_input = rgb_input.view(1,-1,
                                       rgb_input.shape[-2], rgb_input.shape[-1])
            op_input = op_input.view(1,-1,
                                     op_input.shape[-2], op_input.shape[-1])
            #
            with torch.no_grad():
                rgb_G_output, op_G_output, diff, _ = model(rgb_input, op_input) #(b,c,h,w)

            #
            # print("shape: ",rgb_G_output.size(),rgb_target.size())
            rgb_target = rgb_target.unsqueeze(0)
            test_psnr = util.psnr_error(rgb_G_output, rgb_target).item()
            test_mse = util.mse_error(rgb_G_output, rgb_target).item()
            test_ssim = util.ssim_error(rgb_G_output, rgb_target).item()
            # scalar_tensor in, scalar out， 此处可以用 item()代替
            psnrs[idx + frame_num - 1] = test_psnr
            mses[idx + frame_num - 1] = test_mse
            ssims[idx + frame_num - 1] = test_ssim
            # (1) test_counter是标记当前clip_idx, 外部寻址
            # (2) clip内部的 offeset用idx表示, prediction process is  0...(frame_num-2) => (frame_num-1)
            total += 1
        #
        psnrs[:frame_num - 1] = psnrs[frame_num - 1]  # left (frame_num-1) frame 直接copy-pad
        psnr_records.append(psnrs)
        mses[:frame_num - 1] = mses[frame_num - 1]  # left (frame_num-1) frame 直接copy-pad
        mse_records.append(mses)
        ssims[:frame_num - 1] = ssims[frame_num - 1]  # left (frame_num-1) frame 直接copy-pad
        ssim_records.append(ssims)
        #
        logger.info('finish test video set {}'.format(sub_video_name))

    result_dict = {'dataset': dataset_name,'psnr': psnr_records,
                   'mse': mse_records, 'ssim': ssim_records,
                   'flow': [], 'names': [], 'diff_mask': []}

    used_time = time.time() - time_stamp
    logger.info('total time = {}, fps = {}'.format(used_time, total / used_time))

    with open(pickle_path, 'wb') as fp:
        pickle.dump(result_dict, fp, pickle.HIGHEST_PROTOCOL)

def gen_mul_metrics_manual(psnr_dir, loss_func, evaluate_name_tuple):
    def save_res(save_path, data_dict):
        with open(save_path, "w") as fp:
            json.dump(data_dict, fp)

    tmp_root = os.path.join(psnr_dir, loss_func)
    save_pickle_root = util.get_dir(os.path.join(tmp_root, "save_pickle"))
    for evaluate_name in evaluate_name_tuple:
        ret_dict = eval_metric.evaluate(evaluate_name, save_pickle_root)
        save_path = os.path.join(tmp_root, evaluate_name+".json")
        save_res(save_path, ret_dict)

def evaluate(model, dataset, frame_num, summary_dir, snapshot_dir, psnr_dir,
                evaluate_name_tuple=('compute_auc_by_psnr_mse_ssim',
                                    'calculate_psnr_mse_ssim')
             ):
    def check_ckpt_valid(ckpt):
        # check strategy by 正则表达式， todo
        is_valid = False
        ckpt_name = ''
        if ckpt.endswith('.pth'):
            # TODO specify what's the actual name of ckpt.
            ckpt_name = os.path.split(ckpt)[-1]
            is_valid = True
        return is_valid, ckpt_name

    def get_ckpt_from_name(snapshot_dir, ckpt_name):
        return os.path.join(snapshot_dir, ckpt_name)

    def check_pass(idx_ckpt):
        return False

    def scan_model_folder(dir_root):
        saved_models = set()  # elem value:
        for ckpt in os.listdir(dir_root):  #
            is_valid, ckpt_name = check_ckpt_valid(ckpt)
            if is_valid:
                saved_models.add(ckpt_name)
        return saved_models  # [vqvae_epoch_xxx, ]

    def scan_psnr_folder(dir_root):
        tested_ckpt_in_psnr_sets = set()  # elem value: vqvae_epoch_xxx
        for tested_psnr in os.listdir(dir_root):
            tested_ckpt_in_psnr_sets.add(tested_psnr)
        return tested_ckpt_in_psnr_sets  # [vqvae_epoch_xxx, ]

    # tmp_root = os.path.join(psnr_dir, loss_func)
    save_pickle_root = util.get_dir(os.path.join(psnr_dir, "save_pickle"))

    if os.path.isdir(snapshot_dir):
        # snapshot_dir 是文件夹，说明想需要实时监测 新生成的 model_ckpt to evaluate
        tested_ckpt_sets = scan_psnr_folder(save_pickle_root)  # 保存了tested_ckpt, # [vqvae_epoch_xxx, ],初始为空
        while True:
            all_model_ckpts = scan_model_folder(snapshot_dir)  # [vqvae_epoch_xxx, ], each while就更新一下 all_ckpt
            new_model_ckpts = all_model_ckpts - tested_ckpt_sets  # 本次循环扫描新增的model_ckpt, vqvae_epoch_xxx
            new_model_ckpts = sorted(new_model_ckpts)
            logger.info("new_model_ckpts: ", new_model_ckpts)
            for idx_ckpt, ckpt_name in enumerate(new_model_ckpts):  #
                pickle_path = os.path.join(save_pickle_root, ckpt_name)  # ckpt_name:
                #
                # (1)-inference
                if check_pass(idx_ckpt):
                    with open(pickle_path, "a+") as fp:
                        pass # make file: pickle_path
                    continue # 跳过本次
                load_ckpt_path = get_ckpt_from_name(snapshot_dir, ckpt_name)  # /path/to/vqvae_epoch_xxx.pth
                is_valid, ckpt_name = check_ckpt_valid(load_ckpt_path)
                if is_valid is True:
                    model.load_state_dict(torch.load(load_ckpt_path, map_location=device))
                    with SummaryWriter(log_dir=summary_dir) as writer:
                        # (1)
                        gen_loss_file(frame_num, model, dataset, pickle_path, writer)
                        # (2)
                        # ret_dict["res_list"], ret_dict["optimal_kv"]
                        #
                        auc_list = eval_metric.evaluate(evaluate_name_tuple[0], save_pickle_root)
                        auc_by_loss = ['auc_by_psnr', 'auc_by_mse', 'auc_by_ssim']
                        for idx in range(3):
                            writer.add_text("test/{}/res_list".format(auc_by_loss[idx]),
                                            auc_list[idx]["res_list"])
                            writer.add_text("test/{}/optimal_kv".format(auc_by_loss[idx]),
                                            auc_list[idx]["optimal_kv"])
                        #
                        score_list = eval_metric.evaluate(evaluate_name_tuple[1], save_pickle_root)
                        score_by_loss = ['psnr_score', 'mse_score', 'ssim_score']
                        for idx in range(3):
                            writer.add_text("test/{}/res_list".format(score_by_loss[idx]),
                                            score_list[idx]["res_list"])
                            writer.add_text("test/{}/optimal_kv".format(score_by_loss[idx]),
                                            score_list[idx]["optimal_kv"])



                    # (2)-mark tested_ckpt_name
                    tested_ckpt_sets.add(ckpt_name)
                else:
                    logger.info("is_valid error")
                    exit()
            logger.info('waiting for models...')
            time.sleep(60)  # 间隔60s等待 training process写入最新的ckpt到checkpoints dir
    else:
        # inference, single ckpt test mode
        load_ckpt_path = snapshot_dir # /path/to/vqvae_epoch_xxx.pth
        is_valid, ckpt_name = check_ckpt_valid(load_ckpt_path)
        if is_valid is True:
            model.load_state_dict(torch.load(load_ckpt_path, map_location=device))
            with SummaryWriter(log_dir=summary_dir) as writer:
                pass
        else:
            logger.info("is_valid 2 error")
            exit()
    #



if __name__ =='__main__':
    #
    # save_dir
    save_dir = const.MODEL_PARAMS

    # log_path = os.path.join(util.get_dir(os.path.join(proj_root, "log_dir", save_dir)), "test.txt")
    summary_dir = os.path.join(proj_root, "summary", save_dir)
    psnr_dir = os.path.join(proj_root, "psnrs", save_dir)
    snapshot_dir = os.path.join(proj_root, "checkpoints", save_dir, "generator")
    #
    ckptfile = const.CKPTFILE
    if ckptfile is not None:
        snapshot_dir = os.path.join(snapshot_dir, ckptfile)  # single test mode
    # model
    model = get_model(in_channel=input_channels, output_channel=output_channels,
                        embed_dim=embed_dim, n_embed=n_embed, k=k,
                        layer_nums=num_unet_layers,
                        features_root=features_root).to(device).eval()
    #
    mode = "testing"
    path_rgb = os.path.join(data_dir,
                            "{}/{}/frames".format(dataset_name, mode))  #
    path_optical_flow = os.path.join(data_dir,
                                     "{}/optical_flow/{}/frames/flow".format(
                                         dataset_name, mode))  #
    video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
    dataset = two_stream_dataset.TwoStream_Test_DS(video_folder,
                                                    (frame_num, frame_num - 1))
    #
    evaluate(model, dataset, frame_num, summary_dir, snapshot_dir, psnr_dir)