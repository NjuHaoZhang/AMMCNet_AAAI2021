import os,time
from ..main.constant_test import const
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= const.gpu_idx
import pickle
import json
#
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

from ..utils import utils
from ..main import eval_metric #


def gen_loss_file_single_normal(model, dataset_fn, net_tag,
         data_root, len_clip, data_type,
         dataset_name, pickle_path, loss_name, logger):
         
    time_stamp = time.time()
    total = 0  # count total num of processed frame
    psnr_records = []
    loss_func_mapp = {
         "psnr": utils.psnr_error,
         "mse": utils.mse_error,
         "ssim": utils.ssim_error,
         "epe": utils.epe_error,
    }
    all_sub_video_name_list = os.listdir(data_root)
    all_sub_video_name_list.sort()
    print("all_sub_video_name_list: ", all_sub_video_name_list)
    for video_id, sub_video_name in enumerate(all_sub_video_name_list):
        path = os.path.join(data_root, sub_video_name)
        ds = dataset_fn(path, len_clip, data_type) # test_dataset
        ds_loader = DataLoader(ds,
                 batch_size=16, shuffle=False, num_workers=8)
        num_frame = len(ds) + len_clip - 1
        print("len of sub_video: {}".format(num_frame))
        psnrs = np.empty(shape=(num_frame,), dtype=np.float32)
        if data_type == "op":
            psnrs = np.empty(shape=(num_frame+1,), dtype=np.float32)
            cnt = -1
            for idx_b, sample in enumerate(ds_loader):
                rgb = sample.cuda()  # (b,t,c,h,w)
                rgb_input = rgb[:, :-1, :, :, :]
                rgb_target = rgb[:, -1, :, :, :]
                rgb_input = rgb_input.view(rgb_input.shape[0],
                    -1,
                rgb_input.shape[-2], rgb_input.shape[-1])  # as (b,t*c,h,w)
                
                with torch.no_grad():
                    output = model(rgb_input) #out: (b,c,h,w)
                
                if net_tag == "unet":
                    rgb_G_output = output
                if net_tag == "unet_vq_topk_res":
                    rgb_G_output, latent_diff, embed_ind_tuple = output
                 
                for idx in range(rgb_G_output.size()[0]): #
                    cnt += 1 #
                    gen = rgb_G_output[idx].unsqueeze(0)
                    gt = rgb_target[idx].unsqueeze(0)
                    test_psnr = loss_func_mapp[loss_name](gen, gt).item()
                    psnrs[cnt + len_clip-1] = test_psnr
                    total += 1
            psnrs[:len_clip-1] = psnrs[len_clip-1]  #
            if data_type == "op": #
                psnrs[num_frame] = psnrs[num_frame-1] # #
             # print("len of psnr: ", len(psnrs))
            psnr_records.append(psnrs)
            logger.info('finish test video set {}'.format(sub_video_name))
        result_dict = {'dataset': dataset_name,'psnr': psnr_records,
                        'flow': [], 'names': [], 'diff_mask': []}
        used_time = time.time() - time_stamp
        logger.info('total time = {}, fps = {}'.format(used_time, total / used_time))
        with open(pickle_path, 'wb') as fp:
            pickle.dump(result_dict, fp, pickle.HIGHEST_PROTOCOL)


#        def gen_loss_file_twostream_normal(model, dataset_fn, net_tag,
#             data_root, len_clip, data_type,
#             dataset_name, rgb_pickle_path, op_pickle_path,
#             loss_name, logger):
#
#         time_stamp = time.time()
#         rgb_total, op_total = 0, 0  # count total num of processed frame
#         rgb_psnr_records, op_psnr_records = [], []
#         loss_func_mapp = {
#             "psnr": utils.psnr_error,
#             "mse": utils.mse_error,
#             "ssim": utils.ssim_error,
#         }
#
#         rgb_root, op_root = data_root
#         # print("rgb_root, op_root: ", rgb_root, op_root) #
#         all_sub_video_name_list = os.listdir(rgb_root)
#         all_sub_video_name_list.sort()
#         print("all_sub_video_name_list: ", all_sub_video_name_list)
#         #
#
#         rgb_len_clip, op_len_clip = len_clip
#         for video_id, sub_video_name in enumerate(all_sub_video_name_list):
#             rgb_folder = os.path.join(rgb_root, sub_video_name)
#             rgb_ds = dataset_fn(rgb_folder, rgb_len_clip, "rgb")
#             op_folder = os.path.join(op_root, sub_video_name)
#             op_ds = dataset_fn(op_folder, op_len_clip, "op") #
#             #
#             rgb_loader = DataLoader(rgb_ds, batch_size=16,
#                                     shuffle=False, num_workers=4)
#             op_loader = DataLoader(op_ds, batch_size=16,
#                                    shuffle=False, num_workers=4)
#             num_frame = len(rgb_ds) + rgb_len_clip - 1 #
#             print("len of sub_video: {}".format(num_frame))
#             rgb_psnrs = np.empty(shape=(num_frame,), dtype=np.float32)
#             op_psnrs = np.empty(shape=(num_frame,), dtype=np.float32) #
#             rgb_cnt, op_cnt = -1, -1
#             for idx, (rgb, op) in enumerate(zip(rgb_loader, op_loader)):
#                 rgb = rgb.cuda()  # (1,t,c,h,w)
#                 op = op.cuda()
#                 rgb_input = rgb[:, :-1, :, :, :]
#                 rgb_target = rgb[:, -1, :, :, :]
#                 op_input = op[:, :-1, :, :, :]
#                 op_target = op[:, :-1, :, :, :]
#                 # as (1,t*c,h,w)
#                 rgb_input = rgb_input.view(rgb_input.shape[0],
#                                            -1,
#                                            rgb_input.shape[-2], rgb_input.shape[-1])
#                 op_input = op_input.view(op_input.shape[0],
#                                          -1,
#                                          op_input.shape[-2], op_input.shape[-1])
#                 # print("rgb_input, op_input: ", rgb_input.size(), op_input.size()) #
#
#                 with torch.no_grad():
#                     output = model(rgb_input, op_input)  # out: (b,c,h,w)
#
#                 if net_tag == "unet" or net_tag == "unet_vq_twostream":
#                     rgb_G_output, op_G_output, diff_tuple, _ = output
#                     rgb_diff, op_diff = diff_tuple[0], diff_tuple[1]
#
#                 for idx in range(rgb_G_output.size()[0]):  #
#                     rgb_cnt += 1
#                     gen = rgb_G_output[idx].unsqueeze(0)
#                     gt = rgb_target[idx].unsqueeze(0)
#                     rgb_loss_func = loss_func_mapp[loss_name]
#                     rgb_test_psnr = rgb_loss_func(gen, gt).item()
#
#                     rgb_psnrs[rgb_cnt + rgb_len_clip - 1] = rgb_test_psnr
#                     rgb_total += 1
#                 for idx in range(rgb_G_output.size()[0]):
#                     op_cnt += 1
#                     gen = op_G_output[idx].unsqueeze(0)
#                     gt = op_target[idx].unsqueeze(0)
#                     op_loss_func = loss_func_mapp[loss_name]
#                     op_test_psnr = op_loss_func(gen, gt).item()
#
#                     op_psnrs[op_cnt + op_len_clip - 1] = op_test_psnr
#                     op_total += 1
#             rgb_psnrs[:rgb_len_clip - 1] = rgb_psnrs[rgb_len_clip - 1]  #
#             op_psnrs[:op_len_clip - 1] = op_psnrs[op_len_clip - 1]  #
#             op_psnrs[num_frame-1] = op_psnrs[num_frame - 2]  #
#             #
#             rgb_psnr_records.append(rgb_psnrs)
#             op_psnr_records.append(op_psnrs)
#             logger.info('finish test video set {}'.format(sub_video_name))
#         rgb_result_dict = {'dataset': dataset_name, '{}'.format(loss_name): rgb_psnr_records,
#                        'flow': [], 'names': [], 'diff_mask': []}
#         op_result_dict = {'dataset': dataset_name, '{}'.format(loss_name): op_psnr_records,
#                            'flow': [], 'names': [], 'diff_mask': []}
#         used_time = time.time() - time_stamp
#         logger.info('total time = {}, fps = {}'.format(used_time, rgb_total / used_time))
#         with open(rgb_pickle_path, 'wb') as fp:
#             pickle.dump(rgb_result_dict, fp, pickle.HIGHEST_PROTOCOL)
#         with open(op_pickle_path, 'wb') as fp:
#             pickle.dump(op_result_dict, fp, pickle.HIGHEST_PROTOCOL)
#
#
#def gen_loss_file_twostream_normal_v2(model, dataset_fn, net_tag,
#             data_root, len_clip, data_type,
#             dataset_name, rgb_pickle_path, op_pickle_path,
#             loss_name, logger):
#
#    time_stamp = time.time()
#    rgb_total, op_total = 0, 0  # count total num of processed frame
#    rgb_psnr_records, op_psnr_records = [], []
#    rgb_fea_comm_records, op_fea_comm_records = [], []
#    loss_func_mapp = { #
#         "psnr": utils.psnr_error,
#         "mse": utils.mse_error,
#         "ssim": utils.ssim_error,
#     }
#
#     rgb_root, op_root = data_root
#     # print("rgb_root, op_root: ", rgb_root, op_root) #
#     all_sub_video_name_list = os.listdir(rgb_root)
#     all_sub_video_name_list.sort()
#     print("all_sub_video_name_list: ", all_sub_video_name_list)
#     #
#     rgb_len_clip, op_len_clip = len_clip
#     for video_id, sub_video_name in enumerate(all_sub_video_name_list):
#
#        rgb_folder = os.path.join(rgb_root, sub_video_name)
#        rgb_ds = dataset_fn(rgb_folder, rgb_len_clip, "rgb")
#        op_folder = os.path.join(op_root, sub_video_name)
#        op_ds = dataset_fn(op_folder, op_len_clip, "op") #
#         #
#        rgb_loader = DataLoader(rgb_ds, batch_size=16,
#                                 shuffle=False, num_workers=4)
#        op_loader = DataLoader(op_ds, batch_size=16,
#                                shuffle=False, num_workers=4)
#        num_frame = len(rgb_ds) + rgb_len_clip - 1 #
#        print("len of sub_video: {}".format(num_frame))
#        rgb_psnrs = np.empty(shape=(num_frame,), dtype=np.float32) #
#        rgb_fea_comm_losses = np.empty(shape=(num_frame,), dtype=np.float32) #
#        op_psnrs = np.empty(shape=(num_frame,), dtype=np.float32) #
#        op_fea_comm_losses = np.empty(shape=(num_frame,), dtype=np.float32) #
#        rgb_cnt, op_cnt = -1, -1
#        for idx, (rgb, op) in enumerate(zip(rgb_loader, op_loader)):
#            rgb = rgb.cuda()  # (1,t,c,h,w)
#            op = op.cuda()
#            rgb_input = rgb[:, :-1, :, :, :]
#            rgb_target = rgb[:, -1, :, :, :]
#            op_input = op[:, :-1, :, :, :]
#            op_target = op[:, :-1, :, :, :]
#             # as (1,t*c,h,w)
#            rgb_input = rgb_input.view(rgb_input.shape[0],
#                                        -1,
#                                        rgb_input.shape[-2], rgb_input.shape[-1])
#            op_input = op_input.view(op_input.shape[0],
#                                      -1,
#                                      op_input.shape[-2], op_input.shape[-1])
#             # print("rgb_input, op_input: ", rgb_input.size(), op_input.size()) #
#            with torch.no_grad():
#                output = model(rgb_input, op_input)  # out: (b,c,h,w)
#            if net_tag == "unet" or net_tag == "unet_vq_twostream":
#                rgb_G_output, op_G_output, diff_tuple, _ = output
#                rgb_diff, op_diff = diff_tuple[0], diff_tuple[1]
#            for idx in range(rgb_G_output.size()[0]):  #
#
#                rgb_cnt += 1  #
#                gen = rgb_G_output[idx].unsqueeze(0)
#                gt = rgb_target[idx].unsqueeze(0)
#                 #
#                rgb_loss_func = loss_func_mapp[loss_name]
#                rgb_test_psnr = rgb_loss_func(gen, gt).item()
#                rgb_fea_comm_loss = rgb_diff #
#                rgb_psnrs[rgb_cnt + rgb_len_clip - 1] = rgb_test_psnr
#                rgb_fea_comm_losses[rgb_cnt + rgb_len_clip - 1] = rgb_fea_comm_loss
#                rgb_total += 1
#            for idx in range(rgb_G_output.size()[0]):  #
#                op_cnt += 1  #
#                gen = op_G_output[idx].unsqueeze(0)
#                gt = op_target[idx].unsqueeze(0)
#                 #
#                op_loss_func = loss_func_mapp[loss_name]
#                op_test_psnr = op_loss_func(gen, gt).item()
#                 #
#                op_fea_comm_loss = op_diff #
#                op_psnrs[op_cnt + op_len_clip - 1] = op_test_psnr
#                op_fea_comm_losses[op_cnt + op_len_clip - 1] = op_fea_comm_loss
#                op_total += 1
#        rgb_psnrs[:rgb_len_clip - 1] = rgb_psnrs[rgb_len_clip - 1]  #
#        op_psnrs[:op_len_clip - 1] = op_psnrs[op_len_clip - 1]  #
#        op_psnrs[num_frame-1] = op_psnrs[num_frame - 2]  #
#         #
#        rgb_fea_comm_losses[:rgb_len_clip - 1] = rgb_fea_comm_losses[rgb_len_clip - 1]  #
#        op_fea_comm_losses[:op_len_clip - 1] = op_fea_comm_losses[op_len_clip - 1]  #
#        op_fea_comm_losses[num_frame - 1] = op_fea_comm_losses[num_frame - 2]  #
#         #
#        rgb_psnr_records.append(rgb_psnrs)
#        op_psnr_records.append(op_psnrs)
#         #
#        rgb_fea_comm_records.append(rgb_fea_comm_losses)
#        op_fea_comm_records.append(op_fea_comm_losses)
#        logger.info('finish test video set {}'.format(sub_video_name))
#    rgb_result_dict = {'dataset': dataset_name,
#                        '{}'.format(loss_name): rgb_psnr_records,
#                        'rgb_fea_comm_loss': rgb_fea_comm_records,
#                         'flow': [], 'names': [], 'diff_mask': []}
#    op_result_dict = {'dataset': dataset_name,
#                       '{}'.format(loss_name): op_psnr_records,
#                       'op_fea_comm_loss': op_fea_comm_records,
#                        'flow': [], 'names': [], 'diff_mask': []}
#    used_time = time.time() - time_stamp
#    logger.info('total time = {}, fps = {}'.format(used_time, rgb_total / used_time))
#    with open(rgb_pickle_path, 'wb') as fp:
#        pickle.dump(rgb_result_dict, fp, pickle.HIGHEST_PROTOCOL)  #
#    with open(op_pickle_path, 'wb') as fp:
#        pickle.dump(op_result_dict, fp, pickle.HIGHEST_PROTOCOL)  #


# def gen_loss_file_twostream_normal_fea_comm_tmp(model, dataset_fn, net_tag,
#         data_root, len_clip, data_type,
#         dataset_name, rgb_pickle_path, op_pickle_path,
#         loss_name, logger):
#
#    time_stamp = time.time()
#     rgb_total, op_total = 0, 0  #
#     rgb_psnr_records, op_psnr_records = [], []
#     loss_func_mapp = { #
#         "psnr": utils.psnr_error,
#         "mse": utils.mse_error,
#         "ssim": utils.ssim_error,
#     }
#     rgb_root, op_root = data_root
#     # print("rgb_root, op_root: ", rgb_root, op_root) #
#     all_sub_video_name_list = os.listdir(rgb_root)
#     all_sub_video_name_list.sort()
#     print("all_sub_video_name_list: ", all_sub_video_name_list)
#     #
#     rgb_len_clip, op_len_clip = len_clip
#     for video_id, sub_video_name in enumerate(all_sub_video_name_list):
#         rgb_folder = os.path.join(rgb_root, sub_video_name)
#         rgb_ds = dataset_fn(rgb_folder, rgb_len_clip, "rgb")
#         op_folder = os.path.join(op_root, sub_video_name)
#         op_ds = dataset_fn(op_folder, op_len_clip, "op") #
#         #
#         rgb_loader = DataLoader(rgb_ds, batch_size=16,
#                                 shuffle=False, num_workers=4)
#         op_loader = DataLoader(op_ds, batch_size=16,
#                                shuffle=False, num_workers=4)
#         num_frame = len(rgb_ds) + rgb_len_clip - 1 #
#         print("len of sub_video: {}".format(num_frame))
#         rgb_psnrs = np.empty(shape=(num_frame,), dtype=np.float32)
#         op_psnrs = np.empty(shape=(num_frame,), dtype=np.float32) #
#         rgb_cnt, op_cnt = -1, -1
#         for idx, (rgb, op) in enumerate(zip(rgb_loader, op_loader)):
#             rgb = rgb.cuda()  # (1,t,c,h,w)
#             op = op.cuda()
#             rgb_input = rgb[:, :-1, :, :, :]
#             rgb_target = rgb[:, -1, :, :, :]
#             op_input = op[:, :-1, :, :, :]
#             op_target = op[:, :-1, :, :, :]
#             # as (1,t*c,h,w)
#             rgb_input = rgb_input.view(rgb_input.shape[0],
#                                        -1,
#                                        rgb_input.shape[-2], rgb_input.shape[-1])
#             op_input = op_input.view(op_input.shape[0],
#                                      -1,
#                                      op_input.shape[-2], op_input.shape[-1])
#             # print("rgb_input, op_input: ", rgb_input.size(), op_input.size()) #
#             with torch.no_grad():
#                 output = model(rgb_input, op_input)  # out: (b,c,h,w)
#             if net_tag == "unet" or net_tag == "unet_vq_twostream":
#                 rgb_G_output, op_G_output, diff_tuple, _ = output
#                 rgb_diff, op_diff = diff_tuple[0], diff_tuple[1]
#             for idx in range(rgb_G_output.size()[0]):  #
#                 rgb_cnt += 1  #
#                 gen = rgb_G_output[idx].unsqueeze(0)
#                 gt = rgb_target[idx].unsqueeze(0)
#                 rgb_loss_func = loss_func_mapp[loss_name]
#                 rgb_test_psnr = rgb_diff #
#                 rgb_psnrs[rgb_cnt + rgb_len_clip - 1] = rgb_test_psnr
#                 rgb_total += 1
#             for idx in range(rgb_G_output.size()[0]):  #
#                 op_cnt += 1
#                 gen = op_G_output[idx].unsqueeze(0)
#                 gt = op_target[idx].unsqueeze(0)
#                 op_loss_func = loss_func_mapp[loss_name]
#                 op_test_psnr = op_diff
#                 op_psnrs[op_cnt + op_len_clip - 1] = op_test_psnr
#                 op_total += 1
#         rgb_psnrs[:rgb_len_clip - 1] = rgb_psnrs[rgb_len_clip - 1]  #
#         op_psnrs[:op_len_clip - 1] = op_psnrs[op_len_clip - 1]  #
#         op_psnrs[num_frame-1] = op_psnrs[num_frame - 2]  #
#         #
#         rgb_psnr_records.append(rgb_psnrs)
#         op_psnr_records.append(op_psnrs)
#         logger.info('finish test video set {}'.format(sub_video_name))
#     #
#     rgb_result_dict = {'dataset': dataset_name, '{}'.format(loss_name): rgb_psnr_records,
#                    'flow': [], 'names': [], 'diff_mask': []}
#     op_result_dict = {'dataset': dataset_name, '{}'.format(loss_name): op_psnr_records,
#                        'flow': [], 'names': [], 'diff_mask': []}
#     used_time = time.time() - time_stamp
#     logger.info('total time = {}, fps = {}'.format(used_time, rgb_total / used_time))
#     with open(rgb_pickle_path, 'wb') as fp:
#         pickle.dump(rgb_result_dict, fp, pickle.HIGHEST_PROTOCOL)  #
#    with open(op_pickle_path, 'wb') as fp:
#         pickle.dump(op_result_dict, fp, pickle.HIGHEST_PROTOCOL)  #


def gen_loss_file_twostream_normal_all(model, dataset_fn, net_tag,
       data_root, len_clip,
       dataset_name, pickle_path,
       loss_name, logger):

    time_stamp = time.time()
    rgb_total, op_total = 0, 0
    rgb_img_pred_records, rgb_fea_comm_records = [], []
    op_img_pred_records, op_fea_comm_records = [], []
    loss_func_mapp = { #
        "psnr": utils.psnr_error,
        "mse": utils.mse_error,
        "ssim": utils.ssim_error,
    }
    rgb_root, op_root = data_root
    # print("rgb_root, op_root: ", rgb_root, op_root) #
    all_sub_video_name_list = os.listdir(rgb_root)
    all_sub_video_name_list.sort()
    print("all_sub_video_name_list: ", all_sub_video_name_list)
    #
    rgb_len_clip, op_len_clip = len_clip
    for video_id, sub_video_name in enumerate(all_sub_video_name_list):
        rgb_folder = os.path.join(rgb_root, sub_video_name)
        rgb_ds = dataset_fn(rgb_folder, rgb_len_clip, "rgb")
        op_folder = os.path.join(op_root, sub_video_name)
        op_ds = dataset_fn(op_folder, op_len_clip, "op") #
        #
        rgb_loader = DataLoader(rgb_ds, batch_size=16,
                                shuffle=False, num_workers=8)
        op_loader = DataLoader(op_ds, batch_size=16,
                               shuffle=False, num_workers=4)
        num_frame = len(rgb_ds) + rgb_len_clip - 1 #
        print("len of sub_video: {}".format(num_frame))
        rgb_img_pred_arr = np.empty(shape=(num_frame,), dtype=np.float32)
        rgb_fea_comm_arr = np.empty(shape=(num_frame,), dtype=np.float32)
        op_img_pred_arr = np.empty(shape=(num_frame,), dtype=np.float32)
        op_fea_comm_arr = np.empty(shape=(num_frame,), dtype=np.float32)
        rgb_cnt, op_cnt = -1, -1
        for idx, (rgb, op) in enumerate(zip(rgb_loader, op_loader)):
            rgb = rgb.cuda()  # (1,t,c,h,w)
            op = op.cuda()
            rgb_input = rgb[:, :-1, :, :, :]
            rgb_target = rgb[:, -1, :, :, :]
            op_input = op[:, :-1, :, :, :]
            op_target = op[:, :-1, :, :, :]
            # as (1,t*c,h,w)
            rgb_input = rgb_input.view(rgb_input.shape[0],
                                       -1,
                                       rgb_input.shape[-2], rgb_input.shape[-1])
            op_input = op_input.view(op_input.shape[0],
                                     -1,
                                     op_input.shape[-2], op_input.shape[-1])
            # print("rgb_input, op_input: ", rgb_input.size(), op_input.size()) #
            with torch.no_grad():
                output = model(rgb_input, op_input)  # out: (b,c,h,w)
            if net_tag == "unet_vq_twostream":
                rgb_G_output, op_G_output, diff_tuple, _ = output
                rgb_diff, op_diff = diff_tuple[0], diff_tuple[1]
            for idx in range(rgb_G_output.size()[0]):
                rgb_cnt += 1  #
                gen = rgb_G_output[idx].unsqueeze(0)
                gt = rgb_target[idx].unsqueeze(0)
                rgb_img_pred_func = loss_func_mapp["psnr"]
                rgb_img_pred = rgb_img_pred_func(gen, gt)
                rgb_fea_comm = rgb_diff
                rgb_img_pred_arr[rgb_cnt + rgb_len_clip - 1] = rgb_img_pred
                rgb_fea_comm_arr[rgb_cnt + rgb_len_clip - 1] = rgb_fea_comm
                rgb_total += 1
            for idx in range(rgb_G_output.size()[0]):
                op_cnt += 1  #
                gen = op_G_output[idx].unsqueeze(0)
                gt = op_target[idx].unsqueeze(0)
                op_img_pred_func = loss_func_mapp["psnr"]
                op_img_pred = op_img_pred_func(gen, gt)
                op_fea_comm = op_diff
                op_img_pred_arr[op_cnt + op_len_clip - 1] = op_img_pred
                op_fea_comm_arr[op_cnt + op_len_clip - 1] = op_fea_comm
                op_total += 1
        rgb_img_pred_arr[:rgb_len_clip - 1] = rgb_img_pred_arr[rgb_len_clip - 1]
        op_img_pred_arr[:op_len_clip - 1] = op_img_pred_arr[op_len_clip - 1]
        op_img_pred_arr[num_frame-1] = op_img_pred_arr[num_frame - 2]
        rgb_img_pred_records.append(rgb_img_pred_arr)
        op_img_pred_records.append(op_img_pred_arr)
        #
        rgb_fea_comm_arr[:rgb_len_clip - 1] = rgb_fea_comm_arr[rgb_len_clip - 1]  #
        op_fea_comm_arr[:op_len_clip - 1] = op_fea_comm_arr[op_len_clip - 1]  #
        op_fea_comm_arr[num_frame - 1] = op_fea_comm_arr[num_frame - 2]  #
        rgb_fea_comm_records.append(rgb_fea_comm_arr)
        op_fea_comm_records.append(op_fea_comm_arr)
        #
        logger.info('finish test video set {}'.format(sub_video_name))
    #
    result_dict = {'dataset': dataset_name,
                   'rgb_img_pred_records': rgb_img_pred_records,
                   'rgb_fea_comm_records': rgb_fea_comm_records,
                   'op_img_pred_records': op_img_pred_records,
                   'op_fea_comm_records': op_fea_comm_records,
                   }
    used_time = time.time() - time_stamp
    logger.info('total time = {}, fps = {}'.format(used_time, rgb_total / used_time))
    with open(pickle_path, 'wb') as fp:
        pickle.dump(result_dict, fp, pickle.HIGHEST_PROTOCOL)


def gen_mul_metrics_manual(test_save_dir, metric_name, lam):

    ret_dict = eval_metric.evaluate(metric_name,
                                        test_save_dir, lam)
    loss_file = ret_dict["optimal_loss"]
    auc = ret_dict["auc"]
    print("================================================================================ ")
    print("the optimal loss_file is: ", loss_file)
    print("the optimal auc = ", auc)
    print("================================================================================ ")


class test_single_Helper(object):
    """
    """
    def __init__(self, model, dataset, loss, optimizer,
                 params):
        """
        Class constructor.
        :param dataset: dataset class.
        :param model: pytorch model to evaluate.
        :param checkpoint: path of the checkpoint for the model.
        :param output_file: text file where to save results.
        """
        self.model = model
        self.dataset = dataset
        self.params = params # 直接传 const

    def evaluate_img_pred_fea_comm_twostream(self):
        print("==========================================")
        print("evaluate_img_pred_fea_comm_twostream")
        print("==========================================")
        model = self.model.generator
        #
        loss_name = self.params.loss_name
        metric_name = self.params.metric_name
        start_step = self.params.start_step
        summary_dir = self.params.summary_dir
        logger = self.params.logger
        test_load_ckpt = self.params.test_load_ckpt
        test_save_dir = self.params.test_save_dir
        len_clip = self.params.clip_length  # (5,4)
        dataset_name = self.params.dataset_name
        data_type = self.params.data_type
        dataset_fn = self.params.dataset_fn
        net_tag = self.params.net_tag
        data_root = self.params.video_folder
        #
        exp_tag = self.params.exp_tag
        loss_dir = utils.get_dir(os.path.join(test_save_dir, metric_name))
        save_pickle_root = utils.get_dir(os.path.join(loss_dir, "save_pickle"))
        print("save_pickle_root: ", save_pickle_root)
        print("test_load_ckpt: ", test_load_ckpt)

        # ============================================================================ #
        params = {
            "save_pickle_root": save_pickle_root,
            "loss_name": loss_name,
            "summary_dir": summary_dir
        }
        # params_save_path = metric_cal_need_info_save
        # exp_tag_params_map_file = exp_metric_cal_need_info_map
        # utils.save_params(params, params_save_path, exp_tag, exp_tag_params_map_file)
        # ============================================================================ #

        load_ckpt_path = test_load_ckpt
        pickle_path = os.path.join(save_pickle_root, dataset_name)
        model.load_state_dict(torch.load(load_ckpt_path, map_location="cuda:0"))
        model.cuda().eval()
        #
        gen_loss_file_twostream_normal_all(model, dataset_fn, net_tag,
                                        data_root, len_clip,
                                        dataset_name, pickle_path,
                                        loss_name, logger)
        lam_map = {
            "avenue":(0.04, 0.65),
            "ped2":(0.01, 0.55),
            "shanghaitech": (0.13, 0.60),
        }
        gen_mul_metrics_manual(save_pickle_root, metric_name, lam_map[dataset_name])

