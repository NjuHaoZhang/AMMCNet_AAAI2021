import numpy as np
import scipy.io as scio
import os
import argparse
import pickle
from sklearn import metrics
import json
import socket
import matplotlib.pyplot as plt

# config
DATA_DIR = '/p300/dataset' # const.data_dir_gt, dataset_root_dir

#
NORMALIZE = True 
num_his = 4 
DECIDABLE_IDX = num_his


class RecordResult(object):
    def __init__(self, fpr=None, tpr=None, auc=-np.inf, dataset=None, loss_file=None, lam_rgb_fea_comm=None, lam_smooth=None):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = auc
        self.dataset = dataset
        self.loss_file = loss_file
        self.lam_rgb_fea_comm = lam_rgb_fea_comm
        self.lam_smooth = lam_smooth

    def __lt__(self, other):
        return self.auc < other.auc

    def __gt__(self, other):
        return self.auc > other.auc

    def __str__(self):
        return 'dataset = {}, loss file = {}, auc = {}, lam_rgb_fea_comm={}, ' \
               'lam_smooth={}'.format(self.dataset, self.loss_file, self.auc, self.lam_rgb_fea_comm, self.lam_smooth)


class GroundTruthLoader(object):
    AVENUE = 'avenue'
    PED1 = 'ped1'
    PED1_PIXEL_SUBSET = 'ped1_pixel_subset'
    PED2 = 'ped2'
    ENTRANCE = 'enter'
    EXIT = 'exit'
    SHANGHAITECH = 'shanghaitech'
    SHANGHAITECH_LABEL_PATH = os.path.join(DATA_DIR, 'shanghaitech/testing/test_frame_mask')
    TOY_DATA = 'toydata'
    TOY_DATA_LABEL_PATH = os.path.join(DATA_DIR, TOY_DATA, 'toydata.json')

    NAME_MAT_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/avenue.mat'),
        PED1: os.path.join(DATA_DIR, 'ped1/ped1.mat'),
        PED2: os.path.join(DATA_DIR, 'ped2/ped2.mat'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/enter.mat'),
        EXIT: os.path.join(DATA_DIR, 'exit/exit.mat')
    }

    NAME_FRAMES_MAPPING = {
        AVENUE: os.path.join(DATA_DIR, 'avenue/testing/frames'),
        PED1: os.path.join(DATA_DIR, 'ped1/testing/frames'),
        PED2: os.path.join(DATA_DIR, 'ped2/testing/frames'),
        ENTRANCE: os.path.join(DATA_DIR, 'enter/testing/frames'),
        EXIT: os.path.join(DATA_DIR, 'exit/testing/frames')
    }

    def __init__(self, mapping_json=None):
        """
        Initial a ground truth loader, which loads the ground truth with given dataset name.

        :param mapping_json: the mapping from dataset name to the path of ground truth.
        """

        if mapping_json is not None:
            with open(mapping_json, 'rb') as json_file:
                self.mapping = json.load(json_file)
        else:
            self.mapping = GroundTruthLoader.NAME_MAT_MAPPING

    def __call__(self, dataset):
        """ get the ground truth by provided the name of dataset.

        :type dataset: str
        :param dataset: the name of dataset.
        :return: np.ndarray, shape(#video)
                 np.array[0] contains all the start frame and end frame of abnormal events of video 0,
                 and its shape is (#frapsnr, )
        """

        if dataset == GroundTruthLoader.SHANGHAITECH:
            gt = self.__load_shanghaitech_gt()
        elif dataset == GroundTruthLoader.TOY_DATA:
            gt = self.__load_toydata_gt()
        else:
            gt = self.__load_ucsd_avenue_subway_gt(dataset)
        return gt

    def __load_ucsd_avenue_subway_gt(self, dataset):
        assert dataset in self.mapping, 'there is no dataset named {} \n Please check {}' \
            .format(dataset, GroundTruthLoader.NAME_MAT_MAPPING.keys())

        mat_file = self.mapping[dataset]
        abnormal_events = scio.loadmat(mat_file, squeeze_me=True)['gt']

        if abnormal_events.ndim == 2:
            abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0], abnormal_events.shape[1])

        num_video = abnormal_events.shape[0]
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        assert num_video == len(video_list), 'ground true does not match the number of testing videos. {} != {}' \
            .format(num_video, len(video_list))

        # get the total frames of sub video
        def get_video_length(sub_video_number):
            # video_name = video_name_template.format(sub_video_number)
            video_name = os.path.join(dataset_video_folder, video_list[sub_video_number])
            assert os.path.isdir(video_name), '{} is not directory!'.format(video_name)

            length = len(os.listdir(video_name))

            return length

        # need to test [].append, or np.array().append(), which one is faster
        gt = []
        for i in range(num_video):
            length = get_video_length(i)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            sub_abnormal_events = abnormal_events[i]
            if sub_abnormal_events.ndim == 1:
                sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))

            _, num_abnormal = sub_abnormal_events.shape

            for j in range(num_abnormal):
                # (start - 1, end - 1)
                start = sub_abnormal_events[0, j] - 1
                end = sub_abnormal_events[1, j]

                sub_video_gt[start: end] = 1

            gt.append(sub_video_gt)

        return gt

    @staticmethod
    def __load_shanghaitech_gt():
        video_path_list = os.listdir(GroundTruthLoader.SHANGHAITECH_LABEL_PATH)
        video_path_list.sort()

        gt = []
        for video in video_path_list:
            # print(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video))
            gt.append(np.load(os.path.join(GroundTruthLoader.SHANGHAITECH_LABEL_PATH, video)))

        return gt

    @staticmethod
    def __load_toydata_gt():
        with open(GroundTruthLoader.TOY_DATA_LABEL_PATH, 'r') as gt_file:
            gt_dict = json.load(gt_file)

        gt = []
        for video, video_info in gt_dict.items():
            length = video_info['length']
            video_gt = np.zeros((length,), dtype=np.uint8)
            sub_gt = np.array(np.matrix(video_info['gt']))

            for anomaly in sub_gt:
                start = anomaly[0]
                end = anomaly[1] + 1
                video_gt[start: end] = 1
            gt.append(video_gt)
        return gt

    @staticmethod
    def get_pixel_masks_file_list(dataset):
        # pixel mask folder
        pixel_mask_folder = os.path.join(DATA_DIR, dataset, 'pixel_masks')
        pixel_mask_file_list = os.listdir(pixel_mask_folder)
        pixel_mask_file_list.sort()

        # get all testing videos
        dataset_video_folder = GroundTruthLoader.NAME_FRAMES_MAPPING[dataset]
        video_list = os.listdir(dataset_video_folder)
        video_list.sort()

        # get all testing video names with pixel masks
        pixel_video_ids = []
        ids = 0
        for pixel_mask_name in pixel_mask_file_list:
            while ids < len(video_list):
                if video_list[ids] + '.npy' == pixel_mask_name:
                    pixel_video_ids.append(ids)
                    ids += 1
                    break
                else:
                    ids += 1

        assert len(pixel_video_ids) == len(pixel_mask_file_list)

        for i in range(len(pixel_mask_file_list)):
            pixel_mask_file_list[i] = os.path.join(pixel_mask_folder, pixel_mask_file_list[i])

        return pixel_mask_file_list, pixel_video_ids


def load_psnr(loss_file):
    """
    load image psnr or optical flow psnr.
    :param loss_file: loss file path
    :return:
    """
    with open(loss_file, 'rb') as reader:
        # results {
        #   'dataset': the name of dataset
        #   'psnr': the psnr of each testing videos,
        # }

        # psnr_records['psnr'] is np.array, shape(#videos)
        # psnr_records[0] is np.array   ------>     01.avi
        # psnr_records[1] is np.array   ------>     02.avi
        #               ......
        # psnr_records[n] is np.array   ------>     xx.avi

        results = pickle.load(reader)
    psnrs = results['psnr']
    return psnrs


def load_psnr_gt(loss_file):
    try:
        with open(loss_file, 'rb') as reader:
            results = pickle.load(reader)
            # results {
            #   'dataset': the name of dataset
            #   'psnr': the psnr of each testing videos,
            # }

            # psnr_records['psnr'] is np.array, shape(#videos)
            # psnr_records[0] is np.array   ------>     01.avi
            # psnr_records[1] is np.array   ------>     02.avi
            #               ......
            # psnr_records[n] is np.array   ------>     xx.avi

            dataset = results['dataset']
            psnr_records = results['psnr']

            num_videos = len(psnr_records)

            # load ground truth
            gt_loader = GroundTruthLoader()
            gt = gt_loader(dataset=dataset)

            assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
                .format(num_videos, len(gt))

            return dataset, psnr_records, gt
    except EOFError:
        return None, None, None


def get_scores_labels(loss_file):
    # the name of dataset, loss, and ground truth
    dataset, psnr_records, gt = load_psnr_gt(loss_file=loss_file)

    # the number of videos
    num_videos = len(psnr_records)

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)
    # video normalization
    for i in range(num_videos):
        distance = psnr_records[i]

        if NORMALIZE:
            distance -= distance.min()  # distances = (distance - min) / (max - min)
            distance /= distance.max()
            # distance = 1 - distance

        scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:]), axis=0)
        labels = np.concatenate((labels[:], gt[i][DECIDABLE_IDX:]), axis=0)
    return dataset, scores, labels


def precision_recall_auc(loss_file):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult()
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file)
        precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=0)
        auc = metrics.auc(recall, precision)

        results = RecordResult(recall, precision, auc, dataset, sub_loss_file)

        if optimal_results < results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    print('##### optimal result and model = {}'.format(optimal_results))
    return optimal_results


def cal_eer(fpr, tpr):
    # makes fpr + tpr = 1
    eer = fpr[np.nanargmin(np.absolute((fpr + tpr - 1)))]
    return eer


def compute_eer(loss_file):
    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file) for sub_loss_file in loss_file_list]

    optimal_results = RecordResult(auc=np.inf)
    for sub_loss_file in loss_file_list:
        dataset, scores, labels = get_scores_labels(sub_loss_file)
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        eer = cal_eer(fpr, tpr)

        results = RecordResult(fpr, tpr, eer, dataset, sub_loss_file)

        if optimal_results > results:
            optimal_results = results

        if os.path.isdir(loss_file):
            print(results)
    return optimal_results


def load_img_pred_fea_comm_gt(loss_file):
    try:
        with open(loss_file, 'rb') as reader:
            results = pickle.load(reader)
            # results {
            #   'dataset': the name of dataset
            #   'psnr': the psnr of each testing videos,
            # }

            # psnr_records['psnr'] is np.array, shape(#videos)
            # psnr_records[0] is np.array   ------>     01.avi
            # psnr_records[1] is np.array   ------>     02.avi
            #               ......
            # psnr_records[n] is np.array   ------>     xx.avi

            dataset = results['dataset']

            rgb_img_pred_records = results['rgb_img_pred_records']
            rgb_fea_comm_records = results['rgb_fea_comm_records']
            op_img_pred_records = results['op_img_pred_records']
            op_fea_comm_records = results['op_fea_comm_records']

            num_videos = len(rgb_img_pred_records)

            # load ground truth
            gt_loader = GroundTruthLoader()
            gt = gt_loader(dataset=dataset)

            assert num_videos == len(gt), 'the number of saved videos does not match the ground truth, {} != {}' \
                .format(num_videos, len(gt))

            return dataset, rgb_img_pred_records, rgb_fea_comm_records, \
                   op_img_pred_records, op_fea_comm_records, gt

    except EOFError:
        return None, None, None


def img_pred_fea_comm_single_auc(loss_file, lam=(0.5,0.5)):

    if not os.path.isdir(loss_file):
        loss_file_list = [loss_file]
    else:
        loss_file_list = os.listdir(loss_file)
        loss_file_list = [os.path.join(loss_file, sub_loss_file)
                          for sub_loss_file in loss_file_list]
    optimal_results = RecordResult() 

    for sub_loss_file in loss_file_list:
        dataset, rgb_img_pred_records, rgb_fea_comm_records, \
        op_img_pred_records, op_fea_comm_records, gt = load_img_pred_fea_comm_gt(loss_file=sub_loss_file)

        if dataset is None:
            continue 

        num_videos = len(rgb_img_pred_records)

        labels = np.array([], dtype=np.int8)
        for i in range(num_videos):
            labels = np.concatenate((labels, gt[i][DECIDABLE_IDX:]), axis=0)
        
        def norm_score(num_videos, records):
            scores = np.array([], dtype=np.float32)
            for i in range(num_videos):
                distance = records[i]  # a sub_video loss_record
                if NORMALIZE:
                    distance -= distance.min()  # distances = (distance - min) / (max - min)
                    distance /= distance.max()
                scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
            if NORMALIZE:  # whole video normalize
                scores -= scores.min()  # scores = (scores - min) / (max - min)
                scores /= scores.max()

            return scores
        img_scores = norm_score(num_videos, rgb_img_pred_records) 
        fea_scores = norm_score(num_videos, rgb_fea_comm_records) 
        identity = np.ones_like(fea_scores)
        #
        lam_rgb_fea_comm_list = [x * 0.01 for x in range(0, 100)]
        lam_smooth_list = [x * 0.05 for x in range(0, 20)]

        lam_rgb_fea_comm, lam_smooth = lam[0], lam[1]
        scores = (1-lam_rgb_fea_comm) * img_scores + lam_rgb_fea_comm * (identity - fea_scores)
        scores = [(1-lam_smooth)*scores[idx-1]+lam_smooth*scores[idx] if idx>0 else scores[idx] for idx in range(len(scores))]
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        results = RecordResult(fpr, tpr, auc, dataset, sub_loss_file, lam_rgb_fea_comm, lam_smooth)
        
        if optimal_results < results:
            optimal_results = results
        
        ret_dict = {}
        ret_dict["optimal_loss"] = "{}".format(optimal_results.loss_file)
        ret_dict["auc"] = round(optimal_results.auc,3)
    
    return ret_dict


eval_type_function = {
    'compute_eer': compute_eer,
    'precision_recall_auc': precision_recall_auc,
    'img_pred_fea_comm_rgb_auc': img_pred_fea_comm_single_auc,
}


def evaluate(eval_type, save_file, lam=None):
    assert eval_type in eval_type_function, 'there is no type of evaluation {}, please check {}' \
        .format(eval_type, eval_type_function.keys())
    eval_func = eval_type_function[eval_type]
    optimal_results = eval_func(save_file, lam)
    return optimal_results
