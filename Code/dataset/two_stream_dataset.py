import os,glob,sys
from collections import OrderedDict
from multiprocessing import Pool#
# from pathos.multiprocessing import ProcessingPoll as Pool #
import time,threading
import time,threading
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import gc


import cv2
import numpy as np
from torch.utils.data import Dataset
# from .torch_videovision.videotransforms import (
#     video_transforms, volume_transforms)
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
#
# torch.multiprocessing.set_sharing_strategy('file_system')
#
from tensorboardX import SummaryWriter
# from PIL import Image

from ..utils.flowlib import readFlow, flow_to_image, batch_flow_to_image
from ..utils import utils
from ..utils.img_process import img_dec_TurboJPEG

rng = np.random.RandomState(2017)



def _load_frame_op(args):
    img_path, img_size, transform, flag = args
    fn_map = {
        "rgb": _load_frame,
        "op": _load_op,
    }
    fn = fn_map[flag]

    return fn(img_path, img_size, transform)

# def _load_frame(args):
#     img_path, img_size, transform = args
#     image_width, image_height = img_size
#     img = img_decode(img_path)  # 代替 cv2.imread
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (image_width, image_height))  # 注意要 (w,h)
#     # perform transform, TODO
#     if transform:
#         # img = Image.fromarray(img)
#         img = transform(img)
#
#     return img
#
# def _load_op(args):
#     img_path, img_size, transform = args
#     image_width, image_height = img_size
#     img = readFlow(img_path)  # Note: output [h, w, c]
#     # print(img.shape)
#     img = cv2.resize(img, (image_width, image_height))
#     # 注意：cv2的 resize的 input 是 w,h 而不是一般的 h,w
#     # print(img.shape)
#     if transform:
#         img = transform(img)
#         # img = torch.tensor(img).view(0,2,1)
#
#     return img

def _load_frame(img_path, img_size=(256,256), transform=None):
    image_width, image_height = img_size
    img = img_decode(img_path)  # 代替 cv2.imread
    # print("img-1: ", img.dtype)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_width, image_height))
    # perform transform, TODO
    if transform:
        # img = Image.fromarray(img)
        img = transform(img)
    # print("img-2: ", img.dtype)
    return img # .astype(np.float)

def _load_op(img_path, img_size=(256,256), transform=None):
    image_width, image_height = img_size
    # print("img_path: ", img_path)
    img = readFlow(img_path)  # Note: output [h, w, c]
    # print(img.shap3121212121212121212121212121212./le)
    img = cv2.resize(img, (image_width, image_height))
    # 注意：cv2的 resize的 input 是 w,h 而不是一般的 h,w
    # print(img.shape)
    # if transform:
    img[:,:,0] = img[:,:,0] * 1.0 / image_height # since value is in (-h, h)
    img[:,:,1] = img[:,:,0] * 1.0 / image_width
    img = np.transpose(img, [2, 0, 1]) # (h,w,2) -> (2,h,w)
    img = torch.from_numpy(img)

    return img

def img_decode(img_path):
    '''
    '''
    # return cv2.imread(img_path)  # Note: output [h, w, c] with BGR, uint8
    return img_dec_TurboJPEG(img_path) # output [h, w, c] with BGR, uint8


class clip_Test_DS(Dataset):

    def __init__(self, video_folder, data_type, clip_length=10, size=(256, 256), n_worker = 16):
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10
        self.data_type = data_type
        self.n_worker = n_worker
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  # 当前处于哪个 sub_video, 即 01, 02...
        # self.cur_sub_video_frames = None
        # transform 默认仅仅是 Totensor()
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(), #
                    # transforms.Resize(size), #
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size), #
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]
            self.videos[sub_video_name] = {}  #
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*'))
            self.videos[sub_video_name]['frame'].sort()
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    def test(self, sub_video_name):
        self.cur_sub_video_name = sub_video_name
        self.cur_len = self.videos[sub_video_name]['length'] - \
                       self.clip_length+ 1
        #

        img_path_list = self.videos[sub_video_name]['frame']
        # print("img_path_list: ", img_path_list)
        tmp_clip = []
        data_type = self.data_type
        fn_map = {
            "rgb": _load_frame,
            "op":_load_op,
        }
        worker = fn_map[data_type]
        n_worker = self.n_worker
        #
        img_size_list = [(self.image_height, self.image_width) for idx in range(len(img_path_list))]
        transform_list = [self.transform for idx in range(len(img_path_list))]
        zip_args = list(zip(img_path_list, img_size_list, transform_list))

        with Pool(n_worker) as pool:
            for img in pool.starmap(worker, zip_args): #
                tmp_clip.append(img)
        self.cur_sub_vid_clip = torch.stack(tmp_clip)

    
    # @classmethod
    # def worker(cls, img_path, data_type, img_size, transform):
    #     if data_type == "rgb":
    #         frame_clip = cls._load_frames([img_path],img_size, transform)
    #     elif data_type == "op":
    #         frame_clip = cls._load_ops([img_path], img_size, transform)
    #     else:
    #         print("clip_Test_DS error")
    #         exit()
    #
    #     return frame_clip[0]

    def __len__(self):
        return self.cur_len # cur_sub_video_op, len(dataset),即 num of getitem()

    def __getitem__(self, idx):
        return self.cur_sub_vid_clip[idx:idx + self.clip_length, :,:,:]


    # ----------------------------------------------------- #
    @classmethod
    def _load_frames(cls, img_list, img_size, transform):
        image_width, image_height = img_size
        all_clip = []

        for img_path in img_list:       # [start,end)
            # print("img_path: ", img_path)
            img = cls.img_decode(img_path) #
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_width, image_height)) # 注意要 (w,h)
            # perform transform, TODO
            if transform:
                # img = Image.fromarray(img)
                img = transform(img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  #

        return test_clip

    @classmethod
    def _load_ops(cls, img_list, img_size, transform):
        image_width, image_height = img_size
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (image_width, image_height))
            # print(img.shape)
            if transform:
                img = transform(img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)

        return test_clip
    # --------------------------------------------------------#


class clip_Train_DS(Dataset):

    def __init__(self, video_folder, data_type, clip_length=10, size=(256, 256)):
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length #
        self.data_type = data_type
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0
        self.cur_sub_video_name = None
        # self.cur_sub_video_frames = None
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(),
                    # transforms.Resize(size), #
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size),
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03
            self.videos[sub_video_name] = {}  #
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*')) # jpg and flo
            self.videos[sub_video_name]['frame'].sort()  #
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    def __len__(self):
        return self._get_clips_total_num() # all_clip num

    def _get_clips_total_num(self):
        sub_video_name_list = sorted(list(self.videos.keys()))
        clips_num_list = [ self.videos[sub_video_name]['length']-self.clip_length+1
                for sub_video_name in sub_video_name_list]

        return sum(clips_num_list)

    def __getitem__(self, idx):
        sub_video_name_list = sorted(list(self.videos.keys()))
        sub_vid = rng.randint(0, len(sub_video_name_list))
        sub_video_name = sub_video_name_list[sub_vid]
        cur_cid = rng.randint(0, self.videos[sub_video_name]['length']
                              - self.clip_length)
        # randint: [`low`, `high`)
        frame_path_list = self.videos[sub_video_name]['frame'][cur_cid:
            cur_cid + self.clip_length]
        # (3) load frame and op
        if self.data_type == "rgb":
            frame_clip = self._load_frames(frame_path_list)
        elif self.data_type == "op":
            frame_clip = self._load_ops(frame_path_list)
        else:
            print("clip_Train_DS error")
            exit()
        sample = frame_clip

        return sample

    def _load_frames(self, img_list):
        all_clip = []
        for img_path in img_list:       # [start,end)
            img = self.img_decode(img_path) #
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_width, self.image_height)) #
            # perform transform, TODO
            if self.transform:
                # img = Image.fromarray(img)
                img = self.transform(img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  #

        return test_clip

    def _load_ops(self, img_list):
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (self.image_width, self.image_height))
            img[:, :, 0] = img[:, :, 0] * 1.0 / self.image_height  # since value is in (-h, h)
            img[:, :, 1] = img[:, :, 0] * 1.0 / self.image_width
            img = np.transpose(img, [2, 0, 1])  # (h,w,2) -> (2,h,w)
            img = torch.from_numpy(img)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  #

        return test_clip

    def img_decode(self, img_path):
        # return cv2.imread(img_path)  # Note: output [h, w, c] with BGR
        return img_dec_TurboJPEG(img_path) # output [h, w, c] with BGR

# -------------------------------------  #

class clip_Train_DS_debug(Dataset):
    def __init__(self, video_folder, data_type, clip_length=10, size=(256, 256)):
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10
        self.data_type = data_type
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  #
        # self.cur_sub_video_frames = None
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(), #
                    # transforms.Resize(size), #
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size), #
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]  #
            self.videos[sub_video_name] = {}  #
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*')) # jpg and flo
            self.videos[sub_video_name]['frame'].sort()  #
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    def __len__(self):
        return self._get_clips_total_num() # all_clip num

    def _get_clips_total_num(self):
        sub_video_name_list = sorted(list(self.videos.keys()))
        clips_num_list = [ self.videos[sub_video_name]['length']-self.clip_length+1
                for sub_video_name in sub_video_name_list]

        return sum(clips_num_list)

    def __getitem__(self, idx):
        sub_video_name_list = sorted(list(self.videos.keys()))
        # sub_vid = rng.randint(0, len(sub_video_name_list))
        sub_vid = 0 #
        sub_video_name = sub_video_name_list[sub_vid]
        # cur_cid = rng.randint(0, self.videos[sub_video_name]['length']
        #                       - self.clip_length)
        cur_cid = 0
        #
        # randint: [`low`, `high`)
        frame_path_list = self.videos[sub_video_name]['frame'][cur_cid:
            cur_cid + self.clip_length]
        # (3) load frame and op
        if self.data_type == "rgb":
            frame_clip = self._load_frames(frame_path_list)
        elif self.data_type == "op":
            frame_clip = self._load_ops(frame_path_list)
        else:
            print("clip_Train_DS error")
            exit()
        sample = frame_clip

        return sample

    def _load_frames(self, img_list):
        all_clip = []
        for img_path in img_list:       # [start,end)
            img = self.img_decode(img_path) #
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_width, self.image_height)) #
            # perform transform, TODO
            if self.transform:
                # img = Image.fromarray(img)
                img = self.transform(img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  #

        return test_clip

    def _load_ops(self, img_list):
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (self.image_width, self.image_height))
            # print(img.shape)
            if self.transform:
                img = self.transform(img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  #
        return test_clip

    def img_decode(self, img_path):
        # return cv2.imread(img_path)  # Note: output [h, w, c] with BGR
        return img_dec_TurboJPEG(img_path) # output [h, w, c] with BGR
# -------------------------------------- #

# ******************************************************************************** #
class TwoStream_Train_DS(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256)):
        self.rgb_ds = clip_Train_DS(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0])
        self.op_ds = clip_Train_DS(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1])
    def __len__(self):
        assert len(self.rgb_ds) == len(self.op_ds), "TwoStream_Test_DS error"
        return len(self.rgb_ds)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}

#

def np_load_frame(filename, size=(256,256), transform=None):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [0, 1].
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    resize_width, resize_height = size
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = image_resized / 255.0 # (0,255) -> (0,1)
    image_resized=np.transpose(image_resized,[2,0,1])
    return image_resized

class test_dataset(Dataset):
    # if use have to be very carefully
    # not cross the boundary

    def __init__(self, video_folder, clip_length, data_type, size=(256, 256)):
        self.data_type = data_type
        self.path = video_folder
        self.clip_length = clip_length
        self.img_height, self.img_width = size
        self.setup()
        #
        transform_map = {
            "rgb": transforms.Compose([
                transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
            "op": transforms.Compose([
                transforms.ToTensor(),  # channel_nb=2, div_255=False, numpy=False,
            ]),
        }
        self.transform = transform_map[data_type]

    def setup(self):
        self.pics = glob.glob(os.path.join(self.path, '*'))
        self.pics.sort()
        self.pics_len = len(self.pics)

    def __len__(self):
        # return self.pics_len
        return self.pics_len - self.clip_length + 1  # num_clip

    def __getitem__(self, indice):
        pic_clips = []
        fn_map = {
            "rgb": _load_frame,
            "op": _load_op,
        }
        fn = fn_map[self.data_type]
        for frame_id in range(indice, indice + self.clip_length):
            tmp = fn(img_path=self.pics[frame_id], transform=self.transform)
            pic_clips.append(tmp)
        # pic_clips = np.array(pic_clips)
        # pic_clips = torch.from_numpy(pic_clips).float() #
        # pic_clips = pic_clips.permute(0,3,1,2) # t,h,w,c -> t,c,h,w
        pic_clips = torch.stack(pic_clips)
        # print("pic_clips: ", pic_clips.size(), pic_clips.min(), pic_clips.max()) #
        return pic_clips #
# ********************************************************************************* #

# -------------------------------------------- #
class clip_Test_DS_v1(Dataset):

    def __init__(self, video_folder, data_type, clip_length=10, size=(256, 256)):
        super(clip_Test_DS, self).__init__()
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10
        self.data_type = data_type
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  # 当前处于哪个 sub_video, 即 01, 02...
        # self.cur_sub_video_frames = None
        # transform 默认仅仅是 Totensor()
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size), # 丢到 load 里面实现
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos[sub_video_name] = {}  #
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*'))
            self.videos[sub_video_name]['frame'].sort()  # 一定要排序，确保order正确，在glob前面加个sorted ?
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    def test(self, sub_video_name):
        self.cur_sub_video_name = sub_video_name
        self.cur_len = self.videos[sub_video_name]['length'] - \
                       self.clip_length+ 1
        # print(self.videos["rgb"][sub_video_name]['length'])
        # print(self.clip_length[0])
        # print("cur len", self.cur_len) # 用 assert
        # 注意 self.videos["op"][sub_video_name]['frame'] 和
        # self.videos["op"][sub_video_name]['frame'] 已经预加载了 全部 frame 的 path
        #

    def __len__(self):
        return self.cur_len # cur_sub_video_op, len(dataset),即 num of getitem()

    def __getitem__(self, idx):
        
        sub_video_name = self.cur_sub_video_name
        frame_path_list = self.videos[sub_video_name]['frame'][idx: idx+self.clip_length]
        # (3) load frame and op
        if self.data_type == "rgb":
            frame_clip = self._load_frames(frame_path_list)
        elif self.data_type == "op":
            frame_clip = self._load_ops(frame_path_list)
        else:
            print("clip_Test_DS error")
            exit()
        sample = frame_clip

        return sample

    def _load_frames(self, img_list):
        all_clip = []
        for img_path in img_list:       # [start,end)
            img = self.img_decode(img_path) # 代替 cv2.imread
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_width, self.image_height)) # 注意要 (w,h)
            # perform transform, TODO
            if self.transform:
                # img = Image.fromarray(img)
                img = self.transform(img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  #

        return test_clip

    def _load_ops(self, img_list):
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (self.image_width, self.image_height))
            # 注意：cv2的 resize的 input 是 w,h 而不是一般的 h,w
            # print(img.shape)
            if self.transform:
                img = self.transform(img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中
        return test_clip

    def img_decode(self, img_path):
        # return cv2.imread(img_path)  # Note: output [h, w, c] with BGR
        return img_dec_TurboJPEG(img_path) # output [h, w, c] with BGR

#
class clip_Test_DS_v2(Dataset):

    def __init__(self, video_folder, data_type, clip_length=10, size=(256, 256), n_worker = 16):
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10
        self.data_type = data_type
        self.n_worker = n_worker
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  # 当前处于哪个 sub_video, 即 01, 02...
        # self.cur_sub_video_frames = None
        # transform 默认仅仅是 Totensor()
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size), # 丢到 load 里面实现
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos[sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*'))
            self.videos[sub_video_name]['frame'].sort()  # 一定要排序，确保order正确，在glob前面加个sorted ?
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    def test(self, sub_video_name):
        self.cur_sub_video_name = sub_video_name
        self.cur_len = self.videos[sub_video_name]['length'] - \
                       self.clip_length+ 1
        #

        img_path_list = self.videos[sub_video_name]['frame']
        # print("img_path_list: ", img_path_list)
        tmp_clip = []
        data_type = self.data_type
        fn_map = {
            "rgb": _load_frame,
            "op":_load_op,
        }
        worker = fn_map[data_type]
        n_worker = self.n_worker
        #
        img_size_list = [(self.image_height, self.image_width) for idx in range(len(img_path_list))]
        transform_list = [self.transform for idx in range(len(img_path_list))]
        zip_args = list(zip(img_path_list, img_size_list, transform_list))

        with ThreadPoolExecutor(max_workers=n_worker) as pool:
            for img in pool.map(worker, zip_args): # zip_args 必须要自己整理
                tmp_clip.append(img)
        self.cur_sub_vid_clip = torch.stack(tmp_clip)

    # @classmethod
    # def worker(cls, img_path, data_type, img_size, transform):
    #     if data_type == "rgb":
    #         frame_clip = cls._load_frames([img_path],img_size, transform)
    #     elif data_type == "op":
    #         frame_clip = cls._load_ops([img_path], img_size, transform)
    #     else:
    #         print("clip_Test_DS error")
    #         exit()
    #
    #     return frame_clip[0]

    def __len__(self):
        return self.cur_len # cur_sub_video_op, len(dataset),即 num of getitem()

    def __getitem__(self, idx):
        return self.cur_sub_vid_clip[idx:idx + self.clip_length, :,:,:]


    # ----------------------------------------------------- #
    @classmethod
    def _load_frames(cls, img_list, img_size, transform):
        image_width, image_height = img_size
        all_clip = []

        for img_path in img_list:       # [start,end)
            # print("img_path: ", img_path)
            img = cls.img_decode(img_path) # 代替 cv2.imread
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_width, image_height)) # 注意要 (w,h)
            # perform transform, TODO
            if transform:
                # img = Image.fromarray(img)
                img = transform(img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  # 合并为更高一维, 把所有的frame都合并到一个大 tensor中

        return test_clip

    @classmethod
    def _load_ops(cls, img_list, img_size, transform):
        image_width, image_height = img_size
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (image_width, image_height))
            # print(img.shape)
            if transform:
                img = transform(img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中

        return test_clip
    # --------------------------------------------------------#

# -------------------------------------------- #
class TwoStream_Test_DS_v1(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=2):
        self.rgb_ds = clip_Test_DS(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = clip_Test_DS(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
    def __len__(self):
        assert len(self.rgb_ds) == len(self.op_ds), "TwoStream_Test_DS error"
        return len(self.rgb_ds)

    def test(self, sub_vid):  # 外部调用,设置sub_vid

        self.rgb_ds.test(sub_vid)
        self.op_ds.test(sub_vid)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}

# ----------------------------------- #

class ds_base(Dataset):

    def __init__(self, video_folder, data_type, clip_length=10, size=(256, 256), n_worker = 4):
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10
        self.data_type = data_type
        self.n_worker = n_worker
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  # 当前处于哪个 sub_video, 即 01, 02...
        # self.cur_sub_video_frames = None
        # transform 默认仅仅是 Totensor()
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size), # 丢到 load 里面实现
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos[sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*'))
            self.videos[sub_video_name]['frame'].sort()  # 一定要排序，确保order正确，在glob前面加个sorted ?
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    
    # @classmethod
    # def worker(cls, img_path, data_type, img_size, transform):
    #     if data_type == "rgb":
    #         frame_clip = cls._load_frames([img_path],img_size, transform)
    #     elif data_type == "op":
    #         frame_clip = cls._load_ops([img_path], img_size, transform)
    #     else:
    #         print("clip_Test_DS error")
    #         exit()
    #
    #     return frame_clip[0]

    def __len__(self):
        return self.cur_len # cur_sub_video_op, len(dataset),即 num of getitem()

    def __getitem__(self, idx):
        return self.cur_sub_vid_clip[idx:idx + self.clip_length, :,:,:]


class TwoStream_Test_DS_v2(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=4):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), #
                    # transforms.Resize(size), #
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), #
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        self.cur_sub_video_name = sub_video_name
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list))

        # print("img_path_list: ", img_path_list)
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        # starmap 与 map 性能相同，但支持多参数的func，但是 所有参数都要整理为 同样shape的 iterable
        with Pool(n_worker) as pool:
            for img in pool.starmap(_load_frame, rgb_zip_args): # zip_args 必须要自己整理
                # 必须声明为 类方法才能直接传入 multiprocess
                rgb_tmp_clip.append(img)
            for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
                # 必须声明为 类方法才能直接传入 multiprocess
                op_tmp_clip.append(img)
        #
        # 后面马上再 加 pool 就会死锁，所以将 两个 for 放同一个 pool 里面
        # with Pool(n_worker) as pool:
        #     for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         op_tmp_clip.append(img)
        #
        self.rgb_ds.cur_sub_vid_clip = torch.stack(rgb_tmp_clip)
        self.op_ds.cur_sub_vid_clip = torch.stack(op_tmp_clip)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}


class TwoStream_Test_DS_v3(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=16):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        self.cur_sub_video_name = sub_video_name
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list))

        # print("img_path_list: ", img_path_list)
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        # starmap 与 map 性能相同，但支持多参数的func，但是 所有参数都要整理为 同样shape的 iterable
        # with Pool(n_worker) as pool:
        #     for img in pool.starmap(_load_frame, rgb_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         rgb_tmp_clip.append(img)
        for args in rgb_zip_args:
            img_path, img_size, transform = args
            img = _load_frame(img_path, img_size, transform)
            rgb_tmp_clip.append(img)

        with Pool(n_worker) as pool:
            for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
                # 必须声明为 类方法才能直接传入 multiprocess
                op_tmp_clip.append(img)

        # for args in op_zip_args:
        #     img = _load_op(args)
        #     op_tmp_clip.append(img)
        #
        # 后面马上再 加 pool 就会死锁，所以将 两个 for 放同一个 pool 里面
        # with Pool(n_worker) as pool:
        #     for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         op_tmp_clip.append(img)
        #
        self.rgb_ds.cur_sub_vid_clip = torch.stack(rgb_tmp_clip)
        self.op_ds.cur_sub_vid_clip = torch.stack(op_tmp_clip)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}
#

# 查阅 python multiprocess 原理和api后，
# 我猜测是 with 的锅
# 测试之后，无效
class TwoStream_Test_DS_v4(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=16):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list))

        # print("img_path_list: ", img_path_list)
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        # starmap 与 map 性能相同，但支持多参数的func，但是 所有参数都要整理为 同样shape的 iterable
        pool = Pool(n_worker)
        for img in pool.starmap(_load_frame, rgb_zip_args): # zip_args 必须要自己整理
            # 必须声明为 类方法才能直接传入 multiprocess
            rgb_tmp_clip.append(img)
        for img in pool.starmap(_load_op, op_zip_args):  # zip_args 必须要自己整理
            # 必须声明为 类方法才能直接传入 multiprocess
            op_tmp_clip.append(img)
        pool.close()
        pool.join()
        #
        self.rgb_ds.cur_sub_vid_clip = torch.stack(rgb_tmp_clip)
        self.op_ds.cur_sub_vid_clip = torch.stack(op_tmp_clip)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}


# apply_async 改造下, ：吐槽
# (后面想换成 torch的 multiprocess)
class TwoStream_Test_DS_5(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=16):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list))

        # print("img_path_list: ", img_path_list)
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        # starmap 与 map 性能相同，但支持多参数的func，但是 所有参数都要整理为 同样shape的 iterable
        with Pool(n_worker) as pool:

            img_results = [ pool.apply_async(_load_frame, args=(rgb_img,
                    rgb_size, rgb_transform)) for rgb_img,
                    rgb_size, rgb_transform in rgb_zip_args]# zip_args 必须要自己整理
            op_results = [pool.apply_async(_load_op, args=(op_img,
                    op_size, op_transform)) for op_img,
                    op_size, op_transform in op_zip_args]  # zip_args 必须要自己整理
                # 必须声明为 类方法才能直接传入 multiprocess
        for img in img_results:
            rgb_tmp_clip.append(img.get())
        for img in op_results.get():
            op_tmp_clip.append(img.get())
        #
        # 后面马上再 加 pool 就会死锁，所以将 两个 for 放同一个 pool 里面
        # with Pool(n_worker) as pool:
        #     for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         op_tmp_clip.append(img)
        #
        self.rgb_ds.cur_sub_vid_clip = torch.stack(rgb_tmp_clip)
        self.op_ds.cur_sub_vid_clip = torch.stack(op_tmp_clip)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}
#

# 只用一个 pool, 但是每次 读 一个 pair
# (rgb,op), 读完再拆开
# 不懂为啥还是失败的？？？真的玄学啊？？？
# 我猜是 多进程内存释放有问题导致的
# TODO fix : 诡异的是，
# try-except 不会返回异常，还是一直在阻塞
class TwoStream_Test_DS_v6(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=16):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_flag_list = ["rgb" for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list, rgb_flag_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_flag_list = ["op" for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list, op_flag_list))
        #
        rgb_op_zip_args = []
        rgb_op_zip_args.extend(rgb_zip_args)
        rgb_op_zip_args.extend(op_zip_args)

        # print("img_path_list: ", img_path_list)
        # tmp_clip = []
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        # 诡异的是， try-except 不会返回异常，还是一直在阻塞
        with Pool(n_worker) as pool:
            try:
                import ipdb
                ipdb.set_trace()
                res = pool.starmap(_load_frame_op, rgb_op_zip_args)
            except Exception as ex:
                print(ex)
        tmp_clip = [img for img in res]
        self.rgb_ds.cur_sub_vid_clip = torch.stack(tmp_clip[:len(rgb_img_path_list)])
        self.op_ds.cur_sub_vid_clip = torch.stack(tmp_clip[len(rgb_img_path_list):])
        assert len(self.op_ds.cur_sub_vid_clip) == len(op_img_path_list), "tmp_clip len error!"
        assert self.rgb_ds.cur_sub_vid_clip.size()[1] == 3, "rgb_ds.cur_sub_vid_clip error!"
        assert self.op_ds.cur_sub_vid_clip.size()[1] == 2, "op_ds.cur_sub_vid_clip error!"

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}

# 多线程实现:成功了！！！
# 但是需要 先打包 (rgb, op) 并行得到结果 再解包
class TwoStream_Test_DS_v7(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=16):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_flag_list = ["rgb" for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list, rgb_flag_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_flag_list = ["op" for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list, op_flag_list))
        #
        rgb_op_zip_args = []
        rgb_op_zip_args.extend(rgb_zip_args)
        rgb_op_zip_args.extend(op_zip_args)

        # print("img_path_list: ", img_path_list)
        tmp_clip = []
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        args = rgb_op_zip_args
        with ThreadPoolExecutor(max_workers=n_worker) as pool:

            # obj_list = []
            # for page in range(1, 5):
            #     obj = t.submit(spider, page)
            #     obj_list.append(obj)
            #
            # for future in as_completed(obj_list):
            #     data = future.result()
            #     print(f"main: {data}")

            # for img in pool.map(lambda p: _load_frame_op(*p), args): # zip_args 必须要自己整理
            for img in pool.map(_load_frame_op, rgb_op_zip_args):
                tmp_clip.append(img)
        self.rgb_ds.cur_sub_vid_clip = torch.stack(tmp_clip[:len(rgb_img_path_list)])
        self.op_ds.cur_sub_vid_clip = torch.stack(tmp_clip[len(rgb_img_path_list):])
        assert len(self.op_ds.cur_sub_vid_clip) == len(op_img_path_list), "tmp_clip len error!"
        assert self.rgb_ds.cur_sub_vid_clip.size()[1] == 3, "rgb_ds.cur_sub_vid_clip error!"
        assert self.op_ds.cur_sub_vid_clip.size()[1] == 2, "op_ds.cur_sub_vid_clip error!"

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}

# 优化TwoStream_Test_DS_v7，尝试直接 一个 pool 运行两个 for (处理 rgb, op)
# 成功了！！！
# 但是 相比多进程很慢 ，我猜测原因:
# 1-我的任务不是纯粹的IO还有计算，2-多线程并发不如多进程并行
class TwoStream_Test_DS_v8(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=16):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list))

        # print("img_path_list: ", img_path_list)
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        with ThreadPoolExecutor(max_workers=n_worker) as pool:
            for img in pool.map(_load_frame, rgb_zip_args):
                rgb_tmp_clip.append(img)
            for img in pool.map(_load_op, op_zip_args):
                op_tmp_clip.append(img)

        self.rgb_ds.cur_sub_vid_clip = torch.stack(rgb_tmp_clip)
        self.op_ds.cur_sub_vid_clip = torch.stack(op_tmp_clip)
        # assert len(self.op_ds.cur_sub_vid_clip) == len(op_img_path_list), "tmp_clip len error!"
        # assert self.rgb_ds.cur_sub_vid_clip.size()[1] == 3, "rgb_ds.cur_sub_vid_clip error!"
        # assert self.op_ds.cur_sub_vid_clip.size()[1] == 2, "op_ds.cur_sub_vid_clip error!"

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}

# 换 ProcessPoolExecutor (卡死)
class TwoStream_Test_DS_v9(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=4):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list))

        # print("img_path_list: ", img_path_list)
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        # with Pool(n_worker) as pool:
        #     for img in pool.starmap(_load_frame, rgb_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         rgb_tmp_clip.append(img)
        #     for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         op_tmp_clip.append(img)
        with ProcessPoolExecutor(max_workers=n_worker) as executor:
                rgb_tmp = executor.map(_load_frame, rgb_zip_args)
        rgb_tmp_clip = [rgb_ for rgb_ in rgb_tmp]  # since rgb_tmp is generator, need to convert
        #
        # with ProcessPoolExecutor(max_workers=n_worker) as executor:
        #     op_tmp = executor.map(_load_op, op_zip_args)
        # op_tmp_clip = [op_ for op_ in op_tmp]
        op_tmp_clip = rgb_tmp_clip
        # print("rgb_tmp_clip: ", rgb_tmp_clip)
        # executor = ProcessPoolExecutor(max_workers=n_worker)
        # # map-------------------------------------
        # rgb_res = executor.map(_load_frame, rgb_zip_args)
        # rgb_tmp_clip = [rgb for ]
        # op_tmp_clip = executor.map(_load_op, op_zip_args)
        # for future in futures:
        # with ProcessPoolExecutor(max_workers=n_worker) as executor:
        #     tasks = [executor.submit(_load_frame, rgb) for rgb in rgb_zip_args]
        #     for f in as_completed(tasks):
        #         ret = f.done()
        #         if ret:
        #             print
        #             f.result().status_code
        #
        # 后面马上再 加 pool 就会死锁，所以将 两个 for 放同一个 pool 里面
        # with Pool(n_worker) as pool:
        #     for img in pool.starmap(_load_op, op_zip_args): # zip_args 必须要自己整理
        #         # 必须声明为 类方法才能直接传入 multiprocess
        #         op_tmp_clip.append(img)
        #
        self.rgb_ds.cur_sub_vid_clip = torch.stack(rgb_tmp_clip)
        self.op_ds.cur_sub_vid_clip = torch.stack(op_tmp_clip)

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}

# 换 ProcessPoolExecutor + 打包 (rgb,op) 处理， (卡死)
class TwoStream_Test_DS_v10(Dataset):

    def __init__(self, video_folder, data_type="rgb_op",
                 clip_length=(10,9), size=(256, 256), n_worker=4):
        self.rgb_ds = ds_base(video_folder=video_folder[0], data_type="rgb",
                                   clip_length=clip_length[0], n_worker=n_worker)
        self.op_ds = ds_base(video_folder=video_folder[1], data_type="op",
                                   clip_length=clip_length[1], n_worker=n_worker)
        self.n_worker = n_worker
        self.size = size
        self.transform = (
            transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            transforms.Compose([
                        # transforms.Resize(size), # 丢到 load 里面实现
                        transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                    ]),
        )

    def __len__(self):

        return self.cur_len

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        rgb_ds = self.rgb_ds
        self.cur_len = rgb_ds.videos[sub_video_name]['length'] - \
                       rgb_ds.clip_length+ 1
        op_ds = self.op_ds
        #
        rgb_img_path_list = rgb_ds.videos[sub_video_name]['frame']
        # print("rgb_img_path_list: ", rgb_img_path_list)
        op_img_path_list = op_ds.videos[sub_video_name]['frame']
        size = self.size
        #
        rgb_img_size_list = [size for idx in range(len(rgb_img_path_list))]
        rgb_transform_list = [self.transform[0] for idx in range(len(rgb_img_path_list))]
        rgb_flag_list = ["rgb" for idx in range(len(rgb_img_path_list))]
        rgb_zip_args = list(zip(rgb_img_path_list, rgb_img_size_list, rgb_transform_list, rgb_flag_list))
        #
        op_img_size_list = [size for idx in range(len(op_img_path_list))]
        op_transform_list = [self.transform[1] for idx in range(len(op_img_path_list))]
        op_flag_list = ["op" for idx in range(len(op_img_path_list))]
        op_zip_args = list(zip(op_img_path_list, op_img_size_list, op_transform_list, op_flag_list))
        #
        rgb_op_zip_args = []
        rgb_op_zip_args.extend(rgb_zip_args)
        rgb_op_zip_args.extend(op_zip_args)

        # print("img_path_list: ", img_path_list)
        # tmp_clip = []
        rgb_tmp_clip = []
        op_tmp_clip = []
        n_worker = self.n_worker

        with ProcessPoolExecutor(max_workers=n_worker) as executor:
                tmp = executor.map(_load_frame_op, rgb_op_zip_args)
        tmp_clip = [img for img in tmp]

        self.rgb_ds.cur_sub_vid_clip = torch.stack(tmp_clip[:len(rgb_img_path_list)])
        self.op_ds.cur_sub_vid_clip = torch.stack(tmp_clip[len(rgb_img_path_list):])
        assert len(self.op_ds.cur_sub_vid_clip) == len(op_img_path_list), "tmp_clip len error!"
        assert self.rgb_ds.cur_sub_vid_clip.size()[1] == 3, "rgb_ds.cur_sub_vid_clip error!"
        assert self.op_ds.cur_sub_vid_clip.size()[1] == 2, "op_ds.cur_sub_vid_clip error!"

        del tmp_clip
        gc.collect()

    def __getitem__(self, index):
        rgb_clip_tensor = self.rgb_ds.__getitem__(index)
        op_clip_tensor = self.op_ds.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}


# ----------------------------------------- #

# 再试一种简单方案
# (1) naive + 外部for 确定 sub_video_name
class clip_Test_DS_naive(Dataset):

    def __init__(self, video_folder, data_type, clip_length=5, size=(256, 256), n_worker = 16):
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = video_folder
        self.videos = OrderedDict()
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10
        self.data_type = data_type
        self.n_worker = n_worker
        self._setup()
        transform_map = {
            "rgb": transforms.Compose([
                    # transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    # transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)], (0,1)
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size), # 丢到 load 里面实现
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False, (0,1)
                ]),
        }
        self.transform = transform_map[data_type]

    def _setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        # frames下面的若干个子目录的path：path/to/01, 02...
        for video in sorted(videos):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos[sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            # {'path':all_sub_video_path(e.g. [/path/to/01, /path/to/02 ...]),
            #  'frame':all_frame_of_sub_video_path(e.g. [/path/to/01/00001.jpg, /path/to/01/00002.jpg ...]),
            # 'length':len_of_sub_video(e.g. len(01_frames), len(02_frames), ...),即子目录下帧的数目}
            # 并且将 UCSD/ped2 and ped1 的training set frame的channel都转为3了
            # 注：SIST的liu wen已经将 ped1/ped2都从tif转为jpg,并且对数据集做了处理，所以jpg格式统一一切
            self.videos[sub_video_name]['path'] = video
            self.videos[sub_video_name]['frame'] = glob.glob(os.path.join(video, '*'))
            self.videos[sub_video_name]['frame'].sort()  # 一定要排序，确保order正确，在glob前面加个sorted ?
            self.videos[sub_video_name]['length'] = \
                len(self.videos[sub_video_name]['frame'])

    # 可以理解为自定义的高级版 __getitem__()
    def get_clip(self, sub_video_name, start, end):
        fn_map = {
            "rgb": self._load_frames,
            "op": self._load_ops,
        }
        fn = fn_map[self.data_type]
        img_list = self.videos[sub_video_name]["frame"][start:end]
        img_size = self.image_height, self.image_width
        transform = self.transform

        return fn(img_list, img_size, transform)

    def _load_frames(self, img_list, img_size, transform):
        image_width, image_height = img_size
        all_clip = []

        for img_path in img_list:       # [start,end)
            # print("img_path: ", img_path)
            img = img_decode(img_path) # 代替 cv2.imread
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_width, image_height)) # 注意要 (w,h)
            # perform transform, TODO
            if transform:
                # img = Image.fromarray(img)
                img = transform(img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  # 合并为更高一维, 把所有的frame都合并到一个大 tensor中

        return test_clip

    def _load_ops(self, img_list, img_size, transform):
        image_width, image_height = img_size
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (image_width, image_height))
            # 注意：cv2的 resize的 input 是 w,h 而不是一般的 h,w
            # print(img.shape)
            if transform:
                img = transform(img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中

        return test_clip
    # --------------------------------------------------------#
# (2) 外部test(). 确定 sub_video_name

# ============================================================ #
# 最终发现，还是这个方案 +  dataloader, 直接 follow trainining
# 的  data loading 是最快的 速度加载
# 所以 train 和 test 使用同一种 load 就是最佳实践

# ============================================================ #
#
# unit test for TwoStream_Test_DS
class test_TwoStream_Test_DS():
    # 测试 子目录
    def test_1(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            dataset.test(sub_video_name)
            print(dataset.cur_sub_video_name)

    # 测试 一些数目 是否正确 (for len(dataset)方式)
    def test_2(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            print(sub_video_name)
            dataset.test(sub_video_name)
            print(len(dataset),'--', dataset.clip_length, '--', dataset.cur_len)
            # for idx, sample in enumerate(dataset): # enumerate 似乎有问题?TODO
            for idx in range(len(dataset)):
                sample = dataset[idx]
                print(idx) # 1429 in avenue-01, since 1438-9+1 = 1430
                print(sample["rgb"].shape, '--', sample["op"].shape)
            #
            # 测试 iter_num (没有batch,等于frame_num; 有batch, 等于 ceil(frame_num/batch_size))
            assert len(dataset) == idx + 1, "len(dataset) != idx+1"
            # 测试 rgb and op num
            frame_num = dataset.videos["rgb"][sub_video_name]['length']
            op_num = dataset.videos["op"][sub_video_name]['length']
            assert frame_num == op_num + 1, "frame_num != op_num + 1"
            # 测试 clip_num
            c1 = frame_num - dataset.clip_length[0] + 1
            c2 = op_num - dataset.clip_length[1] + 1
            assert c1 == c2 and c1 == len(dataset), "error, c1==c2 and c1==len(dataset)"
            #
            break  # 只处理第一个sub_video 即可
        print("exit succ")

    # 测试 一些数目 是否正确 (dataloader方式)
    def test_3(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            print(sub_video_name)
            dataset.test(sub_video_name)
            print(len(dataset),'--', dataset.clip_length, '--', dataset.cur_len)
            batch_size = 1
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            for idx, sample in enumerate(loader): # enumerate 似乎有问题?TODO
            # for idx in range(len(dataset)):
                # sample = dataset[idx]
                print(idx) # 1429 in avenue-01, since 1438-9+1 = 1430
                print(sample["rgb"].shape, '--', sample["op"].shape)
            #
            # 测试 iter_num (没有batch,等于frame_num; 有batch, 等于 ceil(frame_num/batch_size))
            assert len(dataset) == idx + 1, "len(dataset) != idx+1"
            # 测试 rgb and op num
            frame_num = dataset.videos["rgb"][sub_video_name]['length']
            op_num = dataset.videos["op"][sub_video_name]['length']
            assert frame_num == op_num + 1, "frame_num != op_num + 1"
            # 测试 clip_num
            c1 = frame_num - dataset.clip_length[0] + 1
            c2 = op_num - dataset.clip_length[1] + 1
            assert c1 == c2 and c1 == len(dataset), "error, c1==c2 and c1==len(dataset)"
            #
            break  # 只处理第一个sub_video 即可
        print("exit succ")

    # 测试 单个 sample & vis (vis 非常完美，这套code可以继承下去)
    def test_4(self, dataset, all_sub_video_name_list,writer):

        vis_info = "vis_test_4"
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            dataset.test(sub_video_name)
            #
            if video_id == 0:
                batch_size = 1
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                for idx, sample in enumerate(loader):
                    rgb = sample["rgb"][0] # 因为 它是 (b,t,c,h,w)
                    op = sample["op"][0] # 默认情况下 batch 是分别施加到 rgb and op
                    if idx==0: # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape) # rgb, [10,3,256,256]
                        print(op.shape) # op, [9,2,256,256]
                        print(rgb.min(), rgb.max()) # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb": rgb, "op": op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)

                    if idx==len(dataset)-1: # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape)  # rgb, [10,3,256,256]
                        print(op.shape)  # op, [9,2,256,256]
                        print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb": rgb, "op": op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)

            if video_id == len(all_sub_video_name_list)-1:
                batch_size = 1
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                for idx, sample in enumerate(loader):
                    rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
                    op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
                    if idx == 0:  # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape)  # rgb, [10,3,256,256]
                        print(op.shape)  # op, [9,2,256,256]
                        print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb":rgb, "op":op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)

                    if idx == len(dataset) - 1:  # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape)  # rgb, [10,3,256,256]
                        print(op.shape)  # op, [9,2,256,256]
                        print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb": rgb, "op": op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)


# unit test for TwoStream_Train_DS
class test_TwoStream_Train_DS():
    # 测试 子目录/
    def test_1(self, dataset):
        batch_size = 1
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print(rgb.shape)  # rgb, [10,3,256,256]
            print(op.shape)  # op, [9,2,256,256]
            print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
            print(op.min(), op.max())  # 测 op value
            #
            print(len(dataset)) # num_all_clip
            print(dataset._get_clips_num_list(), sum(dataset._get_clips_num_list()))
            break

    # 测试 iter_num (ceil(num_all_clip/batch_size))
    def test_2(self, dataset):
        batch_size = 32
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print(rgb.shape)  # rgb, [10,3,256,256]
            print(op.shape)  # op, [9,2,256,256]
            print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
            print(op.min(), op.max())  # 测 op value
            #
            print(len(dataset)) # num_all_clip
            print(dataset._get_clips_num_list(), sum(dataset._get_clips_num_list()))
            print("iter: ",idx)
        assert idx + 1 == torch.ceil(len(dataset) / batch_size)

    # 测试 单个 sample & vis (vis 非常完美，这套code可以继承下去)
    def test_3(self, dataset, writer):

        vis_info = "vis_train_test_4"
        batch_size = 32
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print("iter: ", idx)
            if idx==0: # 第一个和最后一个clip
                # vis to compare with gt
                sample = {"rgb": rgb, "op": op}
                vis_load(sample, writer, vis_info, idx)

            if idx==len(loader)-1: # 第一个和最后一个clip
                sample = {"rgb": rgb, "op": op}
                vis_load(sample, writer, vis_info, idx)

        assert idx + 1 == torch.ceil(len(dataset) / batch_size)

    def test_shanghaitech_bug(self,dataset):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print("iter: ", idx)

# ---------------------------------------------------------- #
# unit test for clip_Test_DS


# unit test for clip_Train_DS


# ---------------------------------------------------------- #

# train
def vis_load(sample, writer, vis_info, idx): # (t,c,h,w)
    seq_len = sample["rgb"].size()[0]
    grid_rgb = get_vis_tensor(sample["rgb"], "rgb", seq_len)
    writer.add_image(vis_info+"/rgb_iter=_{}".format(idx), grid_rgb)
    seq_len = sample["op"].size()[0]
    grid_op = get_vis_tensor(sample["op"], "op", seq_len)
    writer.add_image(vis_info + "/op_iter=_{}".format(idx), grid_op)

# test
def vis_load_gt(dataset, sub_video_name, idx,
                sample, writer, vis_info):
    #
    rgb_list = dataset.videos["rgb"][sub_video_name]['frame'][idx:
    idx + dataset.clip_length[0]]
    op_list = dataset.videos["op"][sub_video_name]['frame'][idx:
    idx + dataset.clip_length[1]]
    rgb, op = dataset._load_frames(rgb_list), dataset._load_ops(op_list)
    # transform
    # rgb, op = transform["rgb"](rgb), transform["op"](op)
    rgb_load_gt = torch.cat([sample["rgb"], rgb], 0)
    op_load_gt = torch.cat([sample["op"], op], 0)
    #
    seq_len = sample["rgb"].size()[0]
    grid_rgb = get_vis_tensor(rgb_load_gt, "rgb", seq_len)
    writer.add_image(vis_info+"/rgb_{}_{}".format(sub_video_name, idx), grid_rgb)
    seq_len = sample["op"].size()[0]
    grid_op = get_vis_tensor(op_load_gt, "op", seq_len)
    writer.add_image(vis_info + "/op_{}_{}".format(sub_video_name, idx), grid_op)

def get_vis_tensor(vis_tensor, dataset_type, nrow):
    if dataset_type == "rgb": # or dataset_type == "optical_flow":
        grid = make_grid(vis_tensor, nrow=nrow, normalize=True, range=(-1, 1))  # normalize, (-1,1) -> (0,1)
    elif dataset_type == "op":
        flow_batch = vis_tensor.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()  # [b, h, w, 2]
        flow_vis_batch = batch_flow_to_image(flow_batch)  # [b, h, w, 3]
        tensor = torch.from_numpy(flow_vis_batch)  # [b, h, w, c]
        tensor = tensor.permute(0, 3, 1, 2)  # [b, c, h, w]
        grid = make_grid(tensor, nrow=nrow)  # (0,1), 无需 normalize
    else:
        grid = None
        print("dataset_type error ! ")
        exit()
    return grid
# =========================================================== #
if __name__ == '__main__':
    # test TwoStream_Test_DS
    def test_test_TwoStream_Test_DS():
        #
        sum_path = utils.get_dir("/p300/test_TwoStream_Test_DS")
        writer = SummaryWriter(log_dir=sum_path)
        dataset_root = "/p300/dataset"  # universial, in p300
        dataset_name = "avenue"  # 其实应该用 toy dataset 来做 unit test
        path_rgb = os.path.join(dataset_root, "{}/testing/frames".format(dataset_name))  #
        path_optical_flow = os.path.join(dataset_root, "{}/optical_flow/testing/frames/flow".format(dataset_name))  #
        video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
        dataset = TwoStream_Test_DS(video_folder)
        all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
        # print(all_sub_video_name_list)
        #
        t_test_TwoStream_Test_DS = test_TwoStream_Test_DS()
        # test_1
        # t_test_TwoStream_Test_DS.test_1(dataset, all_sub_video_name_list)
        # test_2
        # t_test_TwoStream_Test_DS.test_2(dataset, all_sub_video_name_list)
        # test_3
        # t_test_TwoStream_Test_DS.test_3(dataset, all_sub_video_name_list)
        #
        # test_4 : 真正的 unittest, 使用各种 assert传入 gt, 不报错即为 通过
        #
        t_test_TwoStream_Test_DS.test_4(dataset, all_sub_video_name_list, writer)
    # test_test_TwoStream_Test_DS()

    # test TwoStream_Train_DS
    def test_test_TwoStream_Train_DS():
        #
        sum_path = utils.get_dir("/p300/test_TwoStream_Train_DS")
        writer = SummaryWriter(log_dir=sum_path)
        dataset_root = "/p300/dataset"  # universial, in p300
        dataset_name = "avenue"  # 其实应该用 toy dataset 来做 unit test
        path_rgb = os.path.join(dataset_root, "{}/training/frames".format(dataset_name))  #
        path_optical_flow = os.path.join(dataset_root, "{}/optical_flow/training/frames/flow".format(dataset_name))  #
        # print(path_rgb)
        # print(path_optical_flow)
        video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
        dataset = TwoStream_Train_DS(video_folder)
        all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
        # print(all_sub_video_name_list)
        #
        t_test_TwoStream_Train_DS = test_TwoStream_Train_DS()
        # test_1
        # t_test_TwoStream_Train_DS.test_1(dataset)
        # test_2
        # t_test_TwoStream_Train_DS.test_2(dataset)
        #
        # test_3: 真正的 unittest, 使用各种 assert传入 gt, 不报错即为 通过
        #
        t_test_TwoStream_Train_DS.test_3(dataset, writer)
    # test_test_TwoStream_Train_DS()

    def test_test_TwoStream_Train_DS_shanghaitech():
        #
        dataset_root = "/p300/dataset"  # universial, in p300
        dataset_name = "shanghaitech"  # 其实应该用 toy dataset 来做 unit test
        path_rgb = os.path.join(dataset_root, "{}/training/frames".format(dataset_name))  #
        path_optical_flow = os.path.join(dataset_root, "{}/optical_flow/training/frames/flow".format(dataset_name))  #
        # print(path_rgb)
        # print(path_optical_flow)
        video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
        dataset = TwoStream_Train_DS(video_folder)
        all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
        # print(all_sub_video_name_list)
        #
        t_test_TwoStream_Train_DS = test_TwoStream_Train_DS()
        # test_1
        # t_test_TwoStream_Train_DS.test_1(dataset)
        # test_2
        # t_test_TwoStream_Train_DS.test_2(dataset)
        #
        # test_3: 真正的 unittest, 使用各种 assert传入 gt, 不报错即为 通过
        #
        t_test_TwoStream_Train_DS.test_shanghaitech_bug(dataset)
    test_test_TwoStream_Train_DS_shanghaitech()

