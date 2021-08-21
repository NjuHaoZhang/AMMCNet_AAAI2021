import os,time,math
# import pickle
import _pickle as pickle # cpickle 加速 (py3中cpickle更名为 _pickle)
from collections import namedtuple
from io import BytesIO
import multiprocessing
#
import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tv_fn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
#
from ..utils.flowlib import readFlow, flow_to_image, batch_flow_to_image
from ..utils.utils import get_dir


rng = np.random.RandomState(2017)

class LMDBDataset_clip_base(Dataset):

    def __init__(self, path, data_type, clip_length=9, num_width=6,
                 transform=None):
        # print("base init")
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.num_sub_video = int(txn.get('num_sub_video'.encode('utf-8')).decode('utf-8'))
            self.list_len_each_sub_video = pickle.loads(
                txn.get('list_len_each_sub_video'.encode('utf-8'))) # pickle is byte, no need to decode

        #
        self.clip_length = clip_length
        self.num_width = num_width
        self.data_type = data_type
        self.transform = transform # online transform

    def __len__(self):
        pass

    def __getitem__(self, index):
        sub_vid, cur_cid = self.load_addr(index)

        return self.load_sample(sub_vid, cur_cid)

    def load_addr(self, index):
         return index, index

    def load_sample(self, sub_vid, cur_cid):

        key_list = [f'{sub_vid}-{str(cur_cid+i).zfill(self.num_width)}'.encode('utf-8')
                     for i in range(self.clip_length)]

        with self.env.begin(write=False) as txn:
         tmp_clip = []
         for key in key_list:
             sample = txn.get(key)
             sample_tensor = self.get_transform(self.data_type, sample, self.transform)
             tmp_clip.append(sample_tensor)

        return torch.stack(tmp_clip)

    def get_transform(self, data_type, sample, transform=None):
        if data_type == "rgb":
            img = self.rgb_transform_v3(sample, transform)
        elif data_type == "op":
            img = self.op_transform_v2(sample, transform)
        else:
            img = None
            print("get_sample error")
            exit()
        #
        if transform:
            img = transform(img)
        return img

    def rgb_transform_v1(self, sample, transform=None):
        img = pickle.loads(sample)
        img = torch.from_numpy(img).float()  # (h,w,3)
        img = img.permute(2, 0, 1)

        return img

    def rgb_transform_v2(self, sample, transform=None):
        buffer = BytesIO(sample)
        img = Image.open(buffer)
        img = tv_fn.ToTensor()(img)
        img = tv_fn.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img

    def rgb_transform_v3(self, sample, transform=None):
        buffer = BytesIO(sample)
        data = np.load(buffer)
        img = data['rgb']
        img = torch.from_numpy(img)  # (h,w,2)
        img = img.permute(2, 0, 1)

        return img

    def op_transform_v1(self, sample, transform=None):
        img = pickle.loads(sample)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img

    def op_transform_v2(self, sample, transform=None):
        buffer = BytesIO(sample)
        data = np.load(buffer)
        img = data['flo']
        img = torch.from_numpy(img)  # (h,w,2)
        img = img.permute(2, 0, 1)

        return img


#
class LMDBDataset_clip_train(LMDBDataset_clip_base):

    def __len__(self):
        #
        list_num_clip = [len_cur_sub_video - self.clip_length + 1
         for len_cur_sub_video in self.list_len_each_sub_video]
        # assert num_clip == sum(list_num_clip), \
        #     "LMDBDataset_op_train __len__ error, {}!={}".format(
        #         num_clip,sum(list_num_clip))

        return sum(list_num_clip)

    def load_addr(self, index):
        sub_vid = rng.randint(0, self.num_sub_video)
        cur_cid = rng.randint(0, self.list_len_each_sub_video[sub_vid] - self.clip_length)
        #
        return sub_vid, cur_cid


class LMDBDataset_clip_test_v1(LMDBDataset_clip_base):

    def __len__(self):
        num_clip = self.list_len_each_sub_video[self.cur_sub_vid] - self.clip_length + 1

        return num_clip

    def test(self, sub_vid):
        self.cur_sub_vid = sub_vid

    def load_addr(self, index):
        sub_vid = self.cur_sub_vid
        cur_cid = index
        #
        return sub_vid, cur_cid

#
class LMDBDataset_clip_test(LMDBDataset_clip_base):

    
    def __len__(self):
        num_clip = self.list_len_each_sub_video[self.cur_sub_vid] - self.clip_length + 1

        return num_clip # clip_num

    def test(self, sub_vid):
        self.cur_sub_vid = sub_vid
        #
        cur_cid = 0  # [0, len_sub_video)
        len_sub_video = self.list_len_each_sub_video[self.cur_sub_vid] # num of frames in sub_vid
        key_list = [f'{sub_vid}-{str(cur_cid+i).zfill(self.num_width)}'.encode('utf-8')
                    for i in range(len_sub_video)]
        tmp_clip = []
        with self.env.begin(write=False) as txn:
            for key in key_list:
                sample = txn.get(key)
                sample_tensor = self.get_transform(self.data_type, sample, self.transform)
                tmp_clip.append(sample_tensor)

        self.cur_sub_vid_clip = torch.stack(tmp_clip) # (t,c,h,w)

    def __getitem__(self, index):
        return self.cur_sub_vid_clip[index:index+self.clip_length,:,:,:] # 从内存中取一个clip


class LMDBDataset_twostream_train(Dataset):
    def __init__(self, path, data_type="rgb_op", clip_length=(10,9),
                 num_width=(6,6), transform=(None,None)):

        self.lmdb_rgb = LMDBDataset_clip_train(path=path[0],
               data_type="rgb", clip_length=clip_length[0],
               num_width=num_width[0], transform=transform[0])
        self.lmdb_op = LMDBDataset_clip_train(path=path[1],
               data_type="op", clip_length=clip_length[1],
               num_width=num_width[1], transform=transform[1])

    def __len__(self):
        assert len(self.lmdb_rgb)==len(self.lmdb_op), "LMDBDataset_twostream_train error"
        return len(self.lmdb_rgb)

    def __getitem__(self, index):
        rgb_clip_tensor = self.lmdb_rgb.__getitem__(index)
        op_clip_tensor = self.lmdb_op.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op":op_clip_tensor}


class LMDBDataset_twostream_test(Dataset):
    def __init__(self, path, data_type="rgb_op", clip_length=(10, 9),
                 num_width=(6, 6), transform=(None, None)):
        self.lmdb_rgb = LMDBDataset_clip_test(path=path[0],
               data_type="rgb", clip_length=clip_length[0],
               num_width=num_width[0], transform=transform[0])
        self.lmdb_op = LMDBDataset_clip_test(path=path[1],
                data_type="op", clip_length=clip_length[1],
                num_width=num_width[1], transform=transform[1])

    def __len__(self):
        assert len(self.lmdb_rgb) == len(self.lmdb_op), "LMDBDataset_twostream_train error"
        return len(self.lmdb_rgb)

    def test(self, sub_vid):  # 外部调用,设置sub_vid
        self.lmdb_rgb.test(sub_vid)
        self.lmdb_op.test(sub_vid)

    def __getitem__(self, index):
        rgb_clip_tensor = self.lmdb_rgb.__getitem__(index)
        op_clip_tensor = self.lmdb_op.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}


# ==================================================== #
def vis_gen(sample, data_type, writer, vis_info, idx): # (t,c,h,w)
    seq_len = sample.size()[0]
    if data_type == "rgb":
        grid = get_vis_tensor(sample, "rgb", seq_len)
    if data_type == "op":
        grid = get_vis_tensor(sample, "op", seq_len)
    writer.add_image(vis_info+"/rgb_iter=_{}".format(idx), grid)

# test (no banfa)
def vis_gt_gen(dataset, sub_video_name, idx,
                sample, writer, vis_info):
    # gt

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
        grid = make_grid(vis_tensor, nrow=nrow, normalize=True, range=(-1,1))  # normalize, (-1,1) -> (0,1)
    elif dataset_type == "op":
        flow_batch = vis_tensor.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()  # [b, h, w, 2]
        flow_vis_batch = batch_flow_to_image(flow_batch)  # [b, h, w, 3]
        tensor = torch.from_numpy(flow_vis_batch)  # [b, h, w, c]
        tensor = tensor.permute(0, 3, 1, 2)  # [b, c, h, w]
        grid = make_grid(tensor, nrow=nrow)
    else:
        grid = None
        print("dataset_type error ! ")
        exit()
    return grid

# ============unit test ===================================#
# LMDBDataset_clip_train
class test_LMDBDataset_clip_train():
    # 测试 子目录/
    def test_1(self, dataset):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            #
            print(sample.size())  # op, [9,2,256,256]
            print(sample.min(), sample.max())  # 测 rgb value:[-1,1]
            #
            print(len(dataset)) # num_all_clip
            print(dataset.num_sub_video)
            print(dataset.list_len_each_sub_video)
            print("# ========================== #")
            break

    def test_2(self, dataset, data_type, writer):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        #
        vis_info = "vis_train_test_2_{}".format(data_type)
        for idx, sample in enumerate(loader):
            if idx==0:
                # vis to compare with gt

                vis_gen(sample[0], data_type, writer, vis_info, idx)

            # if idx==len(loader)-1:
            #     vis_gen(sample[0], "rgb", writer, vis_info, idx)
            break

    # iter_num and speed (ceil(num_all_clip/batch_size))
    def test_3(self, dataset):
        #
        batch_size = 16
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        #
        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        for idx, sample in enumerate(loader):
            #
            if idx == 0:
                print("sample.size(): ", sample.size())
            #
            cur_time = time.time()
            delta_time = cur_time - pre_time
            log_comm = 'batch_size={}  |  cost_time={}  |  fps={}'.format(
                batch_size, int(delta_time), int(delta_time / batch_size))
            # logging.info(log_comm)
            print(log_comm)
            #
            cnt_batch += batch_size
            cnt_time += delta_time # 累计每一个batch消耗时间
            #
            pre_time = cur_time
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        #
        assert idx + 1 == math.ceil(len(dataset) / batch_size)


# lmdb_1: rgb_v2 + op_v1
def test_train_v1():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad"
    dataset_name = "avenue"
    data_type = "rgb" # rgb
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v1")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)


def test_train_v2():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad"
    dataset_name = "avenue"
    data_type = "op" # op
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v2")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)

# --------- #

# lmdb_2: rgb_v1 + op_v2
def test_train_v3():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_2" # lmdb_vad_2
    dataset_name = "avenue"
    data_type = "rgb"
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v3")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)
    #

def test_train_v4():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_2" # lmdb_vad_2
    dataset_name = "avenue"
    data_type = "op"
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v4")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)
    #

# ---------- #

def test_train_v5():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_final" # lmdb_vad_2
    dataset_name = "avenue"
    data_type = "rgb"
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v5")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)
    #

def test_train_v6():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_final" # lmdb_vad_2
    dataset_name = "avenue"
    data_type = "op"
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 9
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v6")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)
    #

def test_train_v7():
    trainer = test_LMDBDataset_clip_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_final_v2" # lmdb_vad_final_v2
    dataset_name = "avenue"
    data_type = "rgb"
    mode = "training"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_train_v7")
    writer = SummaryWriter(log_dir=sum_path)
    #
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)
    #
# ==================================================== #

# LMDBDataset_clip_test
class test_LMDBDataset_clip_test():
    def test_1(self, dataset):
        for cur_sub_vid in range(dataset.num_sub_video):
            dataset.test(cur_sub_vid) # 设置 __len__
            for idx in range(len(dataset)):
                #
                sample = dataset[idx]
                print(sample.size())  # op, [9,2,256,256]
                print(sample.min(), sample.max())  # 测 rgb value:[-1,1]
                #
                print(len(dataset)) # num_all_clip
                print(dataset.num_sub_video)
                print(dataset.list_len_each_sub_video)
                print("# ========================== #")
                break
            break

    def test_2(self, dataset, data_type, writer):
        vis_info = "vis_test_test_2_{}".format(data_type)
        #
        for cur_sub_vid in range(dataset.num_sub_video):
            dataset.test(cur_sub_vid) # 设置 __len__
            for idx in range(len(dataset)):
                #
                sample = dataset[idx]
                if idx==0:
                    # vis to compare with gt
                    vis_gen(sample, data_type, writer, vis_info, idx)
                break

                # if idx==len(loader)-1: # 第一个和最后一个clip
                #     vis_gen(sample[0], "rgb", writer, vis_info, idx)
            break


    # 测试 iter_num and speed (ceil(num_all_clip/batch_size))
    def test_3(self, dataset):
        pre_time = time.time()
        cnt_batch, cnt_time = 0, 0
        for cur_sub_vid in range(dataset.num_sub_video):
            dataset.test(cur_sub_vid) # 设置 __len__
            for idx in range(len(dataset)):
                #
                sample = dataset[idx]
                if idx == 0:
                    print("sample.size(): ", sample.size())
                #
                cur_time = time.time()
                delta_time = cur_time - pre_time
                log_comm = 'batch_size=1  |  cost_time={}  |  fps={}'.format(
                int(delta_time), int(delta_time))
                # logging.info(log_comm)
                print(log_comm)
                #
                cnt_batch += 1
                cnt_time += delta_time # 累计每一个batch消耗时间
                #
                pre_time = cur_time
            log_comm = '# ------------------------------- #' \
                       'batch_total={}  |  cost_time={}  |  fps={}'.format(
                cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
            # logging.info(log_comm)
            print(log_comm)
            #
            # assert idx + 1 == math.ceil(len(dataset) / batch_size)

def test_test_v1():
    trainer = test_LMDBDataset_clip_test()
    #
    lmdb_out = "/p300/dataset/lmdb_vad"
    dataset_name = "avenue"
    data_type = "rgb" # rgb
    mode = "testing"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_test(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_test_v1")
    writer = SummaryWriter(log_dir=sum_path)
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)

def test_test_v2():
    trainer = test_LMDBDataset_clip_test()
    #
    lmdb_out = "/p300/dataset/lmdb_vad"
    dataset_name = "avenue"
    data_type = "op" # op
    mode = "testing"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_test(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_test_v2")
    writer = SummaryWriter(log_dir=sum_path)
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)

def test_test_v3():
    trainer = test_LMDBDataset_clip_test()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_2" #
    dataset_name = "avenue"
    data_type = "rgb"
    mode = "testing"
    path = os.path.join(lmdb_out, dataset_name, data_type, mode)
    clip_length = 10
    ds = LMDBDataset_clip_test(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_test_v3")
    writer = SummaryWriter(log_dir=sum_path)
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)

# ====================================================== #

# LMDBDataset_twostream_train
class test_LMDBDataset_twostream_train():
    def test_1(self, dataset):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            #
            rgb, op = sample["rgb"], sample["op"]

            print(rgb.size())  # op, [9,2,256,256]
            print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
            print(op.size())  # op, [9,2,256,256]
            print(op.min(), op.max())  # 测 rgb value:[-1,1]
            #
            print(len(dataset)) # num_all_clip
            # print(dataset.num_sub_video)
            # print(dataset.list_len_each_sub_video)
            print("# ========================== #")
            break

    def test_2(self, dataset, data_type, writer):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        #
        for idx, sample in enumerate(loader):
            if idx==0: # 第一个和最后一个clip
                # vis to compare with gt
                rgb, op = sample["rgb"], sample["op"]
                vis_info = "vis_train_test_2_rgb"
                vis_gen(rgb[0], "rgb", writer, vis_info, idx)
                vis_info = "vis_train_test_2_op"
                vis_gen(op[0], "op", writer, vis_info, idx)

            # if idx==len(loader)-1: # 第一个和最后一个clip
            #     vis_gen(sample[0], "rgb", writer, vis_info, idx)
            break

    # 测试 iter_num and speed (ceil(num_all_clip/batch_size))
    def test_3(self, dataset):
        batch_size = 16
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        #
        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        for idx, sample in enumerate(loader):
            #
            rgb, op = sample["rgb"], sample["op"]
            if idx == 0:
                print("rgb.size(): ", rgb.size())
                print("op.size(): ", op.size())
            #
            cur_time = time.time()
            delta_time = cur_time - pre_time
            log_comm = 'batch_size={}  |  cost_time={}  |  fps={}'.format(
                batch_size, int(delta_time), int(delta_time / batch_size))
            # logging.info(log_comm)
            print(log_comm)
            #
            cnt_batch += batch_size
            cnt_time += delta_time # 累计每一个batch消耗时间
            #
            pre_time = cur_time
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        #
        # assert idx + 1 == math.ceil(len(dataset) / batch_size)

def test_twostream_train_v1():
    trainer = test_LMDBDataset_twostream_train()
    #
    lmdb_out = "/p300/dataset/lmdb_vad_final"
    dataset_name = "avenue"
    mode = "training"
    rgb_path = os.path.join(lmdb_out, dataset_name, "rgb", mode)
    op_path = os.path.join(lmdb_out, dataset_name, "op", mode)
    path = (rgb_path, op_path)
    data_type = ("rgb", "op")  #
    clip_length = (10,9)
    ds = LMDBDataset_twostream_train(path, data_type, clip_length)
    #
    sum_path = get_dir("/p300/test_dir/test_twostream_train_v1")
    writer = SummaryWriter(log_dir=sum_path)
    trainer.test_1(ds)
    trainer.test_2(ds, data_type, writer)
    trainer.test_3(ds)


# ====================================================== #


# LMDBDataset_twostream_test
class test_LMDBDataset_twostream_test():
    def test_1(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            dataset.test(sub_video_name)
            print(dataset.cur_sub_video_name)

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
            assert len(dataset) == idx + 1, "len(dataset) != idx+1"
            frame_num = dataset.videos["rgb"][sub_video_name]['length']
            op_num = dataset.videos["op"][sub_video_name]['length']
            assert frame_num == op_num + 1, "frame_num != op_num + 1"
            c1 = frame_num - dataset.clip_length[0] + 1
            c2 = op_num - dataset.clip_length[1] + 1
            assert c1 == c2 and c1 == len(dataset), "error, c1==c2 and c1==len(dataset)"
            #
            break  #
        print("exit succ")

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


# ===================================================== #
if __name__ == '__main__':


    # ============================================================= #

    # test_test_v1() # rgb_v1: , op_v1:
    # test_test_v2()  #
    # test_test_v3()  #

    # ============================================================ #
    test_twostream_train_v1()

    # ============================================================ #

'''
python -m pyt_vad_topk_mem_cons.dataset.lmdb_dataset
'''

