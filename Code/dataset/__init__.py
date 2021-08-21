import pickle
import json
import os,time
#
from torch.utils.data import DataLoader
from .lmdb_dataset import (
    LMDBDataset_clip_train,
    LMDBDataset_clip_test,
    LMDBDataset_twostream_train,
    LMDBDataset_twostream_test,
)
#
from .two_stream_dataset import (
    TwoStream_Train_DS,
    clip_Train_DS,
    clip_Test_DS,
    clip_Train_DS_debug,
)
from .two_stream_dataset import clip_Test_DS_naive

from .two_stream_dataset import test_dataset
#
from ..utils.utils import load_params, save_params


class Config(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        # if not name.isupper():
        #     raise self.ConstCaseError('const name {} is not all uppercase'.format(name))

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            print(name, value)
            _str += '\t{}\t{}\n'.format(name, value)

        return _str


# =============================================================================== #

def get_dataset(const):
    #
    mode = const.mode
    exp_tag = const.exp_tag
    # ds_params_map = const.ds_params_map

    if mode == "training":
        # for init dataset
        video_folder = const.video_folder
        clip_length = const.clip_length
        data_type = const.data_type
        which_ds = const.which_ds  # normal, lmdb
        # save for testing
        params = Config()
        params.data_dir = const.data_dir
        params.dataset_name = const.dataset_name
        params.data_type = data_type
        params.clip_length = clip_length
        params.which_ds = which_ds  # normal, lmdb
        ds_params_pickle_save = const.ds_params_pickle_save # save params as pickle
        # save_params(params, ds_params_pickle_save, exp_tag, ds_params_map)
    elif mode == "testing":
        video_folder = const.video_folder #
        data_type = const.data_type #
        clip_length = const.clip_length
        which_ds = const.which_ds  # normal, lmdb
    else:
        print("get_dataset error")
    #
    # two-stream -> train_or_test -> normal_or_lmdb
    twostream_train_ds_map = {
        "normal": TwoStream_Train_DS,
        "lmdb": LMDBDataset_twostream_train,
    }
    twostream_test_ds_map = {
        "normal": test_dataset, #
        "lmdb": LMDBDataset_twostream_test,
    }
    twostream_ds_map = {
        "training": twostream_train_ds_map,
        "testing": twostream_test_ds_map,
    }
    #
    # rgb_or_op -> train_or_test -> normal_or_lmdb
    single_train_ds_map = {
        "normal": clip_Train_DS, # clip_Train_DS_debug,
        "lmdb": LMDBDataset_clip_train,
    }
    single_test_ds_map = {
        "normal": clip_Test_DS,
        "lmdb": LMDBDataset_clip_test,
    }
    single_ds_map = {
        "training": single_train_ds_map,
        "testing": single_test_ds_map,
    }
    #
    # schedule
    ds_map = {
        "rgb_op": twostream_ds_map,
        "rgb": single_ds_map,
        "op": single_ds_map,
    }
    ds_fn = ds_map[data_type][mode][which_ds]

    if mode == "training":
        ds = ds_fn(video_folder=video_folder, data_type=data_type,
                                   clip_length=clip_length)
    else:
        ds = None

    return ds



# ================================================================================ #
# unit test

# ------------------------------------------------------- #


def test_ds_load_cur():
    # training
    params = Config()
    params.mode = "training"
    params.exp_tag = "unit_test_ds_{}".format(str(round(time.time())))
    params.video_folder = "/p300/dataset/avenue/training/frames"
    params.dataset_name = "avenue"
    params.data_type = "rgb"
    params.clip_length = 5
    params.which_ds = "normal"  # normal, lmdb
    params.ds_params_map = "/p300/test_dir/ds_params_map.json"
    params.ds_params_pickle_save = "/p300/test_dir/ds_{}.pkl".format(
        str(round(time.time()))
    )  # save params as pickle

    get_dataset(params)

    # testing
    const2 = Config()
    const2.mode = "testing"
    const2.ds_params_map = params.ds_params_map
    const2.exp_tag = params.exp_tag
    # print("hello")
    ds = get_dataset(const2)
    # print("hello 2")
    ds.test("16")
    # print(ds)
    for idx in range(len(ds)):
        print("ds_sample: ", ds[idx].shape)


# ------------------------------------------------------- #

class stas_v1():

    def __init__(self, dataset, video_folder, data_type, clip_length=5,
                 size=(256,256), batch_size=0, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


    def single_train_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.num_workers)

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        for idx, sample in enumerate(self.loader):
            #
            if idx == 0:
                print("sample.size(): ", sample.size(), sample.min(), sample.max())
            #
            batch_size = sample.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def twostream_train_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.num_workers)

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        for idx, sample in enumerate(self.loader):
            #
            rgb ,op = sample["rgb"], sample["op"]
            if idx == 0:
                print("sample.size(): ", rgb.size(), rgb.min(), rgb.max())
                print("sample.size(): ", op.size(), op.min(), op.max())

            #
            batch_size = rgb.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def single_test_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        dataset = self.dataset

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = sorted(list(dataset.videos.keys()))
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            len_all_frame = dataset.videos[sub_video_name]["length"]
            dataset.test(sub_video_name)
            #
            for idx in range(len(dataset)):
                sample = dataset[idx]  # (t,c,h,w), since no batch
                if idx == 0:
                    print("sample.size(): ", sample.size())
                #
                batch_size = sample.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def twostream_test_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        dataset = self.dataset

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = sorted(list(dataset.rgb_ds.videos.keys()))
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            # import ipdb
            # ipdb.set_trace()
            dataset.test(sub_video_name)
            print("len of sub_video: {}".format(len(dataset)))
            #
            for idx in range(len(dataset)):  # 注意这里特别小心：控制边界
                sample = dataset[idx]  # (t,c,h,w), since no batch
                rgb = sample["rgb"]
                if idx == 0:
                    print("sample.size(): ", rgb.size())
                #
                batch_size = rgb.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)

def get_stas_v1(data_type, which_ds, num_his=4, mode=None,
                data_dir="/p300/dataset", dataset_name="avenue" ):
    #
    if data_type == "rgb_op":
        num_his = (num_his, num_his-1)
    if data_type == "op":
        num_his = num_his - 1

    # two-stream -> train_or_test -> normal_or_lmdb
    twostream_train_ds_map = {
        "normal": TwoStream_Train_DS,
        "lmdb": LMDBDataset_twostream_train,
    }
    twostream_test_ds_map = {
        "normal": TwoStream_Test_DS,
        "lmdb": LMDBDataset_twostream_test,
    }
    twostream_ds_map = {
        "training": twostream_train_ds_map,
        "testing": twostream_test_ds_map,
    }
    #
    # rgb_or_op -> train_or_test -> normal_or_lmdb
    single_train_ds_map = {
        "normal": clip_Train_DS, # clip_Train_DS_debug, #
        "lmdb": LMDBDataset_clip_train,
    }
    single_test_ds_map = {
        "normal": clip_Test_DS,
        "lmdb": LMDBDataset_clip_test,
    }
    single_ds_map = {
        "training": single_train_ds_map,
        "testing": single_test_ds_map,
    }
    #
    # schedule
    ds_map = {
        "rgb_op": twostream_ds_map,
        "rgb": single_ds_map,
        "op": single_ds_map,
    }
    # mode = "training"
    dataset_fn = ds_map[data_type][mode][which_ds]
    # dataset = clip_Train_DS
    # data_dir = "/p300/dataset"  # universial, in p300
    # dataset_name = "avenue"
    #
    if which_ds == "normal":
        path_rgb = os.path.join(data_dir,
                                "{}/{}/frames".format(dataset_name, mode))  #
        path_op = os.path.join(data_dir,
                               "{}/optical_flow/{}/frames/flow".format(dataset_name,
                                                                       mode))  #
    elif which_ds == "lmdb":
        path_rgb = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                                'rgb', mode)  #
        path_op = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                               'op', mode)  #
    else:
        print("# for dataset setting error")
    #
    if data_type == "rgb_op":
        video_folder = (path_rgb, path_op)
        clip_length = (num_his[0] + 1, num_his[1] + 1)
    else:
        tmp_mapp = {"rgb": path_rgb, "op": path_op, }  #
        video_folder = tmp_mapp[data_type]
        clip_length = num_his + 1

    video_path = {"rgb": path_rgb, "op": path_op,
                  "rgb_op": (path_rgb, path_op)}
    video_folder = video_path[data_type]
    print("video_folder: ", video_folder)
    size = (256, 256)
    dataset = dataset_fn(video_folder, data_type, clip_length, size)
    # print("dataset: ", dataset[0].size())

    batch_size = 16
    num_workers = 16

    obj = stas_v1(dataset=dataset, video_folder=video_folder, data_type=data_type,
                  batch_size=batch_size, num_workers=num_workers)

    return obj

def test_for_train(log_file, data_type, which_ds,
                   data_dir="/p300/dataset", dataset_name="avenue"):

    obj = get_stas_v1(data_type, which_ds, mode="training",
                      data_dir=data_dir, dataset_name=dataset_name)

    if data_type == "rgb_op":
        obj.twostream_train_load_time(log_file=log_file)
    else:
        obj.single_train_load_time(log_file=log_file)

def test_for_test(log_file, data_type, which_ds,
                  data_dir="/p300/dataset", dataset_name="avenue"):

    obj = get_stas_v1(data_type, which_ds, mode="testing",
                      data_dir=data_dir, dataset_name=dataset_name)

    if data_type == "rgb_op":
        obj.twostream_test_load_time(log_file=log_file)
    else:
        obj.single_test_load_time(log_file=log_file)



class stas_v2():

    def __init__(self, dataset, batch_size=0, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers



    def single_train_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.num_workers)
        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        for idx, sample in enumerate(self.loader):
            #
            if idx == 0:
                print("sample.size(): ", sample.size(), sample.min(), sample.max())
            #
            batch_size = sample.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def twostream_train_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=self.num_workers)

        # 粗糙计算 处理每个 batch 花费的时间 (batch之间的等待时间 + 实际处理batch时间)
        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        for idx, sample in enumerate(self.loader):
            #
            rgb ,op = sample["rgb"], sample["op"]
            if idx == 0:
                print("sample.size(): ", rgb.size(), rgb.min(), rgb.max())
                print("sample.size(): ", op.size(), op.min(), op.max())

            #
            batch_size = rgb.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def single_test_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        dataset = self.dataset

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = sorted(list(dataset.videos.keys()))
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            len_all_frame = dataset.videos[sub_video_name]["length"]
            dataset.test(sub_video_name)
            #
            for idx in range(len(dataset)):  # 注意这里特别小心：控制边界
                sample = dataset[idx]  # (t,c,h,w), since no batch
                if idx == 0:
                    print("sample.size(): ", sample.size())
                #
                batch_size = sample.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)




    def twostream_test_load_time(self, log_file='logger.log'):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        dataset = self.dataset # 直接 new 两个 然后 用 tuple 包起来 (不在代码层次封装了)
        rgb_ds, op_ds = dataset[0], dataset[1]

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = sorted(list(rgb_ds.videos.keys()))
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            # import ipdb
            # ipdb.set_trace()
            rgb_ds.test(sub_video_name)
            op_ds.test(sub_video_name)
            print("len of sub_video: {}".format(len(rgb_ds)))
            #
            for idx in range(len(rgb_ds)):  # 注意这里特别小心：控制边界
                rgb = rgb_ds[idx]  # (t,c,h,w), since no batch
                op = op_ds[idx]
                if idx == 0:
                    print("rgb.size(): ", rgb.size(), rgb.min(), rgb.max())
                    print("op.size(): ", op.size(), op.min(), op.max())
                #
                batch_size = rgb.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)
#
def get_stas_v2(data_type, which_ds, num_his=4, mode=None,
                data_dir="/p300/dataset", dataset_name="avenue" ):
    # which_ds 都是一样的 接口
    # mode = "training"
    # dataset = clip_Train_DS
    # data_dir = "/p300/dataset"  # universial, in p300
    # dataset_name = "avenue"  # 其实应该用 toy dataset 来做 unit test
    #
    if which_ds == "normal":
        path_rgb = os.path.join(data_dir,
                                "{}/{}/frames".format(dataset_name, mode))  #
        path_op = os.path.join(data_dir,
                               "{}/optical_flow/{}/frames/flow".format(dataset_name,
                                                                       mode))  #
    elif which_ds == "lmdb":
        path_rgb = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                                'rgb', mode)  #
        path_op = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                               'op', mode)  #
    else:
        print("# for dataset setting error")
    #
    # if data_type == "rgb_op":
    #     video_folder = (path_rgb, path_op)
    #     clip_length = (num_his[0] + 1, num_his[1] + 1)
    # else:
    #     tmp_mapp = {"rgb": path_rgb, "op": path_op, }  #
    #     video_folder = tmp_mapp[data_type]
    #     clip_length = num_his + 1

    video_path = {"rgb": path_rgb, "op": path_op,
                  "rgb_op": (path_rgb, path_op)}
    video_folder = video_path[data_type]
    print("video_folder: ", video_folder)
    size = (256, 256)
    #
    rgb_ds = clip_Test_DS(path_rgb, "rgb", 5, size)
    op_ds = clip_Test_DS(path_op, "op", 5, size)
    dataset = (rgb_ds, op_ds)

    batch_size = 16
    num_workers = 16

    obj = stas_v2(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    return obj
#
def test_for_test_v2(log_file, data_type, which_ds,
                  data_dir="/p300/dataset", dataset_name="avenue"):

    obj = get_stas_v2(data_type, which_ds, mode="testing",
                      data_dir=data_dir, dataset_name=dataset_name)

    if data_type == "rgb_op":
        obj.twostream_test_load_time(log_file=log_file)
    else:
        obj.single_test_load_time(log_file=log_file)
#



class stas_v3():
    def __init__(self, video_folder, num_workers=16):
        # self.batch_size = batch_size
        self.num_workers = num_workers
        self.video_folder = video_folder

    #
    def single_test_load_time(self, log_file):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        rgb_root = self.video_folder

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = os.listdir(rgb_root)
        all_sub_video_name_list.sort()
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            # import ipdb
            # ipdb.set_trace()
            rgb_folder = os.path.join(rgb_root, sub_video_name)
            rgb_ds = test_dataset(rgb_folder, 5, "rgb")
            #
            rgb_loader = DataLoader(rgb_ds, batch_size=len(rgb_ds),
                                 shuffle=False, num_workers=self.num_workers)

            print("len of sub_video: {}".format(len(rgb_ds)))
            #
            for tmp_r in rgb_loader: # 会执行 len(rgb_ds) 次 getitem, 但是循环 len(rgb_ds)/batch_size 次
                rgb_buff = tmp_r
                break #
            # for tmp_o in op_loader: # 只需加载一次
            #     op_buff = tmp_o
            #     break

            #
            iter_num = len(rgb_ds)
            for idx in range(iter_num):  # 注意这里特别小心：控制边界
                rgb = rgb_buff[idx]  # (t,c,h,w), since no batch
                if idx == 0:
                    print("rgb.size(): ", rgb.size(), rgb.min(), rgb.max())
                #
                batch_size = rgb.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)
    #
    def twostream_test_load_time(self, log_file):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        rgb_root, op_root = self.video_folder

        # 粗糙计算 处理每个 batch 花费的时间 (batch之间的等待时间 + 实际处理batch时间)
        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = os.listdir(rgb_root)
        all_sub_video_name_list.sort()
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            # import ipdb
            # ipdb.set_trace()
            rgb_folder = os.path.join(rgb_root, sub_video_name)
            rgb_ds = test_dataset(rgb_folder, 5, "rgb")
            op_folder = os.path.join(op_root, sub_video_name)
            op_ds = test_dataset(op_folder, 4, "op")
            #
            rgb_loader = DataLoader(rgb_ds, batch_size=len(rgb_ds),
                                 shuffle=False, num_workers=self.num_workers)
            op_loader = DataLoader(op_ds, batch_size=len(op_ds),
                                    shuffle=False, num_workers=self.num_workers)
            print("len of sub_video: {}".format(len(rgb_ds)))
            #
            for tmp_r in rgb_loader: # 会执行 len(rgb_ds) 次 getitem
                rgb_buff = tmp_r
                break # 只需加载一次, since len(rgb_ds) 一次就加载了所有data
            for tmp_o in op_loader: # 只需加载一次
                op_buff = tmp_o
                break
            print("rgb,op,buff size: ", rgb_buff.size(), op_buff.size())
            assert len(rgb_buff) == len(rgb_ds) and \
                len(op_buff) == len(op_ds) and \
                len(rgb_ds) == len(op_ds), "v3,twostream_test_load_time error"
            #
            iter_num = len(rgb_ds)
            for idx in range(iter_num):  # 注意这里特别小心：控制边界
                rgb = rgb_buff[idx]  # (t,c,h,w), since no batch
                op = op_buff[idx]
                if idx == 0:
                    print("rgb.size(): ", rgb.size(), rgb.min(), rgb.max())
                    print("op.size(): ", op.size(), op.min(), op.max())
                #
                batch_size = rgb.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)
#
def get_stas_v3(which_ds, mode=None,
                data_dir="/p300/dataset", dataset_name="avenue" ):
    # which_ds 都是一样的 接口
    # mode = "training"
    # dataset = clip_Train_DS
    # data_dir = "/p300/dataset"  # universial, in p300
    # dataset_name = "avenue"
    #
    if which_ds == "normal":
        path_rgb = os.path.join(data_dir,
                                "{}/{}/frames".format(dataset_name, mode))  #
        path_op = os.path.join(data_dir,
                               "{}/optical_flow/{}/frames/flow".format(dataset_name,
                                                                       mode))  #
    elif which_ds == "lmdb":
        path_rgb = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                                'rgb', mode)  #
        path_op = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                               'op', mode)  #
    else:
        print("# for dataset setting error")
    #
    # if data_type == "rgb_op":
    #     video_folder = (path_rgb, path_op)
    #     clip_length = (num_his[0] + 1, num_his[1] + 1)
    # else:
    #     tmp_mapp = {"rgb": path_rgb, "op": path_op, }  #
    #     video_folder = tmp_mapp[data_type]
    #     clip_length = num_his + 1

    # video_path = {"rgb": path_rgb, "op": path_op,
    #               "rgb_op": (path_rgb, path_op)}
    video_folder = (path_rgb, path_op)
    print("video_folder: ", video_folder)
    size = (256, 256)
    #
    # # rgb_ds = clip_Test_DS(path_rgb, "rgb", 5, size)
    # # op_ds = clip_Test_DS(path_op, "op", 5, size)
    # dataset = None

    # batch_size = 16
    num_workers = 16

    obj = stas_v3(video_folder, num_workers=num_workers)

    return obj
#
def test_for_test_v3(log_file, data_type, which_ds,
                  data_dir="/p300/dataset", dataset_name="avenue"):

    obj = get_stas_v3(which_ds, mode="testing",
                data_dir=data_dir, dataset_name=dataset_name)

    if data_type == "rgb_op":
        obj.twostream_test_load_time(log_file=log_file)
    else:
        obj.single_test_load_time(log_file=log_file)



#
#===================================================================================#
#
def test_1():
    mode = "training"
    dataset_name = "avenue"
    data_type = "rgb"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    test_for_train(log_file, data_type, which_ds)

def test_2():

    # import ipdb
    mode = "training"
    dataset_name = "avenue"

    data_type = "op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds)

def test_3():

    # import ipdb
    mode = "training"
    dataset_name = "avenue"

    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds)

# ===
def test_4():

    mode = "training"
    dataset_name = "avenue"

    data_type = "rgb"
    which_ds = "lmdb"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # import ipdb
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds)

def test_5():

    # import ipdb
    mode = "training"
    dataset_name = "avenue"
    data_type = "op"
    which_ds = "lmdb"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds)

def test_6():

    # import ipdb
    mode = "training"
    dataset_name = "avenue"
    data_type = "rgb_op"
    which_ds = "lmdb"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds)
# ===

#
def test_7():
    # import ipdb
    # ipdb.set_trace()
    mode = "testing"
    dataset_name = "avenue"

    data_type = "rgb"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    test_for_test(log_file, data_type, which_ds)

def test_8():

    # import ipdb
    mode = "testing"
    dataset_name = "avenue"
    data_type = "op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_test(log_file, data_type, which_ds)

def test_9():

    # import ipdb
    mode = "testing"
    dataset_name = "avenue"

    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_test(log_file, data_type, which_ds)

#
# =================
# 开始 shanghaitech
def test_10():

    # import ipdb
    mode = "testing"

    data_dir = "/p300/dataset"
    dataset_name = "shanghaitech"
    data_type = "rgb"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_test(log_file, data_type, which_ds,
                  data_dir=data_dir, dataset_name=dataset_name)

def test_11():

    # import ipdb
    mode = "testing"

    data_dir = "/p300/dataset"
    dataset_name = "shanghaitech"
    data_type = "op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_test(log_file, data_type, which_ds,
                  data_dir=data_dir, dataset_name=dataset_name)

def test_12():

    # import ipdb
    mode = "testing"

    data_dir = "/p300/dataset"
    dataset_name = "shanghaitech"
    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_test(log_file, data_type, which_ds,
                  data_dir=data_dir, dataset_name=dataset_name)

#

def test_13():

    # import ipdb
    mode = "training"

    data_dir = "/p300/dataset"
    dataset_name = "shanghaitech"
    data_type = "rgb"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds,
                  data_dir=data_dir, dataset_name=dataset_name)

def test_14():

    # import ipdb
    mode = "training"

    data_dir = "/p300/dataset"
    dataset_name = "shanghaitech"
    data_type = "op"
    which_ds = "normal"
    log_file = get_log_file(data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds,
                  data_dir=data_dir, dataset_name=dataset_name)

def test_15():

    # import ipdb
    mode = "training"

    data_dir = "/p300/dataset"
    dataset_name = "shanghaitech"
    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name)
    # ipdb.set_trace()
    test_for_train(log_file, data_type, which_ds,
                  data_dir=data_dir, dataset_name=dataset_name)


# ---------------------------------------------------- #
def test_9_2():

    # import ipdb
    mode = "testing"
    dataset_name = "shanghaitech"

    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="newts")
    # ipdb.set_trace()
    test_for_test_v2(log_file, data_type, which_ds, dataset_name=dataset_name)
def test_9_3():
    # import ipdb
    mode = "testing"
    dataset_name = "shanghaitech"

    data_type = "rgb_op"
    which_ds = "normal"
    data_dir = '/p300/dataset'
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="newts")
    # ipdb.set_trace()

    obj = get_stas_v2(data_type, which_ds, mode="testing",
                      data_dir=data_dir, dataset_name=dataset_name)

    if which_ds == "normal":
        path_rgb = os.path.join(data_dir,
                                "{}/{}/frames".format(dataset_name, mode))  #
        path_op = os.path.join(data_dir,
                               "{}/optical_flow/{}/frames/flow".format(dataset_name,
                                                                       mode))  #
    elif which_ds == "lmdb":
        path_rgb = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                                'rgb', mode)  #
        path_op = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
                               'op', mode)  #
    else:
        print("# for dataset setting error")
    #
    # if data_type == "rgb_op":
    #     video_folder = (path_rgb, path_op)
    #     clip_length = (num_his[0] + 1, num_his[1] + 1)
    # else:
    #     tmp_mapp = {"rgb": path_rgb, "op": path_op, }  #
    #     video_folder = tmp_mapp[data_type]
    #     clip_length = num_his + 1

    video_path = {"rgb": path_rgb, "op": path_op,
                  "rgb_op": (path_rgb, path_op)}
    video_folder = video_path[data_type]
    print("video_folder: ", video_folder)
    size = (256, 256)
    #
    rgb_ds = clip_Test_DS(path_rgb, "rgb", 5, size)
    op_ds = clip_Test_DS(path_op, "op", 5, size)
    dataset = (rgb_ds, op_ds)
    rgb_ds, op_ds = dataset[0], dataset[1]

    pre_time = time.time()
    cnt_batch, cnt_time = 0, 0
    all_sub_video_name_list = sorted(list(rgb_ds.videos.keys()))
    print("all_sub_video_name_list: ", all_sub_video_name_list)
    for video_id, sub_video_name in enumerate(all_sub_video_name_list):
        #
        # import ipdb
        # ipdb.set_trace()
        rgb_ds.test(sub_video_name)
        op_ds.test(sub_video_name)
        print("len of sub_video: {}".format(len(rgb_ds)))
        #
        for idx in range(len(rgb_ds)):  # 注意这里特别小心：控制边界
            rgb = rgb_ds[idx]  # (t,c,h,w), since no batch
            op = op_ds[idx]
            if idx == 0:
                print("rgb.size(): ", rgb.size(), rgb.min(), rgb.max())
                print("op.size(): ", op.size(), op.min(), op.max())
            #
            batch_size = rgb.size()[0]
            cur_time = time.time()
            delta_time = cur_time - pre_time
            log_comm = 'batch_size={}  |  cost_time={}  |  fps={}'.format(
                batch_size, int(delta_time), int(delta_time / batch_size))
            # logging.info(log_comm)
            print(log_comm)
            #
            cnt_batch += batch_size
            cnt_time += delta_time  # 累计每一个batch消耗时间
            #
            pre_time = cur_time
        print("{} complete!".format(sub_video_name))
    log_comm = '# ------------------------------- #' \
               'batch_total={}  |  cost_time={}  |  fps={}'.format(
        cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
    # logging.info(log_comm)
    print(log_comm)
    with open(log_file, "w") as fp:
        fp.write(log_comm)
# ----------------------------------------------------- #
def test_9_4():
    # import ipdb
    data_dir = "/p300/dataset"
    mode = "testing"
    dataset_name = "shanghaitech"

    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_9_4_0727")
    # ipdb.set_trace()
    test_for_test_v3(log_file, data_type, which_ds,
                  data_dir, dataset_name)


#===================================================================================#

class stas_v4():

    def __init__(self, dataset):
        self.dataset = dataset

    def single_test_load_time(self, log_file, num_his):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        dataset = self.dataset # 二级寻址即可，

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = sorted(list(dataset.videos.keys()))
        all_sub_video_name_list.sort()
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            len_all_frame = dataset.videos[sub_video_name]["length"]
            # dataset.test(sub_video_name)
            #
            for idx in range(num_his, len_all_frame):  # 注意这里特别小心：控制边界
                sample = dataset.get_clip(sub_video_name, idx-num_his, idx+1)  # (t,c,h,w), [start, end)
                if idx == 0:
                    print("sample.size(): ", sample.size())
                #
                batch_size = sample.size()[0]
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
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def twostream_test_load_time(self, log_file, num_his):
        # logging.basicConfig(filename=log_file, level=logging.INFO)
        rgb_ds, op_ds = self.dataset

        pre_time = time.time()
        cnt_batch, cnt_time = 0,0
        all_sub_video_name_list = sorted(list(rgb_ds.videos.keys()))
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            #
            # import ipdb
            # ipdb.set_trace()
            len_all_frame = rgb_ds.videos[sub_video_name]["length"]
            #
            for idx in range(num_his, len_all_frame):  # 注意这里特别小心：控制边界
                rgb_sample = rgb_ds.get_clip(sub_video_name, idx-num_his, idx+1)
                op_sample = op_ds.get_clip(sub_video_name, idx-num_his, idx)  # (t,c,h,w), [start, end)
                if idx == 0:
                    print("rgb.size(): ", rgb_sample.size(), rgb_sample.min(), rgb_sample.max())
                    print("op.size(): ", op_sample.size(), op_sample.min(), op_sample.max())

                #
                batch_size = rgb_sample.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)

def get_stas_v4(data_type, which_ds, num_his=4, mode=None,
                data_dir="/p300/dataset", dataset_name="avenue" ):
    #
    if which_ds == "normal":
        path_rgb = os.path.join(data_dir,
                                "{}/{}/frames".format(dataset_name, mode))  #
        path_op = os.path.join(data_dir,
                               "{}/optical_flow/{}/frames/flow".format(
                                   dataset_name, mode))  #
    # elif which_ds == "lmdb":
    #     path_rgb = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
    #                             'rgb', mode)  #
    #     path_op = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
    #                            'op', mode)  #
    # else:
    #     print("# for dataset setting error")
    if data_type == "rgb" or data_type == "op": # ######
        video_path = {"rgb": path_rgb, "op": path_op}
        video_folder = video_path[data_type]
        print("video_folder: ", video_folder)
        if data_type == "rgb":
            dataset = clip_Test_DS_naive(video_folder, "rgb", num_his + 1)
        if data_type == "op":
            dataset = clip_Test_DS_naive(video_folder, "op", num_his)
    if data_type == "rgb_op": ###
        rgb_ds = clip_Test_DS_naive(path_rgb, "rgb", num_his + 1)
        op_ds = clip_Test_DS_naive(path_op, "op", num_his)
        dataset = (rgb_ds, op_ds)

    return stas_v4(dataset)

def test_for_test_v4(log_file, data_type, which_ds,
                  data_dir="/p300/dataset", dataset_name="avenue"):

    obj = get_stas_v4(data_type, which_ds, mode="testing",
                      data_dir=data_dir, dataset_name=dataset_name)

    if data_type == "rgb_op":
        obj.twostream_test_load_time(log_file=log_file, num_his=4)
    else:
        obj.single_test_load_time(log_file=log_file, num_his=4)

## 具体执行测试 ============== #

def get_log_file(mode, data_type, which_ds, dataset_name, tag="xxx"):
    from ..utils.utils import get_dir
    root_path = get_dir("/p300/test_dir/stat_load_time_{}".format(tag))
    log_file = os.path.join(root_path,
                            "{}_{}_{}_{}.txt".format(mode, data_type, which_ds, dataset_name))

    return log_file

def test_x1():
    mode = "testing"
    dataset_name = "avenue"
    data_type = "rgb"
    which_ds = "normal"
    # IO = "ssd"
    # data_dir = "/ssd0/zhanghao"
    IO = "p300"
    data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x1_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir, dataset_name=dataset_name)

def test_x2():
    mode = "testing"
    dataset_name = "avenue"
    data_type = "op"
    which_ds = "normal"
    # IO = "ssd"
    # data_dir = "/ssd0/zhanghao"
    IO = "p300"
    data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x2_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir, dataset_name=dataset_name)

def test_x3():
    mode = "testing"
    dataset_name = "avenue"
    data_type = "rgb_op"
    which_ds = "normal"
    # IO = "ssd"
    # data_dir = "/ssd0/zhanghao"
    IO = "p300"
    data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x3_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir, dataset_name=dataset_name)

def test_x4():
    mode = "testing"
    dataset_name = "shanghaitech"
    data_type = "rgb"
    which_ds = "normal"
    # IO = "ssd"
    # data_dir = "/ssd0/zhanghao"
    IO = "p300"
    data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x4_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir, dataset_name=dataset_name)

def test_x5():
    mode = "testing"
    dataset_name = "shanghaitech"
    data_type = "op"
    which_ds = "normal"
    # IO = "ssd"
    # data_dir = "/ssd0/zhanghao"
    IO = "p300"
    data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x5_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir, dataset_name=dataset_name)

def test_x6():
    mode = "testing"
    dataset_name = "shanghaitech"
    data_type = "rgb_op"
    which_ds = "normal"
    # IO = "ssd"
    # data_dir = "/ssd0/zhanghao"
    IO = "p300"
    data_dir = "/p300/dataset" #
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x6_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir, dataset_name=dataset_name)

#
def test_x11():
    mode = "testing"
    dataset_name = "avenue"
    data_type = "rgb"
    which_ds = "normal"
    IO = "ssd"
    data_dir = "/ssd0/zhanghao"
    # IO = "p300"
    # data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x1_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir)

def test_x21():
    mode = "testing"
    dataset_name = "avenue"
    data_type = "op"
    which_ds = "normal"
    IO = "ssd"
    data_dir = "/ssd0/zhanghao"
    # IO = "p300"
    # data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x2_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir)

def test_x31():
    mode = "testing"
    dataset_name = "avenue"
    data_type = "rgb_op"
    which_ds = "normal"
    IO = "ssd"
    data_dir = "/ssd0/zhanghao"
    # IO = "p300"
    # data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x3_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir)

def test_x41():
    mode = "testing"
    dataset_name = "shanghaitech"
    data_type = "rgb"
    which_ds = "normal"
    IO = "ssd"
    data_dir = "/ssd0/zhanghao"
    # IO = "p300"
    # data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x4_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir)

def test_x51():
    mode = "testing"
    dataset_name = "shanghaitech"
    data_type = "op"
    which_ds = "normal"
    IO = "ssd"
    data_dir = "/ssd0/zhanghao"
    # IO = "p300"
    # data_dir = "/p300/dataset"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x5_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir)

def test_x61():
    mode = "testing"
    dataset_name = "shanghaitech"
    data_type = "rgb_op"
    which_ds = "normal"
    IO = "ssd"
    data_dir = "/ssd0/zhanghao"
    # IO = "p300"
    # data_dir = "/p300/dataset" #
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_x6_{}".format(IO))
    test_for_test_v4(log_file, data_type, which_ds, data_dir=data_dir)


class stas_v5():

    def __init__(self, rgb_root, op_root, data_type="rgb_op"):
        self.video_folder = rgb_root, op_root
        self.data_type = data_type

    def single_test_load_time(self, log_file, num_his):
        rgb_root, op_root = self.video_folder
        if self.data_type == "rgb":
            path = rgb_root
            len_clip = 5
        if self.data_type == "op":
            path = op_root
            len_clip = 4

        pre_time = time.time()
        cnt_batch, cnt_time = 0, 0
        all_sub_video_name_list = os.listdir(rgb_root)
        all_sub_video_name_list.sort()
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):

            rgb_folder = os.path.join(path, sub_video_name)
            rgb_ds = test_dataset(rgb_folder, len_clip, self.data_type)
            rgb_loader = DataLoader(rgb_ds, batch_size=16,
                                    shuffle=False, num_workers=16)
            print("len of sub_video: {}".format(len(rgb_ds)))
            #
            for idx, rgb in enumerate(rgb_loader):  # 会执行 len(rgb_ds) 次 getitem
                if idx == 0:
                    print("rgb.size(): ", rgb.size(), rgb.min(), rgb.max())
                #
                batch_size = rgb.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)

    def twostream_test_load_time(self, log_file, num_his):

        rgb_root, op_root = self.video_folder

        pre_time = time.time()
        cnt_batch, cnt_time = 0, 0
        all_sub_video_name_list = os.listdir(rgb_root)
        all_sub_video_name_list.sort()
        print("all_sub_video_name_list: ", all_sub_video_name_list)
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):

            rgb_folder = os.path.join(rgb_root, sub_video_name)
            rgb_ds = test_dataset(rgb_folder, 5, "rgb")
            op_folder = os.path.join(op_root, sub_video_name)
            op_ds = test_dataset(op_folder, 4, "op")
            #
            rgb_loader = DataLoader(rgb_ds, batch_size=16,
                                    shuffle=False, num_workers=8)
            op_loader = DataLoader(op_ds, batch_size=16,
                                   shuffle=False, num_workers=8)
            print("len of sub_video: {}".format(len(rgb_ds)))
            #
            for idx, (rgb,op) in enumerate(zip(rgb_loader, op_loader)):  # 会执行 len(rgb_ds) 次 getitem
                if idx == 0:
                    print("rgb.size(): ", rgb.size(), rgb.min(), rgb.max())
                    print("op.size(): ", op.size(), op.min(), op.max())
                #
                batch_size = rgb.size()[0]
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
            print("{} complete!".format(sub_video_name))
        log_comm = '# ------------------------------- #' \
                   'batch_total={}  |  cost_time={}  |  fps={}'.format(
            cnt_batch, int(cnt_time), int(cnt_time / cnt_batch))
        # logging.info(log_comm)
        print(log_comm)
        with open(log_file, "w") as fp:
            fp.write(log_comm)

def get_stas_v5(data_type, which_ds, num_his=4, mode=None,
                data_dir="/p300/dataset", dataset_name="avenue" ):
    #
    if which_ds == "normal":
        path_rgb = os.path.join(data_dir,
                                "{}/{}/frames".format(dataset_name, mode))  #
        path_op = os.path.join(data_dir,
                               "{}/optical_flow/{}/frames/flow".format(
                                   dataset_name, mode))  #
    # elif which_ds == "lmdb":
    #     path_rgb = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
    #                             'rgb', mode)  #
    #     path_op = os.path.join(data_dir, "lmdb_vad_final", dataset_name,
    #                            'op', mode)  #
    # else:
    #     print("# for dataset setting error")
    # if data_type == "rgb" or data_type == "op": # ######
    #     video_path = {"rgb": path_rgb, "op": path_op}
    #     video_folder = video_path[data_type]
    #     print("video_folder: ", video_folder)
    #     if data_type == "rgb":
    #         dataset = clip_Test_DS_naive(video_folder, "rgb", num_his + 1)
    #     if data_type == "op":
    #         dataset = clip_Test_DS_naive(video_folder, "op", num_his)
    # if data_type == "rgb_op": ###
    #     rgb_ds = test_dataset(path_rgb, "rgb", num_his + 1)
    #     op_ds = test_dataset(path_op, "op", num_his)
    #     dataset = (rgb_ds, op_ds)

    return stas_v5(path_rgb, path_op, data_type=data_type)

def test_for_test_v5(log_file, data_type, which_ds,
                  data_dir="/p300/dataset", dataset_name="avenue"):

    obj = get_stas_v5(data_type, which_ds, mode="testing",
                      data_dir=data_dir, dataset_name=dataset_name)

    if data_type == "rgb_op":
        obj.twostream_test_load_time(log_file=log_file, num_his=4)
    else:
        obj.single_test_load_time(log_file=log_file, num_his=4)


def test_fjscut_1():
    data_dir = "/p300/dataset"
    mode = "testing"
    dataset_name = "shanghaitech"

    data_type = "rgb_op"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_fjscut")

    test_for_test_v5(log_file, data_type, which_ds,
                     data_dir=data_dir, dataset_name=dataset_name)

def test_fjscut_2():
    data_dir = "/p300/dataset"
    mode = "testing"
    dataset_name = "shanghaitech"

    data_type = "rgb"
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_fjscut_rgb")

    test_for_test_v5(log_file, data_type, which_ds,
                     data_dir=data_dir, dataset_name=dataset_name)

def test_fjscut_3():
    data_dir = "/p300/dataset"
    mode = "testing"
    dataset_name = "shanghaitech"

    data_type = "op" #
    which_ds = "normal"
    log_file = get_log_file(mode, data_type, which_ds, dataset_name, tag="test_fjscut_op")

    test_for_test_v5(log_file, data_type, which_ds,
                     data_dir=data_dir, dataset_name=dataset_name)

if __name__ == '__main__':
    #
    # test_ds_load_cur()
    #
    '''
    avenue, training,
    clip_len = 5, 
    batch_size = 16, num_worker=16
    '''
    # ========================================================= #
    # training, avenue
    #
    # normal in p300, clip=5,4; batch_size=16, num_workers=16
    # test_1() # batch_total=15264  |  cost_time=428  |  fps=0
    #
    # test_2() # batch_total=15248  |  cost_time=1241  |  fps=0
    #
    # test_3() # batch_total=15264  |  cost_time=1351  |  fps=0
    #
    # normal in ssd_48
    #
    # test_1() # batch_total=15264  |  cost_time=309  |  fps=0
    # test_2() # batch_total=15248  |  cost_time=756  |  fps=0 # 加速显著
    # test_3() # batch_total=15264  |  cost_time=1393  |  fps=0
    #
    # ------------------------------------------------- #
    #  training
    # test_4()
    # test_5()
    # test_6()
    #
    # ======================================================== #
    # testing, p300, avenue
    #
    # normal in p300
    # #
    # test_7() #batch_total=60960  |  cost_time=572  | fps=0 #
    # test_8() #batch_total=60960  |  cost_time=572 | fps=0

    # ======================================================== #
    # testing, p300, shanghaitech
    
    #
    # # training, p300, shanghaitech (都很快)
    # test_13()
    # test_14()
    # test_15()

    # ======================================================== #
    # ======================================================== #
    # test_1()
    # test_2()
    # test_3()
    # # 跳过 lmdb 的 4,5,6
    # test_7()
    # test_8()
    # test_9() #
    # #
    # # shanghaitech, in p300, normal_ds, (data_type=3, mode=2 => 6 组)
    # test_10()
    # test_11()
    # test_12()
    # test_13()
    # test_14()
    # test_15()

    # ======================================================== #
    # test_9_2()
    # test_9_3()
    # test_9_4() # #batch_total=201815  |  cost_time=9903

    # ======================================================== #
    # naive inference code
    # test_x1()
    # test_x2()
    # test_x3() #batch_total=76200  |  cost_time=3931
    # test_x4() #batch_total=201815  |  cost_time=11082
    # test_x5()
    # test_x6() #batch_total=201815  |  cost_time=19630
    #
    # test_x11()
    # test_x21()
    # test_x31()
    # test_x41()
    # test_x51()
    # test_x61()

    # ======================================================== #
    #
    test_fjscut_1() #batch_total=40363(本质还是201815)  |  cost_time=966
    # test_fjscut_2()
    # test_fjscut_3()

    'python -m Code.dataset.__init__'

