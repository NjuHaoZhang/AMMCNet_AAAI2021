import _pickle as pickle
import pickle
import json
import os,time
from collections import namedtuple
#
import torch
from torchsummaryX import summary
#
from .flownet2.models import FlowNet2SD as flownet
from .pix2pix_networks import PixelDiscriminator
from .unet import (
    get_unet,
    get_unet_vq, get_unet_vq_topk,
    get_unet_vq_res, get_unet_vq_topk_res,
)
from .vqvae import (
    get_vqvae, get_vqvae_topk, get_vqvae_topk_res,
)
from .vqvae import get_twostream as get_twostream_vqvae
from .unet import get_twostream as get_twostream_unet

from .unet import (
    get_twostream_concat_dire,
    get_twostream_add_dire,
)


class Model(object):
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

def get_model(const):
    #
    mode = const.mode
    exp_tag = const.exp_tag
    net_params_map = const.net_params_map
    if mode == "training":
        init_params = Model()  # Model 其实就是个方便的 object, 并不是 真正意义的Model，所以随便用
        init_params.net_tag = const.net_tag
        init_params.data_type = const.data_type
        init_params.in_channel = const.in_channel
        init_params.out_channel = const.out_channel
        init_params.embed_dim = const.embed_dim
        init_params.n_embed = const.n_embed
        init_params.k = const.k
        net_params_pickle_save = const.net_params_pickle_save
        save_params(init_params, net_params_pickle_save, exp_tag, net_params_map)
        #
        net_tag = init_params.net_tag
        data_type = init_params.data_type
        in_channel = init_params.in_channel
        out_channel = init_params.out_channel
        embed_dim = init_params.embed_dim
        n_embed = init_params.n_embed
        k = init_params.k
    elif mode == "testing":
        init_params = load_params_test(net_params_map)
        #
        net_tag = init_params.net_tag
        data_type = init_params.data_type
        in_channel = init_params.in_channel
        out_channel = init_params.out_channel
        embed_dim = init_params.embed_dim
        n_embed = init_params.n_embed
        k = init_params.k
    else:
        print("get_model error")
    #
    net_map = {
        "vqvae": get_vqvae,
        "vqvae_topk": get_vqvae_topk,
        "vqvae_topk_res": get_vqvae_topk_res,
        "vqvae_twostream": get_twostream_vqvae,
        #
        "unet":get_unet,
        "unet_vq": get_unet_vq,
        "unet_vq_topk": get_unet_vq_topk,
        "unet_vq_res": get_unet_vq_res,
        "unet_vq_topk_res": get_unet_vq_topk_res,
        "unet_vq_twostream": get_twostream_unet,
        # for ablation
        "twostream_concat_dire":get_twostream_concat_dire,
        "twostream_add_dire":get_twostream_add_dire,
    }
    net = net_map[net_tag]
    generator =  net(in_channel=in_channel, out_channel=out_channel,
                    embed_dim=embed_dim, n_embed=n_embed, k=k)
    if net_tag == "vqvae_twostream" or net_tag == "unet_vq_twostream":
        if mode == "training":
            use_fixed_params = const.use_fixed_params
            if use_fixed_params:  # 固定 rgb and op branch to train bridge
                generator.fixed_rgb_op_branch()
    #
    discriminator = None
    flow_network = None
    if mode == "training":
        if data_type=="rgb_op":
            out_channel = out_channel[0]
            # 设置两个 D: discriminator = D_rgb, D_op
        discriminator = PixelDiscriminator(out_channel,
                                           const.d_num_filters, use_norm=False)  # only for rgb
        if data_type == "rgb" or data_type=="rgb_op":
            flow_network = flownet()
    #
    model = Model()
    model.generator = generator
    model.discriminator = discriminator
    model.flow_network = flow_network

    return model

def load_params(exp_tag_params_map_file, exp_tag):

    # (1) load params_save_path from json
    with open(exp_tag_params_map_file, "r") as fp:
        data_dict = json.load(fp)
        params_save_path = data_dict[exp_tag]

    # (2) load params from pickle
    with open(params_save_path, 'rb') as fp:
        params_obj = pickle.load(fp)

    return params_obj


def load_params_test(params_save_path):

    # load params from pickle
    with open(params_save_path, 'rb') as fp:
        params_obj = pickle.load(fp)

    return params_obj


def save_params(params, params_save_path, exp_tag, exp_tag_params_map_file):
    import fcntl, threading

    # (1) pickle, save model to disk; (方便，其实也可以 save __init__参数，再new一个model)
    with open(params_save_path, 'wb') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
        pickle.dump(params, fp, pickle.HIGHEST_PROTOCOL)

    # (2)json, exp_tag => cur_model_save_path;
    if not os.path.exists(exp_tag_params_map_file):
        data_dict = {}
        data_dict[exp_tag] = params_save_path
        with open(exp_tag_params_map_file, "w") as fp: # 创建文件 并直接写
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
            json.dump(data_dict, fp)
    else:
        with open(exp_tag_params_map_file, "r") as fp:  # 先读出原有内容再
            data_dict = json.load(fp)
            data_dict[exp_tag] = params_save_path
        #
        with open(exp_tag_params_map_file, "w+") as fp:  # 追加
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
            json.dump(data_dict, fp)

# ---------------------------------------------------------------------------- #
def get_model_v1(const):
    mode = const.mode
    get_fn_map = {
        "training": get_model_train_v1,
        "testing": get_model_test_v1,
    }

    return get_fn_map[mode](const)
def get_model_test_v1(const):
    exp_tag_model_map_file = const.exp_tag_model_map_file
    exp_tag = const.exp_tag

    return load_model_v1(exp_tag_model_map_file, exp_tag)
def get_model_train_v1(const):
    #
    mode = const.mode
    net_tag = const.net_tag
    data_type = const.data_type
    in_channel = const.in_channel
    out_channel = const.out_channel
    #
    exp_tag = const.exp_tag
    exp_tag_model_map_file = const.exp_tag_model_map_file
    model_pickle_save_path = const.model_pickle_save_path
    #
    # net_map
    if net_tag == "vqvae" or net_tag == "vqvae_topk" or net_tag == "vqvae_topk_res":
        net_map = {
            "vqvae": get_vqvae,
            "vqvae_topk": get_vqvae_topk,
            "vqvae_topk_res": get_vqvae_topk_res,
        }
        net = net_map[net_tag]
        generator =  net(in_channel=in_channel, out_channel=out_channel,
                        embed_dim=const.embed_dim, n_embed=const.n_embed, k=const.k)
    elif net_tag == "twostream_vqvae":
        out_channel = const.out_channel[0]
        generator = get_twostream_vqvae(in_channel=in_channel, out_channel=out_channel,
                                        embed_dim=const.embed_dim, n_embed=const.n_embed, k=const.k)
        if mode == "training":
            use_fixed_params = const.use_fixed_params
            if use_fixed_params:  # 固定 rgb and op branch to train bridge
                generator.fixed_rgb_op_branch()
    #
    #
    elif net_tag == "unet" or net_tag == "unet_vq" or \
                    net_tag == "unet_vq_topk" or net_tag == "unet_vq_topk_res":
        net_map = {
            "unet": get_unet,
            "unet_vq": get_unet_vq,
            "unet_vq_topk": get_unet_topk,
            "unet_vq_topk_res": get_unet_topk_res,
        }
        net = net_map[net_tag]
        generator = net(in_channel=in_channel, out_channel=out_channel,
                        embed_dim=const.embed_dim, n_embed=const.n_embed, k=const.k)
    elif net_tag == "unet_vq_twostream":
        out_channel = const.out_channel[0]
        generator = get_twostream_unet(in_channel=in_channel, out_channel=out_channel,
                                        embed_dim=const.embed_dim, n_embed=const.n_embed, k=const.k)
        if mode == "training":
            use_fixed_params = const.use_fixed_params
            if use_fixed_params:  # 固定 rgb and op branch to train bridge
                generator.fixed_rgb_op_branch()
    else:
        print("switch error !")
        exit()
    #
    discriminator = None
    flow_network = None
    if mode == "training" and (data_type == "rgb" or data_type=="rgb_op"):
        discriminator = PixelDiscriminator(out_channel,
            const.d_num_filters, use_norm=False)  # only for rgb
        flow_network = flownet()

    #
    model = Model()
    model.generator = generator
    model.discriminator = discriminator
    model.flow_network = flow_network

    # save generator config (暂时为了方便，直接 save model as pickle, 所以恢复也非常简单)
    save_model_v1(model, model_pickle_save_path, exp_tag, exp_tag_model_map_file)


    return model
# 这个用于上面的 train_save_model and test_load_model
def save_model_v1(model, model_save_path, exp_tag, exp_tag_model_map_file):
    import fcntl, threading

    # (1) pickle, save model to disk; (方便，其实也可以 save __init__参数，再new一个model)
    with open(model_save_path, 'wb') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
        pickle.dump(model, fp, pickle.HIGHEST_PROTOCOL)

    # (2)json, exp_tag => cur_model_save_path;
    if not os.path.exists(exp_tag_model_map_file):
        data_dict = {}
        data_dict[exp_tag] = model_save_path
        with open(exp_tag_model_map_file, "w") as fp: # 创建文件 并直接写
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
            json.dump(data_dict, fp)
    else:
        with open(exp_tag_model_map_file, "r") as fp:  # 先读出原有内容再
            data_dict = json.load(fp)
            data_dict[exp_tag] = model_save_path
        #
        with open(exp_tag_model_map_file, "w+") as fp:  # 追加
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX)  # 加锁
            json.dump(data_dict, fp)
def load_model_v1(exp_tag_model_map_file, exp_tag):

        # (1) load model_save_path from json
        with open(exp_tag_model_map_file, "r") as fp:
            data_dict = json.load(fp)
            model_save_path = data_dict[exp_tag]

        # (2) load model from pickle
        with open(model_save_path, 'rb') as fp:
            model = pickle.load(fp)

        return model
# ---------------------------------------------------------------------------- #


# ============================================================================== #
# unit test
def test_save_load_model_v1():

    const = Model()

    # (1) save model
    const.mode = "training"
    const.net_tag = "vqvae"
    const.data_type = "rgb"
    const.in_channel = 3
    const.out_channel = 3
    const.embed_dim = 64
    const.n_embed = 512
    const.k = 1
    const.d_num_filters = [128,256,512,512]
    #
    const.exp_tag = "unit_test"
    const.exp_tag_model_map_file = "/p300/test_dir/exp_tag_model_map_file.json"
    const.model_pickle_save_path = "/p300/test_dir/save_model.pkl"
    get_model_v1(const)

    # (2) load model
    const2 = Model()
    const2.mode = "testing"
    const2.exp_tag_model_map_file = const.exp_tag_model_map_file
    const2.exp_tag = const.exp_tag
    model = get_model_v1(const2)
    generator = model.generator

    # test generator
    net = generator
    b, c, h, w = 2, 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    print(generator)

def test_save_load_model():

    const = Model()

    # (1) save model
    const.mode = "training"
    const.net_tag = "vqvae_topk"
    const.data_type = "rgb"
    const.in_channel = 3
    const.out_channel = 3
    const.embed_dim = 64
    const.n_embed = 512
    const.k = 1
    const.d_num_filters = [128,256,512,512]
    #
    const.exp_tag = "unit_test_3"
    const.net_params_pickle_save = os.path.join("/p300/test_dir",
                                                'net-{}.pkl'.format(str(round(time.time()))))  #
    const.net_params_map = os.path.join("/p300/test_dir", "net_params_map.json")  #
    get_model(const)

    # (2) load model
    const2 = Model()
    const2.mode = "testing"
    const2.net_params_map = const.net_params_map
    const2.exp_tag = const.exp_tag
    model = get_model(const2)
    generator = model.generator

    # test generator
    net = generator
    b, c, h, w = 2, 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    print(generator)
# ============================================================================== #
if __name__ == '__main__':
    #
    # test_save_load_model_v1()

    #
    test_save_load_model()

    'python -m Code.models.__init__'


'''
vqvae-2:
return get_vqvae(in_channel=in_channel, out_channel=output_channel,
            channel=channel,
            n_res_block=2, n_res_channel=32,
            embed_dim=embed_dim, n_embed=n_embed, decay=0.99,) # 外部传进来的k不用即可
            
'''
#

'''
def get_model(in_channel, output_channel, embed_dim, n_embed, k,
                layer_nums, features_root):
    # k = 5, 试一下,add就是10
    #
    in_channel = (9*3, 8*2) # 9-rgb -> 1-rgb
    output_channel = (3, 2)
    layer_nums = 4
    features_root = 64
    embed_dim = 64
    n_embed = 512
    k = 1
    bn = False

    gen_x, gen_y, diff_x+diff_y, (embed_ind_tuple_x,embed_ind_tuple_x)
     = model(x,y)
    #
    x = torch.randn(1, 27, 256, 256)  # 9 -> 1, 3-channel
    y = torch.randn(1, 16, 256, 256)  # 8-> 1, 2-c
    gen_x: (1, 3, 256, 256) : [-1,1]
    gen_y: (1, 2, 256, 256): 任意
    model = UNetMem(in_channel, output_channel, layer_nums, features_root,
                  embed_dim, n_embed, k)
    return model
'''

