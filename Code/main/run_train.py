#
import argparse
#
#
from ..run_helper import get_helper
from ..dataset import get_dataset
from ..models import get_model
from ..models.losses import get_loss
from ..models.optimizer import get_optimizer
from .constant_train import const


def run_helper(const):

    # net_tag 有 unet,vqvae,vqvae_topk, vqvae_topk_res, twostream, 5种
    # data_type 有 rgb,op, twostream 3种 (op暂时不做)
    #
    model = get_model(const) # 只传部分参数
    dataset = get_dataset(const)
    loss = get_loss(const)
    optimizer = get_optimizer(model.generator, model.discriminator,
                              const)
    #
    helper = get_helper(model, dataset, loss, optimizer, const)
    #
    # run_func = {
    #     "training":helper.train,
    #     "testing":helper.test,
    # }
    # run_func[const.mode]()
    if const.data_type == "rgb_op":
        use_fixed_params = const.use_fixed_params
        if use_fixed_params:
            pass
        else:
            helper.train_from_multi_pretain()  # 使用baseline的pretrain
            # helper.train() # train from scratch
    else:
        helper.train_base()


if __name__ == "__main__":
    # 后期打算调参的值，设置为默认值，后面有空再调参；
    # 不可能修改的值就直接固定
    # 比如 UNet-4 的 4层，不可能修改了，还不如固定，如果真要改再开个new project
    #
    # TODO 需要调参：(1)做好实验预报告，(2) code中设置新的值，覆盖 const
    # 开始 覆盖 const 的 code, 更好的选择是从文件中读，专门一个调参的input file
    # const_params.py 是默认模板，而 tune_params.ini 才是每次都会变动的调参file
    # 每次都生成一个 tune list 来覆盖默认的 const, 来实现 按需调参
    #
    const.logger.info(const)
    #
    run_helper(const)

'''
# 一个技巧：加速 lmdb读取速度：确实是可以先运行几秒钟然后kill
# 再次 run, 就可以加速 lmdb loading

'''