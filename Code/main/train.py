import sys
sys.path.append('..')
import os

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

# from dataset import img_dataset
from dataset import two_stream_dataset
from models.unet import get_model
from models.pix2pix_networks import PixelDiscriminator
from models.flownet2.models import FlowNet2SD as flownet
#
import util
import utils
from losses import *
from constant import const
# from evaluate import evaluate


device_idx = const.GPU
device = torch.device("cuda:{}".format(device_idx))

batch_size = const.BATCH_SIZE
iterations = const.ITERATIONS
# epoch_num = const.EPOCH_NUM
num_his = const.NUM_HIS
height, width = 256, 256
flow_height, flow_width = const.FLOW_HEIGHT, const.FLOW_WIDTH
num_channel = const.NUM_CHANNEL
# eval_epoch = const.EVAL_EPOCH
sample_size = const.SAMPLE_SIZE

dataset_name = const.DATASET
data_dir = const.DATA_DIR
train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER

num_unet_layers = const.NUM_UNET_LAYES
features_root = const.UNET_FEATURE_ROOT
embed_dim = const.EMBED_DIM
n_embed = const.N_EMBED
k = const.K
#
discriminator_num_filters = const.D_NUM_FILTERS

l_num = const.L_NUM
alpha_num = const.ALPHA_NUM
lam_lp = const.LAM_LP
lam_gdl = const.LAM_GDL
lam_adv = const.LAM_ADV
lam_flow = const.LAM_FLOW
adversarial = (lam_adv != 0)
lam_latent = const.LAM_LATENT
lam_op_L1_loss = const.LAM_OP_L1

lr_g = const.LRATE_G
lr_d = const.LRATE_D
step_decay_G = const.STEP_DECAY_G
step_decay_D = const.STEP_DECAY_D

log_config_path  = const.LOG_CONFIG_TRAIN_PATH
log_save_root = const.LOG_SAVE_ROOT
logger = util.get_logger(log_config_path, log_save_root, "train")
summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR
#
pretrain = const.PRETRAIN
model_generator_save_dir = util.get_dir(os.path.join(snapshot_dir, "generator"))
model_discriminator_save_dir = util.get_dir(os.path.join(snapshot_dir, "discriminator"))
flow_model_path = const.FLOWNET_CHECKPOINT

logger.info(const)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# load ckpt TODO modify
def init_model(device, generator, discriminator,
               generator_model_dir=None, discriminator_model_dir=None,
               pretrain=False):
    g_step_g = 0
    if not pretrain:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
    else:
        generator, g_step_g = util.loader(generator, generator_model_dir,
                                device, logger)
        discriminator, g_step_d = util.loader(discriminator,
                                discriminator_model_dir,device, logger)
        assert g_step_g == g_step_d, "{} != {} g_step error".format(g_step_g,
                                                                    g_step_d)
    logger.info('initializing the model with Generator-Unet {} layers,'
                'PixelDiscriminator with filters {} '.format(num_unet_layers, discriminator_num_filters))

    return generator, discriminator, g_step_g

def train(dataset_loader, generator, discriminator, pretrain=False):

    # init_model
    generator, discriminator, g_step = init_model(device,
                        generator, discriminator,
                        generator_model_dir=model_generator_save_dir,
                        discriminator_model_dir=model_discriminator_save_dir,
                        pretrain=pretrain)
    logger.info("training from g_step = {}".format(str(g_step)))

    # training
    with SummaryWriter(summary_dir) as writer:
        while g_step < iterations:
            for sample in dataset_loader:
                #
                g_step += 1
                # print(g_step)
                # generator = generator.train()
                # discriminator = discriminator.train()
                rgb, op = sample["rgb"].to(device), sample["op"].to(device) # (b,t,c,h,w)
                rgb_input, op_input = rgb[:, :-1, :,:,:], op[:, :-1, :,:,:]
                rgb_target, op_target = rgb[:, -1, :,:,:], op[:, -1, :,:,:]
                rgb_input_last = rgb[:, -1, :, :, :]
                # target.to(device) # TODO 我记得之前有个文件试了一劳永逸的device solu
                #
                # (b,t*c,h,w)
                rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                                           rgb_input.shape[-2], rgb_input.shape[-1])
                op_input = op_input.view(op_input.shape[0], -1,
                                         op_input.shape[-2], op_input.shape[-1])
                #
                rgb_G_output, op_G_output, diff, _ = generator(rgb_input, op_input)
                #
                pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                   rgb_G_output.unsqueeze(2)], 2)
                gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                rgb_target.unsqueeze(2)], 2)

                flow_pred = (flow_network(pred_flow_esti_tensor * 255.0)/255.0).detach()
                flow_gt = (flow_network(gt_flow_esti_tensor * 255.0)/255.0).detach()

                # loss is used to optimize G
                g_adv_loss=adversarial_loss(discriminator(rgb_G_output))
                g_op_loss=op_loss(flow_pred,flow_gt)
                g_int_loss=int_loss(rgb_G_output,rgb_target)
                g_gd_loss=gd_loss(rgb_G_output,rgb_target)
                #  for op : just L1 loss TODO: adv 加入？？？需要channel==3
                g_op_L1_loss = L1_loss(rgb_G_output, rgb_target)
                #
                g_latent_loss = diff
                #
                g_loss=lam_adv * g_adv_loss + lam_gdl * g_gd_loss + \
                       lam_flow * g_op_loss + lam_lp * g_int_loss + \
                       lam_latent * g_latent_loss + lam_op_L1_loss * g_op_L1_loss

                #-------- (1) update optim_D -------
                d_loss = discriminate_loss(discriminator(rgb_target),
                                           discriminator(rgb_G_output.detach()))
                optimizer_D.zero_grad()
                # d_loss.requires_grad=True
                d_loss.backward()
                optimizer_D.step()
                #
                # ------- (2) update optim_G --------------
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                # -------- lr decay -------------------------
                scheduler_D.step()
                scheduler_G.step()

                # ========== log training state ======================== #

                if g_step % 10 == 0:
                    # --------  cal psnr,loss (log info) --------------
                    train_psnr = util.psnr_error(rgb_G_output, rgb_target)

                    log_info = \
                    'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(g_step, d_loss.item(),lr_d) +\
                    'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                    '                 Global      Loss : {}\n'.format(g_loss.item()) + \
                    '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(g_int_loss.item(), lam_lp,
                                                                                            g_int_loss.item() * lam_lp) +\
                    '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(g_gd_loss.item(), lam_gdl,
                                                                                            g_gd_loss.item() * lam_gdl) +\
                    '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(g_adv_loss.item(), lam_adv,
                                                                                            g_adv_loss.item() * lam_adv) +\
                    '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(g_op_loss.item(), lam_flow,
                                                                                            g_op_loss.item() * lam_flow) +\
                    '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(g_latent_loss.item(), lam_latent,
                                                                                              g_latent_loss.item() * lam_latent) + \
                    '                 PSNR  Error      : {}\n'.format(train_psnr)
                    logger.info(log_info)

                if g_step % 100 == 0:
                    writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                    writer.add_scalar('total_loss/g_loss', g_loss.item(), global_step=g_step)
                    writer.add_scalar('total_loss/d_loss', d_loss.item(), global_step=g_step)
                    writer.add_scalar('g_loss/adv_loss', g_adv_loss.item(), global_step=g_step)
                    writer.add_scalar('g_loss/op_loss', g_op_loss.item(), global_step=g_step)
                    writer.add_scalar('g_loss/int_loss', g_int_loss.item(), global_step=g_step)
                    writer.add_scalar('g_loss/gd_loss', g_gd_loss.item(), global_step=g_step)
                    writer.add_scalar('g_loss/latent_loss', g_latent_loss.item(), global_step=g_step)
                    #
                    writer.add_scalar('g_loss/g_op_L1_loss', g_op_L1_loss.item(), global_step=g_step)
                    #
                    if sample_size > rgb_G_output.size()[0]:
                        sample_size = rgb_G_output.size()[0]
                    vis_rgb = utils.utils.get_vis_tensor(torch.cat(
                        [rgb_G_output[:sample_size],rgb_target[:sample_size]],0),
                                                         "rgb", sample_size)
                    writer.add_image('image/train_rgb_output_target', vis_rgb, global_step=g_step)
                    # writer.add_images('image/train_rgb_output_target', rgb_target[:sample_size], global_step=g_step)
                    # writer.add_images('image/train__rgb', rgb_G_output[:sample_size], global_step=g_step)
                    #
                    vis_op = utils.utils.get_vis_tensor(torch.cat(
                        [op_G_output[:sample_size],op_target[:sample_size]],0),
                                                         "op", sample_size)
                    writer.add_image('image/train_op_output_target', vis_op, global_step=g_step)
                    # op_target = utils.utils.get_vis_tensor(op_target[:sample_size], "op", sample_size)
                    # op_G_output = utils.utils.get_vis_tensor(op_G_output[:sample_size], "op", sample_size)
                    # writer.add_image('image/train_target_op', op_target, global_step=g_step)
                    # writer.add_image('image/train_output_op', op_G_output, global_step=g_step)

                if g_step % 1000 == 0:
                    util.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                    util.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                #
                if g_step == iterations:
                    break




if __name__=='__main__':
    # 此处对于多帧的处理: channel-wise concat (t-dim 融合到 channel-dim)
    # 第二种处理: 首先将 t-dim 合并到 b-dim, 然后用 conv3d 做多帧融合 todo
    #
    frame_num = num_his + 1
    input_channels = (3 * num_his, 2*(num_his-1))
    output_channels = (3,2)

    # datasset
    # dataset = img_dataset.ano_pred_Dataset(train_folder, frame_num)
    # sum_path = utils.get_dir("/p300/py_ano_pred_mem")
    # writer = SummaryWriter(log_dir=sum_path)
    # data_dir = "/p300/dataset"  # universial, in p300
    # dataset_name = "avenue"  # 其实应该用 toy dataset 来做 unit test
    mode = "training"
    path_rgb = os.path.join(data_dir,
                "{}/{}/frames".format(dataset_name,mode))  #
    path_optical_flow = os.path.join(data_dir,
                "{}/optical_flow/{}/frames/flow".format(dataset_name,mode))  #
    video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
    dataset = two_stream_dataset.TwoStream_Train_DS(video_folder,
                                                    (frame_num,frame_num-1))
    dataset_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    # model
    generator = get_model(in_channel=input_channels, output_channel=output_channels,
                          embed_dim=embed_dim, n_embed=n_embed, k=k,
                          layer_nums=num_unet_layers, features_root=features_root)
    discriminator = PixelDiscriminator(3, discriminator_num_filters, use_norm=False)
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    flow_network = flownet().to(device).eval() # 注意 flownet 不训练
    flow_network.load_state_dict(torch.load(flow_model_path)['state_dict'])

    # optimizer (flownet没有optimizer, so 无法更新)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    # lr_decay
    scheduler_G = StepLR(optimizer_G, step_size=step_decay_G, gamma=0.1)
    scheduler_D = StepLR(optimizer_D, step_size=step_decay_D, gamma=0.1)
    # "cycle": OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataset_loader), epochs=iterations)

    # loss
    adversarial_loss = Adversarial_Loss().to(device)
    discriminate_loss = Discriminate_Loss().to(device)
    # gd_loss = Gradient_Loss(alpha_num, num_channel).to(device)
    gd_loss = gradient_loss
    op_loss = Flow_Loss().to(device)
    int_loss = Intensity_Loss(l_num).to(device)
    L1_loss = nn.L1Loss().to(device)
    # 加个 feature_loss in cvpr2020, TODO


    train(dataset_loader, generator, discriminator, pretrain)