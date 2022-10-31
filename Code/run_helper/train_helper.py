#
import os, time
from ..main.constant_train import const
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= const.gpu_idx
#
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
#
from ..utils import utils
# from ..models.losses import Twostream_Loss,Discriminate_Loss


class train_twostream_Helper(object):
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
        self.loss = loss  # 仅仅只有 g_loss 无 d_loss
        self.optimizer = optimizer
        self.params = params # 直接传 const

    def train(self):
        # params collect
        # device = torch.device("cuda:{}".format(self.params.gpu))
        batch_size = self.params.batch_size
        num_workers = self.params.num_workers
        #
        model_generator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "generator"))
        model_discriminator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "discriminator"))
        flow_model_path = self.params.flow_model_path
        pretrain = self.params.pretrain
        summary_dir = self.params.summary_dir
        iterations = self.params.iterations
        logger = self.params.logger
        sample_size = self.params.sample_size
        #
        lr_g = self.params.lr_g
        lr_d = self.params.lr_d
        lam_adv = self.params.lam_adv
        lam_gdl = self.params.lam_gdl
        lam_flow = self.params.lam_flow
        lam_lp = self.params.lam_lp
        lam_latent = self.params.lam_latent
        lam_op_l1 = self.params.lam_op_l1
        #
        # model
        generator = self.model.generator
        discriminator = self.model.discriminator
        flow_network = self.model.flow_network
        #
        dataset_loader = DataLoader(self.dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)
        #
        # optimizer & scheduler
        optimizer_D = self.optimizer.optimizer_D
        optimizer_G = self.optimizer.optimizer_G
        scheduler_D = self.optimizer.scheduler_D # lr_decay
        scheduler_G = self.optimizer.scheduler_G
        #
        g_loss_fn = self.loss.g_loss
        d_loss_fn = self.loss.d_loss
        # model init (要么从0init, 要么pretrain)
        generator, discriminator, g_step = \
            utils.init_model(generator, discriminator,
                       generator_model_dir=model_generator_save_dir,
                       discriminator_model_dir=model_discriminator_save_dir,
                       pretrain=pretrain, logger=logger)
        flow_network.load_state_dict(torch.load(flow_model_path)['state_dict'])
        #
        generator = generator.cuda().train()
        discriminator = discriminator.cuda().train()
        flow_network = flow_network.cuda().eval() # 注意是 eval()
        #
        # train
        with SummaryWriter(summary_dir) as writer:
            while g_step < iterations:
                for sample in dataset_loader: # (b,t,c,h,w)
                    #
                    g_step += 1
                    # print(g_step)
                    rgb, op = sample["rgb"].cuda(), sample["op"].cuda()
                    rgb_input, op_input = rgb[:, :-1, :, :, :], op[:, :-1, :, :, :]
                    rgb_target, op_target = rgb[:, -1, :, :, :], op[:, -1, :, :, :]
                    rgb_input_last = rgb[:, -1, :, :, :]
                    #
                    # (b,t*c,h,w)
                    rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                                               rgb_input.shape[-2], rgb_input.shape[-1])
                    op_input = op_input.view(op_input.shape[0], -1,
                                             op_input.shape[-2], op_input.shape[-1])
                    #
                    rgb_G_output, op_G_output, latent_diff = generator(rgb_input, op_input)
                    #
                    pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                       rgb_G_output.unsqueeze(2)], 2)
                    gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                     rgb_target.unsqueeze(2)], 2)
                    # flownet not update
                    flow_pred = (flow_network(pred_flow_esti_tensor * 255.0) / 255.0).detach()
                    flow_gt = (flow_network(gt_flow_esti_tensor * 255.0) / 255.0).detach()

                    # loss is used to optimize G
                    d_gen = discriminator(rgb_G_output)
                    g_loss = g_loss_fn(flow_pred, flow_gt,
                                       rgb_G_output, rgb_target,
                                       op_G_output, op_target,
                                       latent_diff, d_gen)

                    # ======= backward ========================================== #
                    #
                    # -------- (1) update optim_D -------
                    # 注意detach
                    d_real, d_gen = discriminator(rgb_target), \
                                    discriminator(rgb_G_output.detach())
                    d_loss = d_loss_fn(d_real, d_gen) #单独处理
                    #
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
                        d_loss = d_loss.item()
                        g_loss = g_loss_fn.g_loss
                        g_adv_loss = g_loss_fn.g_adv_loss
                        g_flow_loss = g_loss_fn.g_flow_loss
                        g_int_loss = g_loss_fn.g_int_loss
                        g_gd_loss = g_loss_fn.g_gd_loss
                        g_latent_loss = g_loss_fn.g_latent_loss  # 直接传进来
                        g_op_L1_loss = g_loss_fn.g_op_L1_loss
                        #
                        train_psnr = utils.psnr_error(rgb_G_output, rgb_target)

                        log_info = \
                            'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                                    g_step, d_loss, lr_d) + \
                            'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                            '                 Global      Loss : {}\n'.format(g_loss) + \
                            '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_int_loss, lam_lp,g_int_loss * lam_lp) + \
                            '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                            '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                            '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                            '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_latent_loss, lam_latent, g_latent_loss * lam_latent) + \
                            '                 op_L1       Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_op_L1_loss, lam_op_l1, g_op_L1_loss * lam_op_l1) + \
                            '                 PSNR  Error      : {}\n'.format(train_psnr)
                        logger.info(log_info)

                    if g_step % 100 == 0:
                        writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                        writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                        writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_latent_loss', g_latent_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_op_L1_loss', g_op_L1_loss, global_step=g_step)
                        #
                        if sample_size > rgb_G_output.size()[0]:
                            sample_size = rgb_G_output.size()[0]
                        #
                        vis_rgb = utils.get_vis_tensor(torch.cat(
                            [rgb_G_output[:sample_size], rgb_target[:sample_size]], 0),
                            "rgb", sample_size)
                        writer.add_image('image/train_rgb_output_target',
                                         vis_rgb, global_step=g_step)
                        #
                        vis_op = utils.get_vis_tensor(torch.cat(
                            [op_G_output[:sample_size], op_target[:sample_size]], 0),
                            "op", sample_size)
                        writer.add_image('image/train_op_output_target',
                                         vis_op, global_step=g_step)
                        #
                    if g_step % 1000 == 0:
                        utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                        utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    if g_step == iterations:
                        break

        # t_delta = time.time() - t_start
        # fps = t_delta / (g_step * batch_size)  # 近似，因为 batch 最后一个不是满的
        # 出 train_report: gen_train_report
        # logger.info("fps = ".format(fps))
        logger.info("training complete! ")

    def train_from_multi_pretain(self):
        # ======== result save ==================================================== #
        # 与具体 network 无关
        step_log = self.params.step_log
        step_summ = self.params.step_summ
        step_save_ckpt = self.params.step_save_ckpt
        summary_dir = self.params.summary_dir
        vis_sample_size = self.params.vis_sample_size
        logger = self.params.logger
        # ======= data ===================================================== #
        # 与具体 network 无关
        batch_size = self.params.batch_size
        num_workers = self.params.num_workers
        dataset_loader = DataLoader(self.dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    pin_memory=True)  # pin_memory
        # ======= model ===========================================================#
        # 与具体 network 无关，但 rgb 与 op 有区别
        pretrain = self.params.pretrain
        flow_model_path = self.params.flow_model_path
        rgb_model_path = self.params.rgb_model_path
        op_model_path = self.params.op_model_path
        model_generator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "generator"))
        model_discriminator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt,  "discriminator")) # only for rgb
        #
        generator = self.model.generator
        discriminator = self.model.discriminator
        flow_network = self.model.flow_network
        # model init (要么从0 init, 要么pretrain)
        generator, discriminator, g_step = \
            utils.init_model(generator, discriminator,
                             generator_model_dir=model_generator_save_dir,
                             discriminator_model_dir=model_discriminator_save_dir,
                             logger=logger) # D 会做none过滤
        flow_network.load_state_dict(torch.load(flow_model_path)['state_dict'])
        if pretrain: # from rgb and op single branch
            generator, g_step = utils.loader_rgb_op_branch(generator,
                       rgb_model_path, op_model_path, logger)
            # for discriminator ? (TODO)

        # ======== loss, optimizer, scheduler ==================================== #
        # 与network 有关，rgb 与 op 有区别
        # loss (根据 loss_tag 调度)
        g_loss_fn = self.loss.g_loss
        d_loss_fn = self.loss.d_loss
        iterations = self.params.iterations
        lam_adv = self.params.lam_adv
        lam_gdl = self.params.lam_gdl
        lam_flow = self.params.lam_flow
        lam_lp = self.params.lam_lp
        lam_latent = self.params.lam_latent
        # for op
        lam_lp_op = self.params.lam_lp_op
        # lam_adv_op = self.params.lam_adv_op
        # optimizer
        optimizer_D = self.optimizer.optimizer_D
        optimizer_G = self.optimizer.optimizer_G
        scheduler_D = self.optimizer.scheduler_D # lr_decay
        scheduler_G = self.optimizer.scheduler_G
        # TODO twostream 可能需要针对不同 part 采样不同 lr and lr_decay and optimizer
        # 上面所有直接赋值的不用 if 隔离分类，因为初始化有值
        #
        # ======== train ========================================================== #
        generator = generator.cuda().train()
        discriminator = discriminator.cuda().train()
        flow_network = flow_network.cuda().eval() # 注意是 eval()
        #
        t_start = time.time()
        pre_time = time.time()
        pre_time_data = time.time()
        with SummaryWriter(summary_dir) as writer:
            while g_step < iterations:
                for sample in dataset_loader: # (b,t,c,h,w)
                    #
                    cost_time_data = time.time() - pre_time_data
                    g_step += 1
                    # print(g_step)
                    rgb, op = sample["rgb"].cuda(), sample["op"].cuda()
                    rgb_input, op_input = rgb[:, :-1, :, :, :], op[:, :-1, :, :, :]
                    rgb_target, op_target = rgb[:, -1, :, :, :], op[:, -1, :, :, :]
                    rgb_input_last = rgb[:, -1, :, :, :]
                    #
                    # (b,t*c,h,w)
                    rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                                               rgb_input.shape[-2], rgb_input.shape[-1])
                    op_input = op_input.view(op_input.shape[0], -1,
                                             op_input.shape[-2], op_input.shape[-1])
                    #
                    rgb_G_output, op_G_output, latent_diff, _ = generator(rgb_input, op_input)
                    #
                    pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                       rgb_G_output.unsqueeze(2)], 2)
                    gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                     rgb_target.unsqueeze(2)], 2)
                    # flownet not update, 所以 detach
                    # Input for flownet2sd is in (0, 255). 而此处 tensor is in (-1,1)
                    flow_pred = (flow_network((pred_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                    flow_gt = (flow_network((gt_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                    # loss is used to optimize G
                    d_gen = discriminator(rgb_G_output)
                    g_loss_bp = g_loss_fn(flow_pred, flow_gt,
                                       rgb_G_output, rgb_target,
                                       op_G_output, op_target,
                                       latent_diff, d_gen)
                    # ======= backward ========================================== #
                    #
                    # -------- (1) update optim_D -------
                    # 注意detach
                    d_real, d_gen = discriminator(rgb_target), \
                                    discriminator(rgb_G_output.detach())
                    d_loss_bp = d_loss_fn(d_real, d_gen) #单独处理
                    #
                    optimizer_D.zero_grad()
                    # d_loss.requires_grad=True
                    d_loss_bp.backward()
                    optimizer_D.step()
                    #
                    # ------- (2) update optim_G --------------
                    optimizer_G.zero_grad()
                    g_loss_bp.backward()
                    optimizer_G.step()

                    # -------- lr decay -------------------------
                    scheduler_D.step()
                    scheduler_G.step()

                    # ========== log training state ======================== #
                    # (1) log
                    if g_step % step_log == 0:
                        # --------  cal psnr,loss (log info) --------------
                        d_loss = d_loss_bp.item()
                        g_loss = g_loss_fn.g_loss
                        g_adv_loss = g_loss_fn.g_adv_loss
                        g_int_loss = g_loss_fn.g_int_loss
                        g_flow_loss = g_loss_fn.g_flow_loss
                        g_gd_loss = g_loss_fn.g_gd_loss
                        train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                        g_int_loss_op = g_loss_fn.g_int_loss_op
                        train_psnr_op = utils.psnr_error(op_G_output, op_target) # normalize to (-1,1), 所以有界
                        #
                        lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                        lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                        #
                        cost_time = time.time() - pre_time
                        #
                        log_info = \
                            'time of cur_step :          {:.2f}                           \n'.format(
                                cost_time) + \
                            'time of data_load:          {:.2f}                            \n'.format(
                                cost_time_data) + \
                            'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                g_step, d_loss, lr_d) + \
                            'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                            '                 Global      Loss : {}\n'.format(g_loss) + \
                            '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                g_int_loss, lam_lp, g_int_loss * lam_lp) + \
                            '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                            '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                            '                 flow Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                            '                 PSNR  Error      : {}\n'.format(train_psnr) + \
                            '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                            '                 PSNR_op  Error      : {}\n'.format(train_psnr_op)

                        logger.info(log_info)
                    # (2) summ
                    if g_step % step_summ == 0:
                        writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                        writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                        writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_int_loss_op', g_int_loss_op, global_step=g_step)
                        writer.add_scalar('psnr/train_psnr_op', train_psnr_op, global_step=g_step)
                        #
                        vis_sample_size = self.params.vis_sample_size
                        if vis_sample_size > rgb_G_output.size()[0]:
                            vis_sample_size = rgb_G_output.size()[0]
                        vis_rgb = utils.get_vis_tensor(torch.cat(
                            [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                            "rgb", vis_sample_size)
                        writer.add_image('image/train_rgb_output_target',
                                         vis_rgb, global_step=g_step)
                        vis_op = utils.get_vis_tensor(torch.cat(
                            [op_G_output[:vis_sample_size], op_target[:vis_sample_size]], 0),
                            "op", vis_sample_size)
                        writer.add_image('image/train_op_output_target',
                                         vis_op, global_step=g_step)
                    # (3) save model_ckpt
                    if g_step % step_save_ckpt == 0:
                        utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                        utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    if g_step == iterations:
                        break
                    #
                    pre_time_data = time.time()  # 更新为 pre
                    pre_time = time.time()

        t_delta = time.time() - t_start
        fps = t_delta / (g_step * batch_size)  # 近似，因为 batch 最后一个不是满的
        # 出 train_report: gen_train_report
        logger.info("fps = {}".format(fps))
        logger.info("training complete! ")

    def train_base(self):
        # ======== result save ==================================================== #
        # 与具体 network 无关
        step_log = self.params.step_log
        step_summ = self.params.step_summ
        step_save_ckpt = self.params.step_save_ckpt
        summary_dir = self.params.summary_dir
        vis_sample_size = self.params.vis_sample_size
        logger = self.params.logger
        # ======= data ===================================================== #
        # 与具体 network 无关
        batch_size = self.params.batch_size
        num_workers = self.params.num_workers
        dataset_loader = DataLoader(self.dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    pin_memory=True)  # pin_memory
        # ======= model ===========================================================#
        # 与具体 network 无关，但 rgb 与 op 有区别
        pretrain = self.params.pretrain
        model_generator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "generator"))
        model_discriminator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt,  "discriminator"))
        flow_model_path = self.params.flow_model_path
        #
        generator = self.model.generator
        discriminator = self.model.discriminator # op => none
        flow_network = self.model.flow_network # op => none
        # model init (要么从0 init, 要么pretrain)
        generator, discriminator, g_step = \
            utils.init_model(generator, discriminator,
                             generator_model_dir=model_generator_save_dir,
                             discriminator_model_dir=model_discriminator_save_dir,
                             logger=logger) # D 会做none过滤
        generator = generator.cuda().train()
        discriminator = discriminator.cuda().train() # op 也需要 D (pixel_D)
        data_type = self.params.data_type
        if data_type == "rgb":  # op branch 的 D 初始化为 None
            flow_network.load_state_dict(torch.load(flow_model_path)['state_dict'])
            flow_network = flow_network.cuda().eval()  # 注意是 eval()
        # ======== loss, optimizer, scheduler ==================================== #
        # 与network 有关，rgb 与 op 有区别
        # loss (根据 loss_tag 调度)
        g_loss_fn = self.loss.g_loss
        d_loss_fn = self.loss.d_loss
        iterations = self.params.iterations
        lam_adv = self.params.lam_adv
        lam_gdl = self.params.lam_gdl
        lam_flow = self.params.lam_flow
        lam_lp = self.params.lam_lp
        lam_latent = self.params.lam_latent
        # for op
        lam_lp_op = self.params.lam_lp_op
        lam_adv_op = self.params.lam_adv_op
        # optimizer
        optimizer_D = self.optimizer.optimizer_D
        optimizer_G = self.optimizer.optimizer_G
        scheduler_D = self.optimizer.scheduler_D # lr_decay
        scheduler_G = self.optimizer.scheduler_G
        # TODO twostream 可能需要针对不同 part 采样不同 lr and lr_decay and optimizer
        # 上面所有直接赋值的不用 if 隔离分类，因为初始化有值
        #
        # ======== train ========================================================== #
        t_start = time.time()
        pre_time = time.time()
        pre_time_data = time.time()
        with SummaryWriter(summary_dir) as writer:
            while g_step < iterations:
                for sample in dataset_loader: # (b,t,c,h,w)
                    #
                    g_step += 1
                    cost_time_data = time.time() - pre_time_data
                    #
                    # 根据不同情况调度 不同 input
                    rgb = sample.cuda()
                    # print("range is:", rgb.min(), rgb.max())
                    rgb_input = rgb[:, :-1, :, :, :]
                    rgb_target = rgb[:, -1, :, :, :]
                    rgb_input_last = rgb[:, -1, :, :, :]
                    # (b,t*c,h,w)
                    rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                                               rgb_input.shape[-2], rgb_input.shape[-1])

                    # TODO:(1) v2,v4 还有根据v1改，(2) v1~v4 提取公共部分
                    # 根据 loss_tag 来拆分出不同子函数
                    def inference_v1(rgb_input, rgb_target, rgb_input_last):
                        # "rgb_int_gdl_flow_adv"
                        # ======== forward ======================================================= #
                        #
                        rgb_G_output = generator(rgb_input)
                        #
                        pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                           rgb_G_output.unsqueeze(2)], 2)
                        gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                         rgb_target.unsqueeze(2)], 2)
                        # flownet not update, 所以 detach
                        # Input for flownet2sd is in (0, 255). 而此处 tensor is in (-1,1)
                        flow_pred = (flow_network((pred_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        flow_gt = (flow_network((gt_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        #
                        # pred_flow_esti_tensor = torch.cat([
                        #     rgb_input_last.view(-1,3,1,rgb_input_last.shape[-2],rgb_input_last.shape[-1]),
                        #     rgb_G_output.view(-1,3,1,rgb_G_output.shape[-2],rgb_G_output.shape[-1])],
                        #     2)
                        # gt_flow_esti_tensor = torch.cat([
                        #     rgb_input_last.view(-1,3,1,rgb_input_last.shape[-2],rgb_input_last.shape[-1]),
                        #     rgb_target.view(-1,3,1,rgb_target.shape[-2],rgb_target.shape[-1])],
                        #     2)
                        # flow_gt=flow_network(gt_flow_esti_tensor*255.0)
                        # flow_pred=flow_network(pred_flow_esti_tensor*255.0)
                        #
                        d_gen = discriminator(rgb_G_output) # loss is used to optimize G
                        #
                        g_loss_bp = g_loss_fn(flow_pred, flow_gt,
                                           rgb_G_output, rgb_target,d_gen)
                        # ======= backward ======================================================== #
                        #
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        #
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        # (1) log
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            d_loss = d_loss_bp.item()
                            g_loss = g_loss_fn.g_loss
                            g_adv_loss = g_loss_fn.g_adv_loss
                            g_int_loss = g_loss_fn.g_int_loss
                            g_flow_loss = g_loss_fn.g_flow_loss
                            g_gd_loss = g_loss_fn.g_gd_loss
                            train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            #
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss, lam_lp, g_int_loss * lam_lp) + \
                                '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                                '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                                '                 PSNR  Error      : {}\n'.format(train_psnr)
                            logger.info(log_info)
                        # (2) summ
                        if g_step % step_summ == 0:
                            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                            writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_rgb_output_target',
                                             vis_rgb, global_step=g_step)
                        # (3) save model_ckpt
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    def inference_v2(rgb_input, rgb_target, rgb_input_last):
                        # "op_int_adv" TODO 传参就要改，加 adv
                        # ======== forward ======================================================= #
                        #
                        rgb_G_output = generator(rgb_input)
                        d_gen = discriminator(rgb_G_output)  # loss is used to optimize G
                        g_loss_bp = g_loss_fn(rgb_G_output, rgb_target, d_gen)
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        #
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        # (1) log
                        if g_step % step_log == 0: # step_log
                            # --------  cal psnr,loss (log info) --------------
                            d_loss_op = d_loss_bp.item()
                            g_loss_op = g_loss_fn.g_loss_op
                            g_int_loss_op = g_loss_fn.g_int_loss_op
                            g_adv_loss_op = g_loss_fn.g_adv_loss_op

                            epe_error = utils.epe_error(rgb_G_output, rgb_target)
                            # op value 是 unbounded的，不是 (x,1), 所以 直接用 psnr 不合适
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            ##
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss_op, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss_op) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss_op, lam_adv_op, g_adv_loss_op * lam_adv_op) + \
                                '                 EPE  Error      : {}\n'.format(epe_error)
                            logger.info(log_info)
                        # (2) summ
                        if g_step % step_summ == 0: # step_summ
                            writer.add_scalar('epe/train_epe_error', epe_error, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss_op', g_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss_op', g_int_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss_op', g_adv_loss_op, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_op_output_target',
                                             vis_rgb, global_step=g_step)
                        # (3) save model_ckpt
                        if g_step % step_save_ckpt == 0: # step_save_ckpt
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    def inference_v3(rgb_input, rgb_target, rgb_input_last):
                        # ======== forward ============================================= #
                        rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                        #
                        pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                           rgb_G_output.unsqueeze(2)], 2)
                        gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                         rgb_target.unsqueeze(2)], 2)
                        flow_pred = (flow_network((pred_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        flow_gt = (flow_network((gt_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        # loss is used to optimize G
                        d_gen = discriminator(rgb_G_output)
                        g_loss_bp = g_loss_fn(flow_pred, flow_gt,
                                           rgb_G_output, rgb_target,
                                           latent_diff, d_gen)
                        # ======= backward ========================================== #
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            d_loss = d_loss_bp.item()
                            g_loss = g_loss_fn.g_loss
                            g_adv_loss = g_loss_fn.g_adv_loss
                            g_flow_loss = g_loss_fn.g_flow_loss
                            g_int_loss = g_loss_fn.g_int_loss
                            g_gd_loss = g_loss_fn.g_gd_loss
                            g_latent_loss = g_loss_fn.g_latent_loss  # 直接传进来
                            #
                            train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            #
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss, lam_lp, g_int_loss * lam_lp) + \
                                '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                                '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                                '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_latent_loss, lam_latent, g_latent_loss * lam_latent) + \
                                '                 PSNR  Error      : {}\n'.format(train_psnr)
                            logger.info(log_info)
                        #
                        if g_step % step_summ == 0:
                            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                            writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_latent_loss', g_latent_loss, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                "rgb", vis_sample_size)
                            writer.add_image('image/train_rgb_output_target',
                                             vis_rgb, global_step=g_step)
                        #
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    def inference_v4(rgb_input, rgb_target, rgb_input_last):
                        # "op_int_adv" TODO 传参就要改，加 adv
                        # ======== forward ======================================================= #
                        #
                        rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                        d_gen = discriminator(rgb_G_output)  # loss is used to optimize G
                        g_loss_bp = g_loss_fn(rgb_G_output, rgb_target, d_gen, latent_diff)
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        #
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        # (1) log
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            d_loss_op = d_loss_bp.item()
                            g_loss_op = g_loss_fn.g_loss_op
                            g_int_loss_op = g_loss_fn.g_int_loss_op
                            g_adv_loss_op = g_loss_fn.g_adv_loss_op
                            g_latent_loss = g_loss_fn.g_latent_loss # 直接传进来

                            epe_error = utils.epe_error(rgb_G_output, rgb_target)
                            # op value 是 unbounded的，不是 (x,1), 所以 直接用 psnr 不合适
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            ##
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss_op, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss_op) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss_op, lam_adv_op, g_adv_loss_op * lam_adv_op) + \
                                '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_latent_loss, lam_latent, g_latent_loss * lam_latent) + \
                                '                 EPE  Error      : {}\n'.format(epe_error)
                            logger.info(log_info)
                        # (2) summ
                        if g_step % step_summ == 0:
                            writer.add_scalar('epe/train_epe_error', epe_error, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss_op', g_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss_op', g_int_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss_op', g_adv_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_latent_loss', g_latent_loss, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_op_output_target',
                                             vis_rgb, global_step=g_step)
                        # (3) save model_ckpt
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)

                    def inference_v4_1(rgb_input, rgb_target, rgb_input_last):
                        #
                        rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                        g_loss = g_loss_fn(
                            rgb_G_output, rgb_target,
                            latent_diff)

                        # ======= backward ========================================== #
                        # ------- (2) update optim_G --------------
                        optimizer_G.zero_grad()
                        g_loss.backward()
                        optimizer_G.step()

                        # -------- lr decay -------------------------
                        scheduler_G.step()

                        # ========== log training state ======================== #
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            g_loss_op = g_loss_fn.g_loss_op
                            g_int_loss_op = g_loss_fn.g_int_loss_op
                            g_latent_loss_op = g_loss_fn.g_latent_loss_op  # 直接传进来
                            #
                            train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            #
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss_op) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                                '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_latent_loss_op, lam_latent, g_latent_loss_op * lam_latent) + \
                                '                 PSNR  Error      : {}\n'.format(train_psnr)
                            logger.info(log_info)

                        if g_step % step_summ == 0:
                            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss', g_int_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_latent_loss', g_latent_loss_op, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_op = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_op_output_target',
                                             vis_op, global_step=g_step)
                        #
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                    #
                    infer_fn_loss_map = {
                        "rgb_int_gdl_flow_adv": inference_v1,
                        "op_int_adv": inference_v2,
                        "rgb_int_gdl_flow_adv_vq": inference_v3,
                        "op_int_adv_vq": inference_v4,
                        # "twostream": inference_v5,
                        # "twostream_vq": inference_v6,
                    }
                    loss_tag = self.params.loss_tag
                    infer_fn = infer_fn_loss_map[loss_tag]
                    #
                    infer_fn(rgb_input, rgb_target, rgb_input_last)
                    #
                    if g_step == iterations:
                        break
                    #
                    pre_time_data = time.time()  # 更新为 pre
                    pre_time = time.time()
        t_delta = time.time() - t_start
        fps = t_delta / (g_step * batch_size)  # 近似，因为 batch 最后一个不是满的
        # 出 train_report: gen_train_report
        logger.info("training fps = {}".format(fps))
        logger.info("training complete! ")

class train_single_Helper(object):
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
        self.loss = loss  # 仅仅只有 g_loss 无 d_loss
        self.optimizer = optimizer
        self.params = params # 直接传 const

    def train_rgb(self):
        #
        # params collect
        # device = self.params.device
        # torch.cuda.set_device(0)
        batch_size = self.params.batch_size
        num_workers = self.params.num_workers
        iterations = self.params.iterations
        #
        step_log = self.params.step_log
        step_summ = self.params.step_summ
        step_save_ckpt = self.params.step_save_ckpt
        #
        pretrain = self.params.pretrain
        model_generator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "generator"))
        model_discriminator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt,  "discriminator"))
        #
        flow_model_path = self.params.flow_model_path
        #
        summary_dir = self.params.summary_dir
        sample_size = self.params.sample_size
        logger = self.params.logger
        #
        lr_g = self.params.lr_g
        lr_d = self.params.lr_d
        lam_adv = self.params.lam_adv
        lam_gdl = self.params.lam_gdl
        lam_flow = self.params.lam_flow
        lam_lp = self.params.lam_lp
        lam_latent = self.params.lam_latent
        #
        # model
        generator = self.model.generator
        discriminator = self.model.discriminator
        flow_network = self.model.flow_network
        #
        dataset_loader = DataLoader(self.dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True) # pin_memory
        #
        # optimizer & scheduler
        optimizer_D = self.optimizer.optimizer_D
        optimizer_G = self.optimizer.optimizer_G
        scheduler_D = self.optimizer.scheduler_D # lr_decay
        scheduler_G = self.optimizer.scheduler_G
        #
        g_loss_fn = self.loss.g_loss
        d_loss_fn = self.loss.d_loss
        # model init (要么从0init, 要么pretrain)
        generator, discriminator, g_step = \
            utils.init_model(generator, discriminator,
                       generator_model_dir=model_generator_save_dir,
                       discriminator_model_dir=model_discriminator_save_dir,
                       logger=logger)
        flow_network.load_state_dict(torch.load(flow_model_path)['state_dict'])
        #
        generator = generator.cuda().train()
        discriminator = discriminator.cuda().train()
        flow_network = flow_network.cuda().eval() # 注意是 eval()
        #
        # train
        pre_time = time.time()
        pre_time_data = time.time()
        with SummaryWriter(summary_dir) as writer:
            while g_step < iterations:
                for sample in dataset_loader: # (b,t,c,h,w)
                    #
                    g_step += 1
                    cost_time_data = time.time() - pre_time_data

                    # print(g_step)
                    rgb = sample.cuda()
                    rgb_input = rgb[:, :-1, :, :, :]
                    rgb_target= rgb[:, -1, :, :, :]
                    rgb_input_last = rgb[:, -1, :, :, :]
                    # (b,t*c,h,w)
                    rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                       rgb_input.shape[-2], rgb_input.shape[-1])
                    #
                    # ======== forward ============================================= #
                    # print("============= rgb_input: ", rgb_input.size(),
                    #       rgb_input.min(), rgb_input.max())
                    rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                    #
                    pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                       rgb_G_output.unsqueeze(2)], 2)
                    gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                     rgb_target.unsqueeze(2)], 2)
                    # flownet not update
                    flow_pred = (flow_network(pred_flow_esti_tensor * 255.0) / 255.0).detach()
                    flow_gt = (flow_network(gt_flow_esti_tensor * 255.0) / 255.0).detach()

                    # loss is used to optimize G
                    d_gen = discriminator(rgb_G_output)

                    g_loss = g_loss_fn(flow_pred, flow_gt,
                               rgb_G_output, rgb_target,
                               latent_diff, d_gen)

                    # ======= backward ========================================== #
                    #
                    # -------- (1) update optim_D -------
                    # 注意detach
                    d_real, d_gen = discriminator(rgb_target), \
                                    discriminator(rgb_G_output.detach())
                    d_loss = d_loss_fn(d_real, d_gen) #单独处理
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
                    if g_step % step_log == 0:
                        # --------  cal psnr,loss (log info) --------------
                        d_loss = d_loss.item()
                        g_loss = g_loss_fn.g_loss
                        g_adv_loss = g_loss_fn.g_adv_loss
                        g_flow_loss = g_loss_fn.g_flow_loss
                        g_int_loss = g_loss_fn.g_int_loss
                        g_gd_loss = g_loss_fn.g_gd_loss
                        g_latent_loss = g_loss_fn.g_latent_loss  # 直接传进来
                        #
                        train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                        #
                        lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                        lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                        #
                        cost_time = time.time() - pre_time
                        #
                        log_info = \
                            'time of cur_step :          {:.2f}                           \n'.format(
                                                    cost_time) + \
                            'time of data_load:          {:.2f}                            \n'.format(
                                                    cost_time_data) + \
                            'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                                    g_step, d_loss, lr_d) + \
                            'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                            '                 Global      Loss : {}\n'.format(g_loss) + \
                            '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_int_loss, lam_lp,g_int_loss * lam_lp) + \
                            '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                            '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                            '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                            '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_latent_loss, lam_latent, g_latent_loss * lam_latent) + \
                            '                 PSNR  Error      : {}\n'.format(train_psnr)
                        logger.info(log_info)

                    if g_step % step_summ == 0:
                        writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                        writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                        writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_latent_loss', g_latent_loss, global_step=g_step)
                        #
                        if sample_size > rgb_G_output.size()[0]:
                            sample_size = rgb_G_output.size()[0]
                        vis_rgb = utils.get_vis_tensor(torch.cat(
                            [rgb_G_output[:sample_size], rgb_target[:sample_size]], 0),
                            "rgb", sample_size)
                        writer.add_image('image/train_rgb_output_target',
                                         vis_rgb, global_step=g_step)
                        #
                        # writer.add_images('image/train_rgb_output_target', rgb_target[:sample_size], global_step=g_step)
                        # writer.add_images('image/train__rgb', rgb_G_output[:sample_size], global_step=g_step)
                        #
                        # op_target = utils.utils.get_vis_tensor(op_target[:sample_size], "op", sample_size)
                        # op_G_output = utils.utils.get_vis_tensor(op_G_output[:sample_size], "op", sample_size)
                        # writer.add_image('image/train_target_op', op_target, global_step=g_step)
                        # writer.add_image('image/train_output_op', op_G_output, global_step=g_step)

                    if g_step % step_save_ckpt == 0:
                        utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                        utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    if g_step == iterations:
                        break

                    #
                    pre_time_data = time.time()  # 更新为 pre
                    pre_time = time.time()
        # t_delta = time.time() - t_start
        # fps = t_delta / (g_step * batch_size)  # 近似，因为 batch 最后一个不是满的
        # 出 train_report: gen_train_report
        # logger.info("fps = ".format(fps))
        logger.info("training complete! ")

    def train_op(self):
        #
        # params collect
        # device = torch.device("cuda:{}".format(self.params.gpu))
        batch_size = self.params.batch_size
        num_workers = self.params.num_workers
        #
        step_log = self.params.step_log
        step_summ = self.params.step_summ
        step_save_ckpt = self.params.step_save_ckpt
        #
        #
        model_generator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "generator"))
        pretrain = self.params.pretrain
        summary_dir = self.params.summary_dir
        iterations = self.params.iterations
        logger = self.params.logger
        sample_size = self.params.sample_size
        #
        lr_g = self.params.lr_g
        lam_lp_op = self.params.lam_lp_op
        lam_latent = self.params.lam_latent
        #
        # model
        generator = self.model.generator
        #
        dataset_loader = DataLoader(self.dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True)
        #
        # optimizer & scheduler
        optimizer_G = self.optimizer.optimizer_G
        scheduler_G = self.optimizer.scheduler_G
        #
        g_loss_fn = self.loss.g_loss
        # model init (要么从0init, 要么pretrain)
        generator, discriminator, g_step = \
            utils.init_model(generator, None,
                       generator_model_dir=model_generator_save_dir,
                       discriminator_model_dir=None,
                       logger=logger)
        #
        generator = generator.cuda().train()
        #
        # train
        pre_time = time.time()
        pre_time_data = time.time()
        with SummaryWriter(summary_dir) as writer:
            while g_step < iterations:
                for sample in dataset_loader: # (b,t,c,h,w)
                    #
                    g_step += 1
                    cost_time_data = time.time() - pre_time_data
                    pre_time_data = time.time()
                    # print(g_step)
                    rgb = sample.cuda()
                    rgb_input = rgb[:, :-1, :, :, :]
                    rgb_target= rgb[:, -1, :, :, :]
                    rgb_input_last = rgb[:, -1, :, :, :]
                    # (b,t*c,h,w)
                    rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                       rgb_input.shape[-2], rgb_input.shape[-1])
                    #
                    # ======== forward ============================================= #
                    # print("============= rgb_input: ", rgb_input.size(),
                    #       rgb_input.min(), rgb_input.max())
                    rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                    #
                    g_loss = g_loss_fn(
                               rgb_G_output, rgb_target,
                               latent_diff)

                    # ======= backward ========================================== #
                    # ------- (2) update optim_G --------------
                    optimizer_G.zero_grad()
                    g_loss.backward()
                    optimizer_G.step()

                    # -------- lr decay -------------------------
                    scheduler_G.step()

                    # ========== log training state ======================== #
                    if g_step % step_log == 0:
                        # --------  cal psnr,loss (log info) --------------

                        g_loss_op = g_loss_fn.g_loss_op
                        g_int_loss_op = g_loss_fn.g_int_loss_op
                        g_latent_loss_op = g_loss_fn.g_latent_loss_op  # 直接传进来
                        #
                        train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                        #
                        lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                        # lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                        #
                        cost_time = time.time() - pre_time
                        pre_time = time.time()
                        # # print("g_latent_loss_op, lam_latent: ",g_latent_loss_op, lam_latent,
                        # #       type(g_latent_loss_op), type(lam_latent))
                        # print("g_int_loss_op, lam_lp: ", g_int_loss_op, lam_lp,
                        #       type(g_int_loss_op), type(lam_lp))
                        log_info = \
                            'time of cur_step :          {:.2f}                           \n'.format(
                                cost_time) + \
                            'time of data_load:          {:.2f}                            \n'.format(
                                cost_time_data) + \
                            'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                            '                 Global      Loss : {}\n'.format(g_loss_op) + \
                            '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                            '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                                    g_latent_loss_op, lam_latent, g_latent_loss_op * lam_latent) + \
                            '                 PSNR  Error      : {}\n'.format(train_psnr)
                        logger.info(log_info)

                    if g_step % step_summ == 0:
                        writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                        writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                        writer.add_scalar('g_loss/g_int_loss', g_int_loss_op, global_step=g_step)
                        writer.add_scalar('g_loss/g_latent_loss', g_latent_loss_op, global_step=g_step)
                        #
                        if sample_size > rgb_G_output.size()[0]:
                            sample_size = rgb_G_output.size()[0]
                        #
                        vis_op = utils.get_vis_tensor(torch.cat(
                            [rgb_G_output[:sample_size], rgb_target[:sample_size]], 0),
                            "op", sample_size)
                        writer.add_image('image/train_op_output_target',
                                         vis_op, global_step=g_step)

                    if g_step % step_save_ckpt == 0:
                        utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                    #
                    if g_step == iterations:
                        break

        # t_delta = time.time() - t_start
        # fps = t_delta / (g_step * batch_size)  # 近似，因为 batch 最后一个不是满的
        # 出 train_report: gen_train_report
        # logger.info("fps = ".format(fps))
        logger.info("training complete! ")
    # utils for train

    #
    # 仅考虑 single branch, rgb_and_op
    def train_base(self):
        # ======== result save ==================================================== #
        # 与具体 network 无关
        step_log = self.params.step_log
        step_summ = self.params.step_summ
        step_save_ckpt = self.params.step_save_ckpt
        summary_dir = self.params.summary_dir
        vis_sample_size = self.params.vis_sample_size
        logger = self.params.logger
        # ======= data ===================================================== #
        # 与具体 network 无关
        batch_size = self.params.batch_size
        num_workers = self.params.num_workers
        dataset_loader = DataLoader(self.dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    pin_memory=True)  # pin_memory
        # ======= model ===========================================================#
        # 与具体 network 无关，但 rgb 与 op 有区别
        pretrain = self.params.pretrain
        model_generator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt, "generator"))
        model_discriminator_save_dir = utils.get_dir(os.path.join(
            self.params.train_save_ckpt,  "discriminator"))
        flow_model_path = self.params.flow_model_path
        #
        generator = self.model.generator
        discriminator = self.model.discriminator # op => none
        flow_network = self.model.flow_network # op => none
        # model init (要么从0 init, 要么pretrain)
        generator, discriminator, g_step = \
            utils.init_model(generator, discriminator,
                             generator_model_dir=model_generator_save_dir,
                             discriminator_model_dir=model_discriminator_save_dir,
                             logger=logger) # D 会做none过滤
        generator = generator.cuda().train()
        discriminator = discriminator.cuda().train() # op 也需要 D (pixel_D)
        data_type = self.params.data_type
        if data_type == "rgb":  # op branch 的 D 初始化为 None
            flow_network.load_state_dict(torch.load(flow_model_path)['state_dict'])
            flow_network = flow_network.cuda().eval()  # 注意是 eval()
        # ======== loss, optimizer, scheduler ==================================== #
        # 与network 有关，rgb 与 op 有区别
        # loss (根据 loss_tag 调度)
        g_loss_fn = self.loss.g_loss
        d_loss_fn = self.loss.d_loss
        iterations = self.params.iterations
        lam_adv = self.params.lam_adv
        lam_gdl = self.params.lam_gdl
        lam_flow = self.params.lam_flow
        lam_lp = self.params.lam_lp
        lam_latent = self.params.lam_latent
        # for op
        lam_lp_op = self.params.lam_lp_op
        lam_adv_op = self.params.lam_adv_op
        # optimizer
        optimizer_D = self.optimizer.optimizer_D
        optimizer_G = self.optimizer.optimizer_G
        scheduler_D = self.optimizer.scheduler_D # lr_decay
        scheduler_G = self.optimizer.scheduler_G
        # TODO twostream 可能需要针对不同 part 采样不同 lr and lr_decay and optimizer
        # 上面所有直接赋值的不用 if 隔离分类，因为初始化有值
        #
        # ======== train ========================================================== #
        t_start = time.time()
        pre_time = time.time()
        pre_time_data = time.time()
        with SummaryWriter(summary_dir) as writer:
            while g_step < iterations:
                for sample in dataset_loader: # (b,t,c,h,w)
                    #
                    g_step += 1
                    cost_time_data = time.time() - pre_time_data
                    #
                    # 根据不同情况调度 不同 input
                    rgb = sample.cuda()
                    # print("range is:", rgb.min(), rgb.max())
                    rgb_input = rgb[:, :-1, :, :, :]
                    rgb_target = rgb[:, -1, :, :, :]
                    rgb_input_last = rgb[:, -1, :, :, :]
                    # (b,t*c,h,w)
                    rgb_input = rgb_input.view(rgb_input.shape[0], -1,
                                               rgb_input.shape[-2], rgb_input.shape[-1])

                    # TODO:(1) v2,v4 还有根据v1改，(2) v1~v4 提取公共部分
                    # 根据 loss_tag 来拆分出不同子函数
                    def inference_v1(rgb_input, rgb_target, rgb_input_last):
                        # "rgb_int_gdl_flow_adv"
                        # ======== forward ======================================================= #
                        #
                        rgb_G_output = generator(rgb_input)
                        #
                        pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                           rgb_G_output.unsqueeze(2)], 2)
                        gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                         rgb_target.unsqueeze(2)], 2)
                        # flownet not update, 所以 detach
                        # Input for flownet2sd is in (0, 255). 而此处 tensor is in (-1,1)
                        flow_pred = (flow_network((pred_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        flow_gt = (flow_network((gt_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        #
                        # pred_flow_esti_tensor = torch.cat([
                        #     rgb_input_last.view(-1,3,1,rgb_input_last.shape[-2],rgb_input_last.shape[-1]),
                        #     rgb_G_output.view(-1,3,1,rgb_G_output.shape[-2],rgb_G_output.shape[-1])],
                        #     2)
                        # gt_flow_esti_tensor = torch.cat([
                        #     rgb_input_last.view(-1,3,1,rgb_input_last.shape[-2],rgb_input_last.shape[-1]),
                        #     rgb_target.view(-1,3,1,rgb_target.shape[-2],rgb_target.shape[-1])],
                        #     2)
                        # flow_gt=flow_network(gt_flow_esti_tensor*255.0)
                        # flow_pred=flow_network(pred_flow_esti_tensor*255.0)
                        #
                        d_gen = discriminator(rgb_G_output) # loss is used to optimize G
                        #
                        g_loss_bp = g_loss_fn(flow_pred, flow_gt,
                                           rgb_G_output, rgb_target,d_gen)
                        # ======= backward ======================================================== #
                        #
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        #
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        # (1) log
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            d_loss = d_loss_bp.item()
                            g_loss = g_loss_fn.g_loss
                            g_adv_loss = g_loss_fn.g_adv_loss
                            g_int_loss = g_loss_fn.g_int_loss
                            g_flow_loss = g_loss_fn.g_flow_loss
                            g_gd_loss = g_loss_fn.g_gd_loss
                            train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            #
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss, lam_lp, g_int_loss * lam_lp) + \
                                '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                                '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                                '                 PSNR  Error      : {}\n'.format(train_psnr)
                            logger.info(log_info)
                        # (2) summ
                        if g_step % step_summ == 0:
                            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                            writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_rgb_output_target',
                                             vis_rgb, global_step=g_step)
                        # (3) save model_ckpt
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    def inference_v2(rgb_input, rgb_target, rgb_input_last):
                        # "op_int_adv" TODO 传参就要改，加 adv
                        # ======== forward ======================================================= #
                        #
                        rgb_G_output = generator(rgb_input)
                        d_gen = discriminator(rgb_G_output)  # loss is used to optimize G
                        g_loss_bp = g_loss_fn(rgb_G_output, rgb_target, d_gen)
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        #
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        # (1) log
                        if g_step % step_log == 0: # step_log
                            # --------  cal psnr,loss (log info) --------------
                            d_loss_op = d_loss_bp.item()
                            g_loss_op = g_loss_fn.g_loss_op
                            g_int_loss_op = g_loss_fn.g_int_loss_op
                            g_adv_loss_op = g_loss_fn.g_adv_loss_op

                            epe_error = utils.epe_error(rgb_G_output, rgb_target)
                            # op value 是 unbounded的，不是 (x,1), 所以 直接用 psnr 不合适
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            ##
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss_op, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss_op) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss_op, lam_adv_op, g_adv_loss_op * lam_adv_op) + \
                                '                 EPE  Error      : {}\n'.format(epe_error)
                            logger.info(log_info)
                        # (2) summ
                        if g_step % step_summ == 0: # step_summ
                            writer.add_scalar('epe/train_epe_error', epe_error, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss_op', g_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss_op', g_int_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss_op', g_adv_loss_op, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_op_output_target',
                                             vis_rgb, global_step=g_step)
                        # (3) save model_ckpt
                        if g_step % step_save_ckpt == 0: # step_save_ckpt
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    def inference_v3(rgb_input, rgb_target, rgb_input_last):
                        # ======== forward ============================================= #
                        rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                        #
                        pred_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                           rgb_G_output.unsqueeze(2)], 2)
                        gt_flow_esti_tensor = torch.cat([rgb_input_last.unsqueeze(2),
                                                         rgb_target.unsqueeze(2)], 2)
                        flow_pred = (flow_network((pred_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        flow_gt = (flow_network((gt_flow_esti_tensor * 0.5 + 0.5) * 255.0) / 255.0).detach()
                        # loss is used to optimize G
                        d_gen = discriminator(rgb_G_output)
                        g_loss_bp = g_loss_fn(flow_pred, flow_gt,
                                           rgb_G_output, rgb_target,
                                           latent_diff, d_gen)
                        # ======= backward ========================================== #
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            d_loss = d_loss_bp.item()
                            g_loss = g_loss_fn.g_loss
                            g_adv_loss = g_loss_fn.g_adv_loss
                            g_flow_loss = g_loss_fn.g_flow_loss
                            g_int_loss = g_loss_fn.g_int_loss
                            g_gd_loss = g_loss_fn.g_gd_loss
                            g_latent_loss = g_loss_fn.g_latent_loss  # 直接传进来
                            #
                            train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            #
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss, lam_lp, g_int_loss * lam_lp) + \
                                '                 gradient    Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_gd_loss, lam_gdl, g_gd_loss * lam_gdl) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss, lam_adv, g_adv_loss * lam_adv) + \
                                '                 flownet     Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_flow_loss, lam_flow, g_flow_loss * lam_flow) + \
                                '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_latent_loss, lam_latent, g_latent_loss * lam_latent) + \
                                '                 PSNR  Error      : {}\n'.format(train_psnr)
                            logger.info(log_info)
                        #
                        if g_step % step_summ == 0:
                            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                            writer.add_scalar('total_loss/d_loss', d_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss', g_adv_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_flow_loss', g_flow_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss', g_int_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_gd_loss', g_gd_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_latent_loss', g_latent_loss, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                "rgb", vis_sample_size)
                            writer.add_image('image/train_rgb_output_target',
                                             vis_rgb, global_step=g_step)
                        #
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)
                    #
                    def inference_v4(rgb_input, rgb_target, rgb_input_last):
                        # "op_int_adv" TODO 传参就要改，加 adv
                        # ======== forward ======================================================= #
                        #
                        rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                        d_gen = discriminator(rgb_G_output)  # loss is used to optimize G
                        g_loss_bp = g_loss_fn(rgb_G_output, rgb_target, d_gen, latent_diff)
                        # ------- (1) update optim_D --------------
                        # When training discriminator, don't train generator,
                        # so use .detach() to cut off gradients.
                        d_real, d_gen = discriminator(rgb_target), \
                                        discriminator(rgb_G_output.detach())
                        d_loss_bp = d_loss_fn(d_real, d_gen)  # 单独处理
                        optimizer_D.zero_grad()
                        d_loss_bp.backward()
                        optimizer_D.step()
                        # -------- (2) update optim_G -------
                        optimizer_G.zero_grad()
                        g_loss_bp.backward()
                        optimizer_G.step()
                        #
                        # -------- lr decay -------------------------
                        scheduler_D.step()
                        scheduler_G.step()
                        # ========== log training state ======================== #
                        # (1) log
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            d_loss_op = d_loss_bp.item()
                            g_loss_op = g_loss_fn.g_loss_op
                            g_int_loss_op = g_loss_fn.g_int_loss_op
                            g_adv_loss_op = g_loss_fn.g_adv_loss_op
                            g_latent_loss = g_loss_fn.g_latent_loss # 直接传进来

                            epe_error = utils.epe_error(rgb_G_output, rgb_target)
                            # op value 是 unbounded的，不是 (x,1), 所以 直接用 psnr 不合适
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            lr_d = optimizer_D.state_dict()['param_groups'][0]['lr']
                            ##
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'DiscriminatorModel: Step {} | Global Loss: {:.6f}, lr = {:.6f}\n'.format(
                                    g_step, d_loss_op, lr_d) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss_op) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                                '                 adversarial Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_adv_loss_op, lam_adv_op, g_adv_loss_op * lam_adv_op) + \
                                '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_latent_loss, lam_latent, g_latent_loss * lam_latent) + \
                                '                 EPE  Error      : {}\n'.format(epe_error)
                            logger.info(log_info)
                        # (2) summ
                        if g_step % step_summ == 0:
                            writer.add_scalar('epe/train_epe_error', epe_error, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss_op', g_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss_op', g_int_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_adv_loss_op', g_adv_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_latent_loss', g_latent_loss, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_rgb = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_op_output_target',
                                             vis_rgb, global_step=g_step)
                        # (3) save model_ckpt
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                            utils.saver(discriminator.state_dict(), model_discriminator_save_dir, g_step, logger)

                    def inference_v4_1(rgb_input, rgb_target, rgb_input_last):
                        #
                        rgb_G_output, latent_diff, embed_ind_tuple = generator(rgb_input)
                        g_loss = g_loss_fn(
                            rgb_G_output, rgb_target,
                            latent_diff)

                        # ======= backward ========================================== #
                        # ------- (2) update optim_G --------------
                        optimizer_G.zero_grad()
                        g_loss.backward()
                        optimizer_G.step()

                        # -------- lr decay -------------------------
                        scheduler_G.step()

                        # ========== log training state ======================== #
                        if g_step % step_log == 0:
                            # --------  cal psnr,loss (log info) --------------
                            g_loss_op = g_loss_fn.g_loss_op
                            g_int_loss_op = g_loss_fn.g_int_loss_op
                            g_latent_loss_op = g_loss_fn.g_latent_loss_op  # 直接传进来
                            #
                            train_psnr = utils.psnr_error(rgb_G_output, rgb_target)
                            #
                            lr_g = optimizer_G.state_dict()['param_groups'][0]['lr']
                            #
                            cost_time = time.time() - pre_time
                            #
                            log_info = \
                                'time of cur_step :          {:.2f}                           \n'.format(
                                    cost_time) + \
                                'time of data_load:          {:.2f}                            \n'.format(
                                    cost_time_data) + \
                                'GeneratorModel : Step {}, lr = {:.6f}\n'.format(g_step, lr_g) + \
                                '                 Global      Loss : {}\n'.format(g_loss_op) + \
                                '                 intensity   Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_int_loss_op, lam_lp_op, g_int_loss_op * lam_lp_op) + \
                                '                 latent      Loss : ({:.4f} * {:.4f} = {:.4f})\n'.format(
                                    g_latent_loss_op, lam_latent, g_latent_loss_op * lam_latent) + \
                                '                 PSNR  Error      : {}\n'.format(train_psnr)
                            logger.info(log_info)

                        if g_step % step_summ == 0:
                            writer.add_scalar('psnr/train_psnr', train_psnr, global_step=g_step)
                            writer.add_scalar('total_loss/g_loss', g_loss, global_step=g_step)
                            writer.add_scalar('g_loss/g_int_loss', g_int_loss_op, global_step=g_step)
                            writer.add_scalar('g_loss/g_latent_loss', g_latent_loss_op, global_step=g_step)
                            #
                            vis_sample_size = self.params.vis_sample_size
                            if vis_sample_size > rgb_G_output.size()[0]:
                                vis_sample_size = rgb_G_output.size()[0]
                            vis_op = utils.get_vis_tensor(torch.cat(
                                [rgb_G_output[:vis_sample_size], rgb_target[:vis_sample_size]], 0),
                                data_type, vis_sample_size)
                            writer.add_image('image/train_op_output_target',
                                             vis_op, global_step=g_step)
                        #
                        if g_step % step_save_ckpt == 0:
                            utils.saver(generator.state_dict(), model_generator_save_dir, g_step, logger)
                    #
                    infer_fn_loss_map = {
                        "rgb_int_gdl_flow_adv": inference_v1,
                        "op_int_adv": inference_v2,
                        "rgb_int_gdl_flow_adv_vq": inference_v3,
                        "op_int_adv_vq": inference_v4,
                        # "twostream": inference_v5,
                        # "twostream_vq": inference_v6,
                    }
                    loss_tag = self.params.loss_tag
                    infer_fn = infer_fn_loss_map[loss_tag]
                    #
                    infer_fn(rgb_input, rgb_target, rgb_input_last)
                    #
                    if g_step == iterations:
                        break
                    #
                    pre_time_data = time.time()  # 更新为 pre
                    pre_time = time.time()
        t_delta = time.time() - t_start
        fps = t_delta / (g_step * batch_size)  # 近似，因为 batch 最后一个不是满的
        # 出 train_report: gen_train_report
        logger.info("training fps = {}".format(fps))
        logger.info("training complete! ")