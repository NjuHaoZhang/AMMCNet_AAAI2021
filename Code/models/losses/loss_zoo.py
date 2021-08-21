import torch
import torch.nn as nn

from .losses_utils import (
    Flow_Loss, Intensity_Loss, Gradient_Loss,
    Adversarial_Loss, Discriminate_Loss, MultiScale,
    smooth_l1_loss,
)


class base_Loss(nn.Module):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def __init__(self, lam_adv=None, lam_gdl=None, lam_flow=None, lam_lp=None,
            lam_latent=None, lam_lp_op=None, lam_adv_op=None):
        # type: (int, float) -> None
        """
        Class constructor.
        :param cpd_channels: number of bins in which the multinomial works.
        :param lam: weight of the autoregression loss.
        """
        super(base_Loss, self).__init__()

        self.lam_lp = lam_lp
        self.lam_adv = lam_adv
        self.lam_gdl = lam_gdl
        self.lam_flow = lam_flow
        #
        self.lam_latent = lam_latent
        #
        self.lam_lp_op = lam_lp_op
        self.lam_adv_op = lam_adv_op

        self.adversarial_loss_fn = Adversarial_Loss()
        self.flow_loss_fn = Flow_Loss()
        self.int_loss_fn = Intensity_Loss()
        self.gd_loss_fn = Gradient_Loss() # or use gdl_loss by haozhang
        # self.discriminate_loss_fn = Discriminate_Loss()
        #
        # self.int_loss_fn_op = smooth_l1_loss #
        self.int_loss_fn_op = Intensity_Loss() #
        #
        self.adversarial_loss_fn_op = Adversarial_Loss() #
        # self.discriminate_loss_fn_op = Discriminate_Loss()

        # Numerical variables
        self.g_adv_loss = None
        self.g_flow_loss = None
        self.g_int_loss = None
        self.g_gd_loss = None
        #
        self.g_int_loss_op = None
        self.g_adv_loss_op = None
        #
        self.g_latent_loss = None #

    def forward(self, flow_pred, flow_gt, rgb_G_output, rgb_target,
                op_G_output, op_target, latent_diff, d_gen):
        pass


class rgb_Loss(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """

    def forward(self, flow_pred, flow_gt, rgb_G_output, rgb_target,
                d_gen):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        g_adv_loss = self.adversarial_loss_fn(d_gen)
        g_flow_loss = self.flow_loss_fn(flow_pred, flow_gt)
        g_int_loss = self.int_loss_fn(rgb_G_output, rgb_target)
        g_gd_loss = self.gd_loss_fn(rgb_G_output, rgb_target)

        g_loss = self.lam_adv * g_adv_loss + self.lam_gdl * g_gd_loss + \
                 self.lam_flow * g_flow_loss + self.lam_lp * g_int_loss
        # d_loss = self.discriminate_loss_fn(d_real, d_gen)

        # Store numerical
        self.g_loss = g_loss.item()
        # self.d_loss = d_loss.item()
        self.g_adv_loss = g_adv_loss.item()
        self.g_flow_loss = g_flow_loss.item()
        self.g_int_loss = g_int_loss.item()
        self.g_gd_loss = g_gd_loss.item()

        return g_loss


class rgb_vq_Loss(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def forward(self, flow_pred, flow_gt, rgb_G_output, rgb_target,
                latent_diff, d_gen):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        g_adv_loss = self.adversarial_loss_fn(d_gen)
        g_flow_loss = self.flow_loss_fn(flow_pred, flow_gt)
        g_int_loss = self.int_loss_fn(rgb_G_output, rgb_target)
        g_gd_loss = self.gd_loss_fn(rgb_G_output, rgb_target)
        #
        g_latent_loss = latent_diff #
        #
        g_loss = self.lam_adv * g_adv_loss + self.lam_gdl * g_gd_loss + \
                 self.lam_flow * g_flow_loss + self.lam_lp * g_int_loss + \
                 self.lam_latent * g_latent_loss
        #
        # d_loss = self.discriminate_loss_fn(d_real, d_gen)

        # Store numerical
        self.g_loss = g_loss.item()
        # self.d_loss = d_loss.item()
        self.g_adv_loss = g_adv_loss.item()
        self.g_flow_loss = g_flow_loss.item()
        self.g_int_loss = g_int_loss.item()
        self.g_gd_loss = g_gd_loss.item()
        self.g_latent_loss = g_latent_loss.item()

        return g_loss

#
class op_loss(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """

    def forward(self, op_G_output, op_target, d_gen):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        g_int_loss_op = self.int_loss_fn_op(op_G_output, op_target)
        g_adv_loss_op = self.adversarial_loss_fn_op(d_gen)
        g_loss_op = self.lam_lp_op * g_int_loss_op + \
                    self.lam_adv_op * g_adv_loss_op
        #
        # Store numerical
        self.g_loss_op = g_loss_op.item()
        self.g_int_loss_op = g_int_loss_op.item()
        self.g_adv_loss_op = g_adv_loss_op.item()

        return g_loss_op


class op_vq_Loss(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def forward(self, op_G_output, op_target, d_gen, latent_diff):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        g_int_loss_op = self.int_loss_fn_op(op_G_output, op_target)
        g_adv_loss_op = self.adversarial_loss_fn_op(d_gen)
        g_latent_loss = latent_diff  # 直接传进来
        g_loss_op = self.lam_lp_op * g_int_loss_op + \
                    self.lam_adv_op * g_adv_loss_op + \
                    self.lam_latent * g_latent_loss
        #
        # Store numerical
        self.g_loss_op = g_loss_op.item()
        self.g_int_loss_op = g_int_loss_op.item()
        self.g_adv_loss_op = g_adv_loss_op.item()
        self.g_latent_loss = g_latent_loss.item()

        return g_loss_op

#================================== #
class op_loss_v1(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """

    def forward(self, op_G_output, op_target):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        g_int_loss_op = self.int_loss_fn_op(op_G_output, op_target)
        #
        g_loss_op = self.lam_lp_op * g_int_loss_op
        #
        # d_loss = self.discriminate_loss_fn(d_real, d_gen)

        # Store numerical
        self.g_loss_op = g_loss_op.item()
        # self.d_loss = d_loss.item()
        self.g_int_loss_op = g_int_loss_op.item()

        return g_loss_op
class op_vq_Loss_v1(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def forward(self, op_G_output, op_target, latent_diff):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        g_int_loss_op = self.int_loss_fn_op(op_G_output, op_target)
        g_latent_loss_op = latent_diff #
        #
        g_loss_op = self.lam_lp_op * g_int_loss_op +\
                 self.lam_latent * g_latent_loss_op
        #
        # d_loss = self.discriminate_loss_fn(d_real, d_gen)

        # Store numerical
        self.g_loss_op = g_loss_op.item()
        self.g_int_loss_op = g_int_loss_op.item()
        self.g_latent_loss_op = g_latent_loss_op.item()

        return g_loss_op
# ================================== #

#
class Twostream_Loss(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def forward(self, flow_pred, flow_gt, rgb_G_output, rgb_target,
                op_G_output, op_target, d_gen):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        g_adv_loss = self.adversarial_loss_fn(d_gen)
        g_flow_loss = self.flow_loss_fn(flow_pred, flow_gt)
        g_int_loss = self.int_loss_fn(rgb_G_output, rgb_target)
        g_gd_loss = self.gd_loss_fn(rgb_G_output, rgb_target)
        #
        g_int_loss_op = self.int_loss_fn_op(op_G_output, op_target)
        #
        g_loss = self.lam_adv * g_adv_loss + self.lam_gdl * g_gd_loss + \
                 self.lam_flow * g_flow_loss + self.lam_lp * g_int_loss + \
                 self.lam_lp_op * g_int_loss_op
        #
        # d_loss = self.discriminate_loss_fn(d_real, d_gen)

        # Store numerical
        self.g_loss = g_loss.item()
        # self.d_loss = d_loss.item()
        self.g_adv_loss = g_adv_loss.item()
        self.g_flow_loss = g_flow_loss.item()
        self.g_int_loss = g_int_loss.item()
        self.g_gd_loss = g_gd_loss.item()
        self.g_int_loss_op = g_int_loss_op.item()

        return g_loss

#
class Twostream_vq_Loss(base_Loss):
    """
    Implements the loss of a LSA model.
    It is a sum of the reconstruction loss and the autoregression loss.
    """
    def forward(self, flow_pred, flow_gt, rgb_G_output, rgb_target,
                op_G_output, op_target, latent_diff, d_gen):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of input samples.
        :param x_r: the batch of reconstructions.
        :param z: the batch of latent representations.
        :param z_dist: the batch of estimated cpds.
        :return: the loss of the model (averaged along the batch axis).
        """
        # Compute pytorch loss
        g_adv_loss = self.adversarial_loss_fn(d_gen)
        g_flow_loss = self.flow_loss_fn(flow_pred, flow_gt)
        g_int_loss = self.int_loss_fn(rgb_G_output, rgb_target)
        g_gd_loss = self.gd_loss_fn(rgb_G_output, rgb_target)
        #
        g_int_loss_op = self.int_loss_fn_op(op_G_output, op_target)
        #
        g_latent_loss = latent_diff #
        #
        g_loss = self.lam_adv * g_adv_loss + self.lam_gdl * g_gd_loss + \
                 self.lam_flow * g_flow_loss + self.lam_lp * g_int_loss + \
                 self.lam_latent * g_latent_loss + \
                 self.lam_lp_op * g_int_loss_op
        #
        # d_loss = self.discriminate_loss_fn(d_real, d_gen)

        # Store numerical
        self.g_loss = g_loss.item()
        # self.d_loss = d_loss.item()
        self.g_adv_loss = g_adv_loss.item()
        self.g_flow_loss = g_flow_loss.item()
        self.g_int_loss = g_int_loss.item()
        self.g_gd_loss = g_gd_loss.item()
        self.g_int_loss_op = g_int_loss_op.item()
        self.g_latent_loss = g_latent_loss.item()

        return g_loss

