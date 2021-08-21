
from .loss_zoo import (
    rgb_Loss, rgb_vq_Loss,
    op_loss, op_vq_Loss,
    Twostream_Loss, Twostream_vq_Loss,
)
from .losses_utils import (
    Discriminate_Loss,Intensity_Loss,
)


class Loss(object):
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


def get_loss(const):

    loss_tag = const.loss_tag #
    #
    loss_mapp = {
        "rgb_int_gdl_flow_adv":get_loss_v1,
        "op_int_adv": get_loss_v2,
        "rgb_int_gdl_flow_adv_vq": get_loss_v3,
        "op_int_adv_vq": get_loss_v4,

        "twostream": get_loss_v5, #
        "twostream_vq": get_loss_v6,
    }

    return loss_mapp[loss_tag](const)

#
def get_loss_v1(const):
    lam_adv = const.lam_adv
    lam_gdl = const.lam_gdl
    lam_flow = const.lam_flow
    lam_lp = const.lam_lp
    g_loss = rgb_Loss(lam_adv=lam_adv, lam_gdl=lam_gdl,
                      lam_flow=lam_flow, lam_lp=lam_lp)
    loss = Loss()
    loss.g_loss = g_loss
    loss.d_loss = Discriminate_Loss()

    return loss #

def get_loss_v2(const):
    lam_lp_op = const.lam_lp_op
    lam_adv_op = const.lam_adv_op
    g_loss = op_loss(lam_lp_op=lam_lp_op, lam_adv_op=lam_adv_op)
    #
    loss = Loss()
    loss.g_loss = g_loss
    loss.d_loss = Discriminate_Loss()

    return  loss

#
def get_loss_v3(const):
    lam_adv = const.lam_adv
    lam_gdl = const.lam_gdl
    lam_flow = const.lam_flow
    lam_lp = const.lam_lp
    lam_latent = const.lam_latent
    g_loss = rgb_vq_Loss(lam_adv=lam_adv, lam_gdl=lam_gdl,
                            lam_flow=lam_flow, lam_lp=lam_lp,
                            lam_latent=lam_latent)
    loss = Loss()
    loss.g_loss = g_loss
    loss.d_loss = Discriminate_Loss()

    return loss #

#
def get_loss_v4(const):

    lam_lp_op = const.lam_lp_op
    lam_adv_op = const.lam_adv_op
    lam_latent = const.lam_latent
    # print("lam_lp_op, lam_adv_op, lam_latent: ", lam_lp_op, lam_adv_op, lam_latent)
    g_loss = op_vq_Loss(lam_lp_op=lam_lp_op,
                        lam_adv_op = lam_adv_op,
                        lam_latent=lam_latent)
    #
    loss = Loss()
    loss.g_loss = g_loss
    loss.d_loss = Discriminate_Loss()


    return loss

#
def get_loss_v5(const):
    lam_adv = const.lam_adv
    lam_gdl = const.lam_gdl
    lam_flow = const.lam_flow
    lam_lp = const.lam_lp
    lam_lp_op = const.lam_lp_op
    g_loss = Twostream_Loss(lam_adv=lam_adv, lam_gdl=lam_gdl, lam_flow=lam_flow, lam_lp=lam_lp,
            lam_lp_op=lam_lp_op)
    #
    loss = Loss()
    loss.g_loss = g_loss
    loss.d_loss = Discriminate_Loss()

    return loss #

#
def get_loss_v6(const):
    lam_adv = const.lam_adv
    lam_gdl = const.lam_gdl
    lam_flow = const.lam_flow
    lam_lp = const.lam_lp
    lam_latent = const.lam_latent
    lam_lp_op = const.lam_lp_op
    g_loss = Twostream_vq_Loss(lam_adv=lam_adv, lam_gdl=lam_gdl, lam_flow=lam_flow, lam_lp=lam_lp,
                lam_latent=lam_latent, lam_lp_op=lam_lp_op)
    #
    loss = Loss()
    loss.g_loss = g_loss
    loss.d_loss = Discriminate_Loss()

    return loss #



