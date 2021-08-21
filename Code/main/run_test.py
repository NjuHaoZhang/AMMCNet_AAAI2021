
from ..run_helper import get_helper
from ..dataset import get_dataset
from ..models import get_model
# from ..models.losses import get_loss # for training, not for test
# from ..models.optimizer import get_optimizer
from .constant_test import const


def run_helper(const):
    #
    model = get_model(const)
    dataset = get_dataset(const)
    #
    helper = get_helper(model, dataset, const=const)
    if const.data_type == "rgb_op":
        helper.evaluate_img_pred_fea_comm_twostream()


if __name__ == "__main__":

    const.logger.info(const)
    run_helper(const)
