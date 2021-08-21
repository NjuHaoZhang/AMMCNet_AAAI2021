def get_helper(model, dataset, loss=None, optimizer=None, const=None):
    helper_tag = const.helper_tag
    mode = const.mode
    if mode == "testing":
        from .test_helper import (
            test_single_Helper,
        )
        exp_mapp = {
            "test_single": test_single_Helper,  # net & loss, for testing
            "test_twostream": test_single_Helper,
        }
    helper = exp_mapp[helper_tag](model, dataset, loss, optimizer,
                                  const)
    return helper
