import torch
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR
)

class OPtimizer(object):
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

def get_optimizer(generator, discriminator, const):

    if const.mode == "testing":
        return None

    step_decay_g = const.step_decay_g
    step_decay_d = const.step_decay_d
    lr_g = const.lr_g
    lr_d = const.lr_d

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g)
    if const.use_fixed_params: # fix TODO
        optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                  generator.parameters()), lr=lr_g)
    #
    # scheduler_G = StepLR(optimizer_G, step_size=step_decay_g, gamma=0.1)
    scheduler_G = MultiStepLR(optimizer_G, milestones=step_decay_g, gamma=0.5)

    # lr_decay
    if discriminator:
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
        # scheduler_D = StepLR(optimizer_D, step_size=step_decay_d, gamma=0.1)
        scheduler_D = MultiStepLR(optimizer_D, milestones=step_decay_d, gamma=0.5)
    else:
        optimizer_D = None
        scheduler_D = None
    # "cycle": OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataset_loader), epochs=iterations)

    optimizer = OPtimizer()
    optimizer.optimizer_G = optimizer_G
    optimizer.optimizer_D = optimizer_D
    optimizer.scheduler_G = scheduler_G
    optimizer.scheduler_D = scheduler_D

    return optimizer


# ======================================================= #

def test_lr_decay():
    pass


# ======================================================= #
if __name__ == '__main__':
    pass

