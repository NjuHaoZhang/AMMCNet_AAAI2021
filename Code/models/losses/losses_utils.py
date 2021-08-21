import os,time
from ...main.constant_train import const
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #
os.environ["CUDA_VISIBLE_DEVICES"]= const.gpu_idx
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow_Loss(nn.Module):
    def __init__(self):
        super(Flow_Loss,self).__init__()

    def forward(self, gen_flows,gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))

class Intensity_Loss(nn.Module):
    def __init__(self, l_num=2):
        super(Intensity_Loss,self).__init__()
        self.l_num=l_num
        self.L1 = L1()
        self.L2 = L2()
    def forward(self, gen_frames, gt_frames):
        if self.l_num == 1:
            return self.L1(gen_frames, gt_frames)
        if self.l_num == 2:
            return self.L2(gen_frames, gt_frames)


class Gradient_Loss(nn.Module):
    def __init__(self,alpha=1, channels=3):
        super(Gradient_Loss,self).__init__()
        self.alpha=alpha
        filter=torch.FloatTensor([[-1.,1.]])

        self.filter_x = filter.view(1,1,1,2).repeat(1,channels,1,1)
        self.filter_y = filter.view(1,1,2,1).repeat(1,channels,1,1)

    def forward(self, gen_frames, gt_frames):

        self.filter_x = self.filter_x.cuda()
        self.filter_y = self.filter_y.cuda()

        # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
        # neg=-1*pos
        # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
        # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
        gen_frames_x=nn.functional.pad(gen_frames,(1,0,0,0))
        gen_frames_y=nn.functional.pad(gen_frames,(0,0,1,0))
        gt_frames_x=nn.functional.pad(gt_frames,(1,0,0,0))
        gt_frames_y=nn.functional.pad(gt_frames,(0,0,1,0))

        gen_dx=nn.functional.conv2d(gen_frames_x,self.filter_x)
        gen_dy=nn.functional.conv2d(gen_frames_y,self.filter_y)
        gt_dx=nn.functional.conv2d(gt_frames_x,self.filter_x)
        gt_dy=nn.functional.conv2d(gt_frames_y,self.filter_y)

        grad_diff_x=torch.abs(gt_dx-gen_dx)
        grad_diff_y=torch.abs(gt_dy-gen_dy)

        return torch.mean(grad_diff_x**self.alpha+grad_diff_y**self.alpha)

# def Gradient_Loss(gen_frames, gt_frames, alpha=1):
#
#     def gradient(x):
#         # tf.image.image_gradients(image)
#         # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
#         # direct_ref:  https://www.jianshu.com/p/b624ff74406f
#
#         # x: (b,c,h,w), 必须是float (input params 必须指明 shape and dtype)
#         # dx, dy: (b,c,h,w)
#
#         h_x = x.size()[-2]
#         w_x = x.size()[-1]
#         # gradient step=1
#         l = x
#         r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
#         t = x
#         b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
#
#         # dx, dy = torch.abs(r - l), torch.abs(b - t) # 这个加abs影响不大
#         dx, dy = r - l, b - t # 和 tf.image.image_gradients(image) 效果完全一致，但注意：pt和tf中保持一致的dtype,非常影响结果
#         # dx will always have zeros in the last column, r-l
#         # dy will always have zeros in the last row,    b-t
#         dx[:, :, :, -1] = 0
#         dy[:, :, -1, :] = 0
#
#         return dx, dy
#
#     # gradient
#     gen_dx, gen_dy = gradient(gen_frames)
#     gt_dx, gt_dy = gradient(gt_frames)
#     #
#     grad_diff_x = torch.abs(gt_dx - gen_dx)
#     grad_diff_y = torch.abs(gt_dy - gen_dy)
#
#     # condense into one tensor and avg
#     return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2) #

class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs, fake_outputs):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)

# ======================================================================================= #

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean() # 就是 L2 loss

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, startScale = 5, numScales = 4, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale)
                            for scale in range(self.numScales)]).cuda()
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1': self.loss = L1()
        else: self.loss = L2()

        self.num_levels = 4
        output_level = 4
        self.multiScales = [nn.AvgPool2d(2**l, 2**l)
                            for l in range(self.num_levels)][::-1][:output_level]

    def forward(self, output, target):
        # if flow is normalized, every output is multiplied by its size
        # correspondingly, groundtruth should be scaled at each level
        outputs = [avg_pool(output) / 2 ** (self.num_levels - l - 1)
                   for l, avg_pool in enumerate(self.multiScales)] + [output]
        targets = [avg_pool(target) / 2 ** (self.num_levels - l - 1)
                   for l, avg_pool in enumerate(self.multiScales)] + [target]
        loss = 0
        for w, o, t in zip(self.loss_weights, outputs, targets):
            loss += w * self.loss(o, t)

        return loss #

#
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

