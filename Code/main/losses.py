import torch
import torch.nn as nn
import torch.nn.functional as F

class Flow_Loss(nn.Module):
    def __init__(self):
        super(Flow_Loss,self).__init__()

    def forward(self, gen_flows,gt_flows):

        return torch.mean(torch.abs(gen_flows - gt_flows))

class Intensity_Loss(nn.Module):
    def __init__(self,l_num):
        super(Intensity_Loss,self).__init__()
        self.l_num=l_num
    def forward(self, gen_frames,gt_frames):

        return torch.mean(torch.abs((gen_frames-gt_frames)**self.l_num))


# class Gradient_Loss(nn.Module):
#     def __init__(self,alpha,channels):
#         super(Gradient_Loss,self).__init__()
#         self.alpha=alpha
#         filter=torch.FloatTensor([[-1.,1.]])
#
#         self.filter_x = filter.view(1,1,1,2).repeat(1,channels,1,1)
#         self.filter_y = filter.view(1,1,2,1).repeat(1,channels,1,1)
#
#
#     def forward(self, gen_frames,gt_frames):
#
#         self.filter_x.to(device)
#         self.filter_y.to(device)
#
#         # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
#         # neg=-1*pos
#         # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
#         # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
#         gen_frames_x=nn.functional.pad(gen_frames,(1,0,0,0))
#         gen_frames_y=nn.functional.pad(gen_frames,(0,0,1,0))
#         gt_frames_x=nn.functional.pad(gt_frames,(1,0,0,0))
#         gt_frames_y=nn.functional.pad(gt_frames,(0,0,1,0))
#
#         gen_dx=nn.functional.conv2d(gen_frames_x,self.filter_x)
#         gen_dy=nn.functional.conv2d(gen_frames_y,self.filter_y)
#         gt_dx=nn.functional.conv2d(gt_frames_x,self.filter_x)
#         gt_dy=nn.functional.conv2d(gt_frames_y,self.filter_y)
#
#         grad_diff_x=torch.abs(gt_dx-gen_dx)
#         grad_diff_y=torch.abs(gt_dy-gen_dy)
#
#         return torch.mean(grad_diff_x**self.alpha+grad_diff_y**self.alpha)

def gradient_loss(gen_frames, gt_frames, alpha=1):

    def gradient(x):
        # tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # direct_ref:  https://www.jianshu.com/p/b624ff74406f

        # x: (b,c,h,w), 必须是float (input params 必须指明 shape and dtype)
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        l = x
        r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        t = x
        b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(r - l), torch.abs(b - t) # 这个加abs影响不大
        dx, dy = r - l, b - t # 和 tf.image.image_gradients(image) 效果完全一致，但注意：pt和tf中保持一致的dtype,非常影响结果
        # dx will always have zeros in the last column, r-l
        # dy will always have zeros in the last row,    b-t
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss,self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)

class Discriminate_Loss(nn.Module):
    def __init__(self):
        super(Discriminate_Loss,self).__init__()
    def forward(self,real_outputs,fake_outputs ):
        return torch.mean((real_outputs-1)**2/2)+torch.mean(fake_outputs**2/2)

class TotalLoss(nn.Module):

    def __init__(self):
        super(TotalLoss, self).__init__()
        pass

    def forward(self):
        pass