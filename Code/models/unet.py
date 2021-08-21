import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(20200525)
from torchsummaryX import summary


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channels, output_channel=3):
        super(UNet, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.tanh(x)

def _test():
    rand = torch.ones([4, 12, 256, 256]).cuda()
    t = UNet(12, 3).cuda()

    r = t(rand)
    print(r.shape)
    print(r.grad_fn)
    print(r.requires_grad)


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim) # (b,h,w,c) -> (num, c)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        #
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # (num, 1) -->[one_hot] (num, n_embed)
        # print("embed_onehot: ", embed_onehot.size())    # one_hot for training, (num, n_embed)

        #
        embed_ind = embed_ind.view(*input.shape[:-1]) # (num,) -> (b,h,w), since num==b*h*w
        # print("embed_ind.size() in view(*input.shape[:-1]): ", embed_ind.size()) # [b, h, w]
        # *input: * collects all the positional arguments in a tuple.
        # return [b, h, w]
        #
        quantize = self.embed_code(embed_ind) #
        # print("quantize in self.embed_code(embed_ind): ", quantize.size()) # [b, h, w, emb_dim]
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class enc_quan_dec(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed):
        super(enc_quan_dec, self).__init__()
        self.enc = nn.Conv2d(in_c, embed_dim, 1)
        self.quantize = Quantize(dim=embed_dim, n_embed=n_embed)
        self.dec = nn.Conv2d(embed_dim, in_c, 1)

    def forward(self,x):
        x = self.enc(x).permute(0, 2, 3, 1) # (b,c,h,w) -> (b, h, w, c)
        quantize, diff, embed_ind = self.quantize(x) # since vq_input is (b, h, w, c)
        quantize = quantize.permute(0, 3, 1, 2) # (b, h, w, c) -> (b,c,h,w)
        diff = diff.unsqueeze(0)
        x = self.dec(quantize)
        return x, diff, embed_ind

class UNetMem_v1(nn.Module):
    '''
    layer_nums mean num of layers of half of the Unet
    and the features change with ratio of 2
    '''
    def __init__(self,in_channel=3, out_channel=3,
                 embed_dim=64, n_embed=512,
                 layer_nums=4, features_root=64, bn=False):
        super(UNetMem_v1,self).__init__()
        self.inc = inconv(in_channel, 64, bn)
        self.down1 = down(64, 128,bn)
        self.down2 = down(128, 256,bn)
        self.down3 = down(256, 512,bn)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = (64, out_channel)
        #
        self.vq_inc = enc_quan_dec(64, embed_dim, n_embed)
        self.vq_down1 = enc_quan_dec(128, embed_dim, n_embed)
        self.vq_down2 = enc_quan_dec(256, embed_dim, n_embed)
        self.vq_down3 = enc_quan_dec(512, embed_dim, n_embed)

    def forward(self, x):
        x1 = self.inc(x)
        # x1, diff_1, embed_ind_1 = self.vq_inc(x1) #
        x2 = self.down1(x1)
        # x2, diff_2, embed_ind_2 = self.vq_down1(x2) #
        x3 = self.down2(x2)
        x3, diff_3, embed_ind_3 = self.vq_down2(x3) #
        x4 = self.down3(x3)
        x4, diff_4, embed_ind_4 = self.vq_down3(x4) #
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        #
        # diff = diff_1 + diff_2 + diff_3 + diff_4
        # embed_ind_tuple = (embed_ind_1, embed_ind_2, embed_ind_3, embed_ind_4)
        diff = diff_3 + diff_4
        embed_ind_tuple = (embed_ind_3, embed_ind_4)
        #
        return torch.sigmoid(x), diff, embed_ind_tuple

# ================================================== #

class enc_res_quan_dec(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed):
        super(enc_res_quan_dec, self).__init__()
        self.quan = enc_quan_dec(in_c, embed_dim, n_embed)

    def forward(self,x):
        out, diff, embed_ind = self.quan(x)
        out += x 
        return out, diff, embed_ind

class UNetMem_v2(nn.Module):
    '''
    layer_nums mean num of layers of half of the Unet
    and the features change with ratio of 2
    '''
    def __init__(self,in_channel=3, output_channel=3,
                 embed_dim=64, n_embed=512,
                 layer_nums=4, features_root=64, bn=False):
        super(UNetMem_v2,self).__init__()
        self.inc = inconv(in_channel, 64, bn)
        self.down1 = down(64, 128,bn)
        self.down2 = down(128, 256,bn)
        self.down3 = down(256, 512,bn)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = (64, output_channel)
        #
        self.vq_inc = enc_res_quan_dec(64, embed_dim, n_embed)
        self.vq_down1 = enc_res_quan_dec(128, embed_dim, n_embed)
        self.vq_down2 = enc_res_quan_dec(256, embed_dim, n_embed)
        self.vq_down3 = enc_res_quan_dec(512, embed_dim, n_embed)

    def forward(self, x):
        x1 = self.inc(x)
        # x1, diff_1, embed_ind_1 = self.vq_inc(x1) #
        x2 = self.down1(x1)
        # x2, diff_2, embed_ind_2 = self.vq_down1(x2) #
        x3 = self.down2(x2)
        x3, diff_3, embed_ind_3 = self.vq_down2(x3) #
        x4 = self.down3(x3)
        x4, diff_4, embed_ind_4 = self.vq_down3(x4) #
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        #
        # diff = diff_1 + diff_2 + diff_3 + diff_4
        # embed_ind_tuple = (embed_ind_1, embed_ind_2, embed_ind_3, embed_ind_4)
        diff = diff_3 + diff_4
        embed_ind_tuple = (embed_ind_3, embed_ind_4)
        #
        return torch.sigmoid(x), diff, embed_ind_tuple

# =================================================== #
class Quantize_topk(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, k=1):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.k = k

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim) # (b,h,w,c) -> (num, c)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])  # (num,) -> (b,h,w), since num==b*h*w
        quantize = self.embed_code(embed_ind)
        _, embed_ind_topk = (-dist).topk(self.k, dim=1)
        embed_ind_topk = embed_ind_topk.view(input.shape[0], input.shape[1], input.shape[2], -1)
        quantize_topk = self.embed_code(embed_ind_topk) # [b, h, w, k, emb_dim]
        quantize_topk = quantize_topk.view(input.shape[0], input.shape[1], input.shape[2], -1)
        assert quantize_topk.shape[-1] == self.k * self.dim
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            ) #
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum) #
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        
        return quantize_topk, diff, quantize

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class enc_quan_dec_topk(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_topk, self).__init__()
        self.enc = nn.Conv2d(in_c, embed_dim, 1) # 1x1 conv for depth_control
        self.quantize = Quantize_topk(dim=embed_dim, n_embed=n_embed, k=k)
        self.dec = nn.Conv2d(embed_dim*k, in_c, 1) # 1x1 conv for depth_control

    def forward(self,x):
        x = self.enc(x).permute(0, 2, 3, 1) # (b,c,h,w) -> (b, h, w, c)
        quantize, diff, quantize_one = self.quantize(x) # quantize: (b, h, w, c*k)
        quantize = quantize.permute(0, 3, 1, 2) # (b, h, w, c*k) -> (b,c*k,h,w)
        diff = diff.unsqueeze(0)
        x = self.dec(quantize)
        return x, diff, quantize_one

class UNetMem_v3(nn.Module):
    '''
    layer_nums mean num of layers of half of the Unet
    and the features change with ratio of 2
    '''
    def __init__(self,in_channel=3, output_channel=3,
                 embed_dim=64, n_embed=512, k=1,
                 layer_nums=4, features_root=64, bn=False):
        super(UNetMem_v3,self).__init__()
        self.inc = inconv(in_channel, 64, bn)
        self.down1 = down(64, 128,bn)
        self.down2 = down(128, 256,bn)
        self.down3 = down(256, 512,bn)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = (64, output_channel)
        #
        self.vq_inc = enc_quan_dec_topk(64, embed_dim, n_embed, k=k)
        self.vq_down1 = enc_quan_dec_topk(128, embed_dim, n_embed, k=k)
        self.vq_down2 = enc_quan_dec_topk(256, embed_dim, n_embed, k=k)
        self.vq_down3 = enc_quan_dec_topk(512, embed_dim, n_embed, k=k)

    def forward(self, x):
        x1 = self.inc(x)
        # x1, diff_1, embed_ind_1 = self.vq_inc(x1) #
        x2 = self.down1(x1)
        # x2, diff_2, embed_ind_2 = self.vq_down1(x2) #
        x3 = self.down2(x2)
        x3, diff_3, embed_ind_3 = self.vq_down2(x3) #
        x4 = self.down3(x3)
        x4, diff_4, embed_ind_4 = self.vq_down3(x4) #
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        #
        # diff = diff_1 + diff_2 + diff_3 + diff_4
        # embed_ind_tuple = (embed_ind_1, embed_ind_2, embed_ind_3, embed_ind_4)
        diff = diff_3 + diff_4
        embed_ind_tuple = (embed_ind_3, embed_ind_4)
        #
        return torch.sigmoid(x), diff, embed_ind_tuple

# =================================================== #

class enc_quan_dec_res_topk(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_res_topk, self).__init__()
        self.quan = enc_quan_dec_topk(in_c, embed_dim, n_embed, k=k)

    def forward(self,x):
        out, diff, embed_ind = self.quan(x)
        out += x
        return out, diff, embed_ind

class UNetMem_v4(nn.Module):
    def __init__(self, input_channels=3, output_channel=3,
                 embed_dim=64, n_embed=512, k=1,
                 layer_nums=4, features_root=64, bn=False):
        super(UNetMem_v4, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)
        #
        # self.vq_inc = enc_quan_dec_res_topk(64, embed_dim, n_embed, k=k)
        # self.vq_down1 = enc_quan_dec_res_topk(128, embed_dim, n_embed, k=k)
        self.vq_down2 = enc_quan_dec_res_topk(256, embed_dim, n_embed, k=k)
        self.vq_down3 = enc_quan_dec_res_topk(512, embed_dim, n_embed, k=k)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3, diff_3, embed_ind_3 = self.vq_down2(x3)  #
        x4 = self.down3(x3)
        x4, diff_4, embed_ind_4 = self.vq_down3(x4)  #
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        #
        diff = diff_3 + diff_4
        embed_ind_tuple = (embed_ind_3, embed_ind_4)

        return torch.tanh(x), diff, embed_ind_tuple


class enc_quan_dec_topk_ND(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_topk_ND, self).__init__()
        self.enc = nn.Conv2d(in_c, embed_dim, 1) # 1x1 conv for depth_control
        self.quantize = Quantize_topk(dim=embed_dim, n_embed=n_embed, k=k)
        # self.dec = nn.Conv2d(embed_dim*k, in_c, 1) # 1x1 conv for depth_control

    def forward(self,x):
        x = self.enc(x).permute(0, 2, 3, 1) # (b,c,h,w) -> (b, h, w, c)
        quantize, diff, embed_ind = self.quantize(x) # quantize: (b, h, w, c*k)
        quantize = quantize.permute(0, 3, 1, 2) # (b, h, w, c*k) -> (b,c*k,h,w)
        diff = diff.unsqueeze(0)
        # x = self.dec(quantize) # no (b,c*k,h,w) -> (b,c,h,w)
        return quantize, diff, embed_ind

class pre_unet(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, n_embed=512, k=1, bn=False):
        super(pre_unet, self).__init__()
        self.inc = inconv(in_c, 64, bn)
        self.vq_inc = enc_quan_dec_topk_ND(64, embed_dim, n_embed, k=k)

    def forward(self,x):
        x1 = self.inc(x)
        x2, diff_1, embed_ind = self.vq_inc(x1)
        return x1,x2, diff_1, embed_ind

class middle_unet(nn.Module):
    def __init__(self, in_c=64, out_c=64, bn=False):
        super(middle_unet, self).__init__()
        self.O2F = double_conv(in_c, in_c, bn)
        self.F20 = double_conv(in_c, in_c, bn)
        self.dec_x = nn.Conv2d(2*in_c, out_c, 1)
        self.dec_y = nn.Conv2d(2*in_c, out_c, 1)

    def forward(self,zx, zy):
        x1 = torch.cat([zx, self.O2F(zy)], 1)  # (b,c*k,h,w) -> (b,c*2*k,h,w)
        y1 = torch.cat([zy, self.F20(zx)], 1)
        x1 = self.dec_x(x1)  # (b,c*2*k,h,w) -> (b,c,h,w)
        y1 = self.dec_y(y1)
        return x1, y1

class post_unet(nn.Module):
    def __init__(self, input_channel=64, output_channel=3,
                 embed_dim=64, n_embed=512, k=1, bn=False):
        super(post_unet, self).__init__()
        self.down1 = down(input_channel, 128, bn)
        self.down2 = down(128, 256, bn)
        self.down3 = down(256, 512, bn)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = (64, output_channel)
        #
        self.vq_inc = enc_quan_dec_topk(input_channel, embed_dim, n_embed, k=k)
        self.vq_down1 = enc_quan_dec_topk(128, embed_dim, n_embed, k=k)
        self.vq_down2 = enc_quan_dec_topk(256, embed_dim, n_embed, k=k)
        self.vq_down3 = enc_quan_dec_topk(512, embed_dim, n_embed, k=k)

    def forward(self, x1, qx1, diff_1, embed_ind_1):
        x2 = self.down1(x1)
        x2, diff_2, embed_ind_2 = self.vq_down1(x2)  #
        x3 = self.down2(x2)
        x3, diff_3, embed_ind_3 = self.vq_down2(x3)  #
        # print("**x3 size**:", x3.size())
        # ipdb.set_trace()
        x4 = self.down3(x3)  # bug to fix, todo
        # print("**x4 size**:", x4.size())
        x4, diff_4, embed_ind_4 = self.vq_down3(x4)  #
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, qx1) # 注意是 quantize_x1
        x = self.outc(x)
        #
        diff = diff_1 + diff_2 + diff_3 + diff_4
        embed_ind_tuple = (embed_ind_1, embed_ind_2, embed_ind_3, embed_ind_4)
        #
        return torch.sigmoid(x), diff, embed_ind_tuple

class UNetMem_v5(nn.Module):
    '''
    layer_nums mean num of layers of half of the Unet
    and the features change with ratio of 2
    '''
    def __init__(self,in_channel=(3,2), output_channel=(3,2),
                 layer_nums=4, features_root=64,
                 embed_dim=64, n_embed=512, k=1, bn=False):
        super(UNetMem_v5,self).__init__()

        self.pre_unet_x = pre_unet(in_channel[0], embed_dim, n_embed, k, bn) # out_c: k*embed_dim
        self.pre_unet_y = pre_unet(in_channel[1], embed_dim, n_embed, k, bn)
        self.middle_unet = middle_unet(k*embed_dim, 64, bn)
        #
        self.post_unet_x = post_unet(64, output_channel[0], embed_dim, n_embed, k, bn)
        self.post_unet_y = post_unet(64, output_channel[1], embed_dim, n_embed, k, bn)

    def forward(self, x, y):

        x1, zx, diff_1_x, embed_ind_1_x = self.pre_unet_x(x)  # x1: (b,64,h,w)
        y1, zy, diff_1_y, embed_ind_1_y = self.pre_unet_y(y)  # zy: (b,emb*k,h,w)
        #
        # (2) fusion stage: (b,emb*k,h,w) -> (b,emb*2*k,h,w) -> (b,64,h,w)
        qx1, qy1 = self.middle_unet(zx, zy)
        #
        # (3) after fusion stage: post unet
        gen_x, diff_x, embed_ind_tuple_x = self.post_unet_x(x1, qx1, diff_1_x, embed_ind_1_x)
        gen_y, diff_y, embed_ind_tuple_y = self.post_unet_y(y1, qy1, diff_1_y, embed_ind_1_y)
        #
        return gen_x, gen_y, diff_x+diff_y, (embed_ind_tuple_x,embed_ind_tuple_x)

def test_UNetMem_v5():
    # unittest for UNetMem_v5
    from torchsummaryX import summary
    #
    in_channel = (27, 16)
    output_channel = (3, 2)
    layer_nums = 4
    features_root = 64
    embed_dim = 64
    n_embed = 512
    k = 1
    bn = False
    net = UNetMem(in_channel, output_channel, layer_nums, features_root,
                    embed_dim, n_embed, k)
    #
    x = torch.randn(1, 27, 256, 256)  # 9 -> 1, 3-channel
    y = torch.randn(1, 16, 256, 256)  # 8-> 1, 2-c
    input_list = x, y
    summary(net, x,y)

    '''
                                 Totals
    Total params             14.799877M
    Trainable params         14.799877M
    Non-trainable params            0.0
    Mult-Adds             87.140865792G
    '''


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 2-conv res_block
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, in_channel=64, out_channel=64,
                 zero_init_residual=False,groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upsample = nn.functional.interpolate
        self.layer1 = self._make_layer(block, 64, layers[0]) # 后面 layer的stride==1 (modify by haozhang)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        # self.layer4(x): (b,128,64, 64)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x, memory): # x: (b,emb*2*k,256,256)
        x = self.conv1(x) # (F=3, P=1, S=1), spatial resolution不做下采样
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # ds (spatial -> 1/2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.upsample(x, scale_factor=2) # up ds
        x = self.layer3(x)
        x = self.layer4(x) # (b,128,?,?)


        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward

def resnet18(pretrained=False, progress=True, **kwargs):
    model = ResNet_Backbone(BasicBlock, [2, 2, 2, 2],
                            in_channel = 64, out_channel = 64)
    return model

class bridge_net(nn.Module):
    pass

class middle_unet(nn.Module):
    def __init__(self, in_c=64, out_c=64, bn=False):
        super(middle_unet, self).__init__()
        self.O2F = double_conv(in_c, in_c, bn) # (b,c*k,h,w) -> (b,c*k,h,w)
        self.F20 = double_conv(in_c, in_c, bn)
        self.dec_x = nn.Conv2d(2*in_c, out_c, 1) # 1x1 conv for depth_control
        self.dec_y = nn.Conv2d(2*in_c, out_c, 1)

    def forward(self,zx, zy):
        x1 = torch.cat([zx, self.O2F(zy)], 1)  # (b,c*k,h,w) -> (b,c*2*k,h,w)
        y1 = torch.cat([zy, self.F20(zx)], 1)
        x1 = self.dec_x(x1)  # (b,c*2*k,h,w) -> (b,c,h,w)
        y1 = self.dec_y(y1)
        return x1, y1

class rgb_post_unet(nn.Module):
    def __init__(self, input_channel=64, output_channel=3,
                 embed_dim=64, n_embed=512, k=1, bn=False):
        super(post_unet, self).__init__()
        self.down1 = down(input_channel, 128, bn)
        self.down2 = down(128, 256, bn)
        self.down3 = down(256, 512, bn)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = (64, output_channel)
        #
        self.vq_down1 = enc_quan_dec_topk(128, embed_dim, n_embed, k=k)
        self.vq_down2 = enc_quan_dec_topk(256, embed_dim, n_embed, k=k)
        self.vq_down3 = enc_quan_dec_topk(512, embed_dim, n_embed, k=k)

    def forward(self, x1, qx1, diff_1, embed_ind_1):
        x2 = self.down1(x1)
        x2, diff_2, embed_ind_2 = self.vq_down1(x2)  #
        x3 = self.down2(x2)
        x3, diff_3, embed_ind_3 = self.vq_down2(x3)  #
        # print("**x3 size**:", x3.size())
        # ipdb.set_trace()
        x4 = self.down3(x3)  # bug to fix, todo
        # print("**x4 size**:", x4.size())
        x4, diff_4, embed_ind_4 = self.vq_down3(x4)  #
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, qx1) # 注意是 quantize_x1
        x = self.outc(x)
        #
        diff = diff_1 + diff_2 + diff_3 + diff_4
        embed_ind_tuple = (embed_ind_1, embed_ind_2, embed_ind_3, embed_ind_4)
        #
        return torch.sigmoid(x), diff, embed_ind_tuple

class op_post_unet(nn.Module):
    def __init__(self, input_channel=64, output_channel=3,
                 embed_dim=64, n_embed=512, k=1, bn=False):
        super(post_unet, self).__init__()
        self.down1 = down(input_channel, 128, bn)
        self.up3 = up(128, 64)
        self.outc = (64, output_channel)
        #
        self.vq_down1 = enc_quan_dec_topk(128, embed_dim, n_embed, k=k)

    def forward(self, x1, qx1, diff_1, embed_ind_1):
        x2 = self.down1(x1)
        x2, diff_2, embed_ind_2 = self.vq_down1(x2)  #
        x = self.up3(x2, qx1) # 注意是 quantize_x1
        x = self.outc(x)
        #
        diff = diff_1 + diff_2
        embed_ind_tuple = (embed_ind_1, embed_ind_2)
        #
        return torch.sigmoid(x), diff, embed_ind_tuple

class UNetMem(nn.Module):
    '''
    layer_nums mean num of layers of half of the Unet
    and the features change with ratio of 2
    '''
    def __init__(self,in_channel=(3,2), output_channel=(3,2),
                 layer_nums=4, features_root=64,
                 embed_dim=64, n_embed=512, k=1, bn=False):
        super(UNetMem,self).__init__()

        self.pre_unet_x = pre_unet(in_channel[0], embed_dim, n_embed, k, bn) # out_c: k*embed_dim
        self.pre_unet_y = pre_unet(in_channel[1], embed_dim, n_embed, k, bn)
        self.middle_unet = middle_unet(k*embed_dim, 64, bn) #
        #
        self.post_unet_x = rgb_post_unet(64, output_channel[0], embed_dim, n_embed, k, bn)
        self.post_unet_y = op_post_unet(64, output_channel[1], embed_dim, n_embed, k, bn)

    def forward(self, x, y):

        x1, zx, diff_1_x, embed_ind_1_x = self.pre_unet_x(x)  # x1: (b,64,h,w)
        y1, zy, diff_1_y, embed_ind_1_y = self.pre_unet_y(y)  # zy: (b,emb*k,h,w)
        qx1, qy1 = self.middle_unet(zx, zy)
        gen_x, diff_x, embed_ind_tuple_x = self.post_unet_x(x1, qx1, diff_1_x, embed_ind_1_x)
        gen_y, diff_y, embed_ind_tuple_y = self.post_unet_y(y1, qy1, diff_1_y, embed_ind_1_y)
        return gen_x, gen_y, diff_x+diff_y, (embed_ind_tuple_x,embed_ind_tuple_x)


def test_UNetMem_v6():
    from torchsummaryX import summary
    #
    in_channel = (27, 16)
    output_channel = (3, 2)
    layer_nums = 4
    features_root = 64
    embed_dim = 64
    n_embed = 512
    k = 1
    bn = False
    net = UNetMem(in_channel, output_channel, layer_nums, features_root,
                    embed_dim, n_embed, k)
    #
    x = torch.randn(1, 27, 256, 256)  # 9 -> 1, 3-channel
    y = torch.randn(1, 16, 256, 256)  # 8-> 1, 2-c
    input_list = x, y
    summary(net, x,y)

    '''

    '''
# ========================================================================= #
#
class bridge_v1_topk_fusion(nn.Module):
    def __init__(self, in_c=64):
        super(middle_unet, self).__init__()
        self.O2F = double_conv(in_c, in_c) # (b,c*k,h,w) -> (b,c*k,h,w)
        self.F20 = double_conv(in_c, in_c)
        self.dec_x = nn.Conv2d(2*in_c, in_c, 1) # 1x1 conv for depth_control
        self.dec_y = nn.Conv2d(2*in_c, in_c, 1)

    def forward(self,zx, zy):
        x1 = torch.cat([zx, self.O2F(zy)], 1)  # (b,c*k,h,w) -> (b,c*2*k,h,w)
        y1 = torch.cat([zy, self.F20(zx)], 1)
        x = self.dec_x(x1)  # (b,c*2*k,h,w) -> (b,c*k,h,w)
        y = self.dec_y(y1)
        return x, y
#
class bridge_v1_topk_fusion_v2(nn.Module):
    def __init__(self, in_c=64):
        super(middle_unet, self).__init__()
        self.O2F = double_conv(in_c, in_c) # (b,c*k,h,w) -> (b,c*k,h,w)
        self.F20 = double_conv(in_c, in_c)

    def forward(self,zx, zy):
        x = zx + self.O2F(zy)  # (b,c*k,h,w)+(b,c*k,h,w) -> (b,c*k,h,w)
        y = zy + self.F20(zx)
        return x, y

class UNetMem_v7(nn.Module):
    def __init__(self, input_channels=3, output_channel=3,
                 embed_dim=64, n_embed=512, k=1,
                 layer_nums=4, features_root=64):
        super(UNetMem_v7, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)
        #
        self.vq_down3 = enc_quan_dec_res_topk(512, embed_dim, n_embed, k=k)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4, diff_4, quantize_one = self.vq_down3(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        #
        diff = diff_4

        return torch.tanh(x), diff, quantize_one


class bridge_v1(nn.Module):
    def __init__(self, in_c=64):
        super(bridge, self).__init__()
        self.O2F = double_conv(in_c, in_c) # (b,c,h,w) -> (b,c,h,w)
        self.F20 = double_conv(in_c, in_c)
        self.dec_x = nn.Conv2d(2*in_c, in_c, 1) # 1x1 conv for depth_control
        self.dec_y = nn.Conv2d(2*in_c, in_c, 1)

    def forward(self,zx, zy):
        x1 = torch.cat([zx, self.O2F(zy)], 1)  # (b,c*2,h,w) -> (b,c*2,h,w)
        y1 = torch.cat([zy, self.F20(zx)], 1)
        x = self.dec_x(x1)  # (b,c*2*k,h,w) -> (b,c*k,h,w)
        y = self.dec_y(y1)
        return x, y

#
class bridge(nn.Module):
    def __init__(self, in_c=64):
        super(bridge, self).__init__()
        self.O2F = double_conv(in_c, in_c) # (b,c*k,h,w) -> (b,c*k,h,w)
        self.F20 = double_conv(in_c, in_c)

    def forward(self,zx, zy):
        x = zx + self.O2F(zy)  # (b,c*k,h,w)+(b,c*k,h,w) -> (b,c*k,h,w)
        y = zy + self.F20(zx)
        return x, y

class twostream(nn.Module):

    def __init__(self, rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                 embed_dim=64, n_embed=512, k=1,
                 layer_nums=4, features_root=64):
        super(twostream, self).__init__()
        self.rgb = UNetMem_v7(rgb_in_c, rgb_out_c,
                  embed_dim, n_embed, k,
                 layer_nums, features_root)
        self.op = UNetMem_v7(op_in_c, op_out_c,
                  embed_dim, n_embed, k,
                  layer_nums, features_root)
        self.bridge = bridge(in_c=512) # in_c in vq_topk_res

    def forward(self, rgb_x, op_x):
        rgb_x1 = self.rgb.inc(rgb_x)
        rgb_x2 = self.rgb.down1(rgb_x1)
        rgb_x3 = self.rgb.down2(rgb_x2)
        rgb_x4 = self.rgb.down3(rgb_x3)
        self.quant_befor = rgb_x4
        rgb_x4, rgb_diff_4, rgb_quant_4 = self.rgb.vq_down3(rgb_x4)
        self.quant_after = rgb_x4
        op_x1 = self.op.inc(op_x)
        op_x2 = self.op.down1(op_x1)
        op_x3 = self.op.down2(op_x2)
        op_x4 = self.op.down3(op_x3)
        op_x4, op_diff_4, op_quant_4 = self.op.vq_down3(op_x4)
        rgb_x4, op_x4 = self.bridge(rgb_x4, op_x4)
        rgb_x = self.rgb.up1(rgb_x4, rgb_x3)
        rgb_x = self.rgb.up2(rgb_x, rgb_x2)
        rgb_x = self.rgb.up3(rgb_x, rgb_x1)
        rgb_x = self.rgb.outc(rgb_x)
        op_x = self.op.up1(op_x4, op_x3)
        op_x = self.op.up2(op_x, op_x2)
        op_x = self.op.up3(op_x, op_x1)
        op_x = self.op.outc(op_x)
        #
        diff = (rgb_diff_4,op_diff_4)
        embed_ind_tuple = (rgb_quant_4,op_quant_4)

        return torch.tanh(rgb_x), torch.tanh(op_x), diff, embed_ind_tuple

#
class bridge_concat_dire(nn.Module):
    def __init__(self, in_c=64):
        super(bridge_concat_dire, self).__init__()
        self.dec = nn.Conv2d(in_c, in_c//2, 1) # (b,c*2,h,w) -> (b,c*1,h,w)

    def forward(self,zx, zy):
        z = torch.cat([zx, zy], 1) # (b,c*1,h,w) -> (b,c*2,h,w)
        z = self.dec(z)

        return z, z

class bridge_add_dire(nn.Module):
    def __init__(self, in_c=64):
        super(bridge_add_dire, self).__init__()
        self.dec = nn.Conv2d(in_c, in_c//2, 1) # (b,c*2,h,w) -> (b,c*1,h,w)

    def forward(self,zx, zy):
        z = zx + zy

        return z, z

class twostream_concat_dire(nn.Module):

    def __init__(self, rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                 embed_dim=64, n_embed=512, k=1,
                 layer_nums=4, features_root=64):
        super(twostream_concat_dire, self).__init__()
        self.rgb = UNetMem_v7(rgb_in_c, rgb_out_c,
                  embed_dim, n_embed, k,
                 layer_nums, features_root)
        self.op = UNetMem_v7(op_in_c, op_out_c,
                  embed_dim, n_embed, k,
                  layer_nums, features_root)
        self.bridge = bridge(in_c=512)

    def forward(self, rgb_x, op_x):
        rgb_x1 = self.rgb.inc(rgb_x)
        rgb_x2 = self.rgb.down1(rgb_x1)
        rgb_x3 = self.rgb.down2(rgb_x2)
        rgb_x4 = self.rgb.down3(rgb_x3)
        rgb_x4, rgb_diff_4, rgb_embed_ind_4 = self.rgb.vq_down3(rgb_x4)
        op_x1 = self.op.inc(op_x)
        op_x2 = self.op.down1(op_x1)
        op_x3 = self.op.down2(op_x2)
        op_x4 = self.op.down3(op_x3)
        op_x4, op_diff_4, op_embed_ind_4 = self.op.vq_down3(op_x4)
        rgb_x4, op_x4 = self.bridge(rgb_x4, op_x4)
        rgb_x = self.rgb.up1(rgb_x4, rgb_x3)
        rgb_x = self.rgb.up2(rgb_x, rgb_x2)
        rgb_x = self.rgb.up3(rgb_x, rgb_x1)
        rgb_x = self.rgb.outc(rgb_x)
        op_x = self.op.up1(op_x4, op_x3)
        op_x = self.op.up2(op_x, op_x2)
        op_x = self.op.up3(op_x, op_x1)
        op_x = self.op.outc(op_x)
        diff = rgb_diff_4 + op_diff_4
        embed_ind_tuple = (rgb_embed_ind_4,op_embed_ind_4)

        return torch.tanh(rgb_x), torch.tanh(op_x), diff, embed_ind_tuple

class twostream_add_dire(nn.Module):

    def __init__(self, rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                 embed_dim=64, n_embed=512, k=1,
                 layer_nums=4, features_root=64):
        super(twostream_add_dire, self).__init__()
        self.rgb = UNetMem_v7(rgb_in_c, rgb_out_c,
                  embed_dim, n_embed, k,
                 layer_nums, features_root)
        self.op = UNetMem_v7(op_in_c, op_out_c,
                  embed_dim, n_embed, k,
                  layer_nums, features_root)
        self.bridge = bridge(in_c=512)

    def forward(self, rgb_x, op_x):
        rgb_x1 = self.rgb.inc(rgb_x)
        rgb_x2 = self.rgb.down1(rgb_x1)
        rgb_x3 = self.rgb.down2(rgb_x2)
        rgb_x4 = self.rgb.down3(rgb_x3)
        rgb_x4, rgb_diff_4, rgb_embed_ind_4 = self.rgb.vq_down3(rgb_x4)
        op_x1 = self.op.inc(op_x)
        op_x2 = self.op.down1(op_x1)
        op_x3 = self.op.down2(op_x2)
        op_x4 = self.op.down3(op_x3)
        op_x4, op_diff_4, op_embed_ind_4 = self.op.vq_down3(op_x4)
        rgb_x4, op_x4 = self.bridge(rgb_x4, op_x4)
        rgb_x = self.rgb.up1(rgb_x4, rgb_x3)
        rgb_x = self.rgb.up2(rgb_x, rgb_x2)
        rgb_x = self.rgb.up3(rgb_x, rgb_x1)
        rgb_x = self.rgb.outc(rgb_x)
        op_x = self.op.up1(op_x4, op_x3)
        op_x = self.op.up2(op_x, op_x2)
        op_x = self.op.up3(op_x, op_x1)
        op_x = self.op.outc(op_x)
        embed_ind_tuple = (rgb_embed_ind_4,op_embed_ind_4)

        return torch.tanh(rgb_x), torch.tanh(op_x), diff, embed_ind_tuple

#
def get_twostream_concat_dire(in_channel, out_channel,
                 embed_dim, n_embed, k,
                 layer_nums=4, features_root=64):
    rgb_in_c, op_in_c = in_channel
    rgb_out_c, op_out_c = out_channel
    return twostream_concat_dire(rgb_in_c=rgb_in_c, rgb_out_c=rgb_out_c,
                     op_in_c=op_in_c, op_out_c=op_out_c,
                 embed_dim=embed_dim, n_embed=n_embed, k=k,
                 layer_nums=layer_nums, features_root=features_root)

def get_twostream_add_dire(in_channel, out_channel,
                 embed_dim, n_embed, k,
                 layer_nums=4, features_root=64):
    rgb_in_c, op_in_c = in_channel
    rgb_out_c, op_out_c = out_channel
    return twostream_add_dire(rgb_in_c=rgb_in_c, rgb_out_c=rgb_out_c,
                     op_in_c=op_in_c, op_out_c=op_out_c,
                 embed_dim=embed_dim, n_embed=n_embed, k=k,
                 layer_nums=layer_nums, features_root=features_root)

# =============================================================================== #
def get_unet(in_channel, out_channel,
             embed_dim=0, n_embed=0, k=0):
    return UNet(in_channel, out_channel)

def test_get_unet():
    in_channel = 4 * 3
    out_channel = 3
    net = get_unet(in_channel, out_channel)
    b, c, h, w = 2, 4 * 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    '''
    Total params              7.192195M
    Trainable params          7.192195M
    Non-trainable params            0.0
    Mult-Adds             37.107010304G
    '''
    return UNetMem_v1()


def get_unet_vq(in_channel, out_channel,
                embed_dim=64, n_embed=512, k=1):
    # print("=== in_c", type(in_channel))
    return UNetMem_v1(in_channel, out_channel, embed_dim, n_embed)

def test_get_unet_vq():
    in_channel = 4 * 3
    out_channel = 3
    net = get_unet_vq(in_channel, out_channel)
    b, c, h, w = 2, 4 * 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    '''
    Total params              7.316291M
    Trainable params          7.316291M
    Non-trainable params            0.0
    Mult-Adds             38.113643264G
    '''


def get_unet_vq_res(in_channel, out_channel,
                embed_dim=64, n_embed=512, k=1):
    return UNetMem_v2(in_channel, out_channel, embed_dim, n_embed)

def test_get_unet_vq_res():
    in_channel = 4 * 3
    out_channel = 3
    net = get_unet_vq_res(in_channel, out_channel)
    b, c, h, w = 2, 4 * 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    '''
    Total params              7.316291M
    Trainable params          7.316291M
    Non-trainable params            0.0
    Mult-Adds             38.113643264G
    '''


def get_unet_vq_topk(in_channel, out_channel,
                embed_dim=64, n_embed=512, k=1):

    return UNetMem_v3(in_channel, out_channel, embed_dim, n_embed, k)

def test_get_unet_vq_topk():
    in_channel = 4 * 3
    out_channel = 3
    net = get_unet_vq_topk(in_channel, out_channel, k=2)
    b, c, h, w = 2, 4 * 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    '''
    Total params              7.377731M
    Trainable params          7.377731M
    Non-trainable params            0.0
    Mult-Adds             38.616959744G
    '''


def get_unet_vq_topk_res(in_channel, out_channel,
                         embed_dim=64, n_embed=512, k=1):
    return UNetMem_v7(in_channel, out_channel, embed_dim, n_embed, k)
    # UNetMem_v4(in_channel, out_channel, embed_dim, n_embed, k)

def test_get_unet_vq_topk_res():
    in_channel = 4 * 3
    out_channel = 3
    net = get_unet_vq_topk_res(in_channel, out_channel, k=2)
    b, c, h, w = 2, 4 * 3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)

    '''
    Total params              7.377731M
    Trainable params          7.377731M
    Non-trainable params            0.0
    Mult-Adds             38.616959744G
    
    Total params              7.805891M
    Trainable params          7.805891M
    Non-trainable params            0.0
    Mult-Adds             42.140175104G
    '''




def get_twostream(in_channel, out_channel,
                 embed_dim, n_embed, k,
                 layer_nums=4, features_root=64):
    rgb_in_c, op_in_c = in_channel
    rgb_out_c, op_out_c = out_channel
    return twostream(rgb_in_c=rgb_in_c, rgb_out_c=rgb_out_c,
                     op_in_c=op_in_c, op_out_c=op_out_c,
                 embed_dim=embed_dim, n_embed=n_embed, k=k,
                 layer_nums=layer_nums, features_root=features_root)

def test_get_twostream():
    rgb_in_c = 3 * 4
    rgb_out_c = 3 * 1
    op_in_c = 2 * 3
    op_out_c = 2 * 1
    embed_dim, n_embed, k = 64, 512, 2
    net = get_twostream(rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                 embed_dim, n_embed, k)
    #
    b, c, h, w = 2, 4 * 3, 256, 256
    rgb = torch.randn(b, c, h, w)
    b, c, h, w = 2, 3 * 2, 256, 256
    op = torch.randn(b, c, h, w)
    #
    summary(net, rgb, op)
    print("=========================")
    print(net.state_dict().keys())
    '''
    -----
                                Totals
    Total params            25.049029M
    Trainable params        25.049029M
    Non-trainable params           0.0
    Mult-Adds             93.67978752G
    '''

def test_twostream_load_pretrain():
    #
    def loader_rgb_op_branch(model, rgb_model_path, op_model_path, logger=None):
        rgb_pretrained_dict = torch.load(rgb_model_path, map_location="cuda:0")
        op_pretrained_dict = torch.load(op_model_path, map_location="cuda:0")
        model_dict = model.state_dict()

        def get_wapper_key(prefix, key):
            wapped_key = prefix + '.' + key

            return wapped_key

        rgb_state_dict = {get_wapper_key("rgb", k): v for k, v in rgb_pretrained_dict.items() if
                          get_wapper_key("rgb", k) in model_dict.keys()}
        op_state_dict = {get_wapper_key("op", k): v for k, v in op_pretrained_dict.items() if
                          get_wapper_key("op", k) in model_dict.keys()}

        model_dict.update(rgb_state_dict)
        model_dict.update(op_state_dict)
        model.load_state_dict(model_dict)

        # logger.info('load model from multi_pretain file: {},{} successfully!'.format(
        # rgb_model_path, op_model_path))
        #
        return model, 0  # g_step

    rgb_in_c = 3 * 4
    rgb_out_c = 3 * 1
    op_in_c = 2 * 3
    op_out_c = 2 * 1
    embed_dim, n_embed, k = 64, 256, 2
    net = get_twostream(rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                 embed_dim, n_embed, k)
    #
    b, c, h, w = 2, 3 * 4, 256, 256
    rgb = torch.randn(b, c, h, w)
    b, c, h, w = 2, 2 * 3, 256, 256
    op = torch.randn(b, c, h, w)
    #
    rgb_ckpt = ""
    op_ckpt = ""
    net,g_step = loader_rgb_op_branch(net, rgb_ckpt, op_ckpt)
    #
    with torch.no_grad():
        rgb, op, diff, _ = net(rgb, op)
    print("rgb: ", rgb.size(), rgb.min(), rgb.max())
    print("op: ", op.size(), op.min(), op.max())

#
def test_twostream_concat_dire():
    rgb_in_c = 3 * 4
    rgb_out_c = 3 * 1
    op_in_c = 2 * 3
    op_out_c = 2 * 1
    embed_dim, n_embed, k = 64, 512, 2
    net = twostream_concat_dire(rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                        embed_dim, n_embed, k)
    #
    b, c, h, w = 2, 4 * 3, 256, 256
    rgb = torch.randn(b, c, h, w)
    b, c, h, w = 2, 3 * 2, 256, 256
    op = torch.randn(b, c, h, w)
    #
    summary(net, rgb, op)
    print("=========================")
    print(net.state_dict().keys())

def test_twostream_add_dire():
    rgb_in_c = 3 * 4
    rgb_out_c = 3 * 1
    op_in_c = 2 * 3
    op_out_c = 2 * 1
    embed_dim, n_embed, k = 64, 512, 2
    net = twostream_add_dire(rgb_in_c, rgb_out_c, op_in_c, op_out_c,
                                embed_dim, n_embed, k)
    #
    b, c, h, w = 2, 4 * 3, 256, 256
    rgb = torch.randn(b, c, h, w)
    b, c, h, w = 2, 3 * 2, 256, 256
    op = torch.randn(b, c, h, w)
    #
    summary(net, rgb, op)
    print("=========================")
    print(net.state_dict().keys())



if __name__ == '__main__':
    # test_UNetMem_v5()
    # test_UNet()
    # test_UNetMem_v6()

    # ------------------------ #

    # test_get_unet()
    # test_get_unet_vq()
    # test_get_unet_vq_res()
    # test_get_unet_vq_topk()
    # test_get_unet_vq_topk_res()
    # test_get_twostream()
    # test_twostream_load_pretrain()
    #
    # for ablation
    test_twostream_concat_dire()
    test_twostream_add_dire()
    'python -m Code.models.unet'
    'python Code/models/unet.py'

