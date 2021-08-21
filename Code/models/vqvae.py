import os
#
import torch
from torch import nn
from torch.nn import functional as F
from torchsummaryX import summary


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
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) #
        embed_ind = embed_ind.view(*input.shape[:-1]) #
        #
        quantize = self.embed_code(embed_ind) #
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


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input # vilia res_block

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        # c->1/2，w(h)->1/4
        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        out_channel, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim, # concat ?
            out_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


def get_vqvae(in_channel=3, out_channel=3,
              channel=128, n_res_block=2, n_res_channel=32,
                embed_dim=64, n_embed=512, decay=0.99,k=None):

    return VQVAE(in_channel=in_channel, out_channel=out_channel,
                 channel=channel, n_res_block=n_res_block,
                 n_res_channel=n_res_channel,
                 embed_dim=embed_dim, n_embed=n_embed, decay=decay)

def test_vqvae():
    in_channel = 9*3
    out_channel = 3
    net = get_vqvae(in_channel, out_channel)
    b, c, h, w = 2, 9*3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)
    '''
                                Totals
    Total params             1.413443M
    Trainable params         1.413443M
    Non-trainable params           0.0
    Mult-Adds             6.534725632G
    '''


# ============================================================== #

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
        _, embed_ind = (-dist).max(1) #
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        _, embed_ind_topk = (-dist).topk(self.k, dim=1)
        embed_ind_topk = embed_ind_topk.view(input.shape[0], input.shape[1], input.shape[2], -1)
        #
        quantize_topk = self.embed_code(embed_ind_topk) # [b, h, w, k, emb_dim]
        quantize_topk = quantize_topk.view(input.shape[0], input.shape[1], input.shape[2], -1)
        assert quantize_topk.shape[-1] == self.k * self.dim
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
        input_topk = input.repeat(1,1,1,self.k) #
        diff = (quantize_topk.detach() - input_topk).pow(2).mean() #
        quantize_topk = input_topk + (quantize_topk - input_topk).detach() #

        return quantize_topk, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class enc_quan_dec_topk(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_topk, self).__init__()
        self.enc = nn.Conv2d(in_c, embed_dim, 1) # 1x1 conv for depth_control
        self.quantize = Quantize_topk(dim=embed_dim, n_embed=n_embed, k=k)
        self.dec = nn.Conv2d(embed_dim*k, embed_dim, 1) # 1x1 conv for depth_reduction

    def forward(self,x):
        x = self.enc(x).permute(0, 2, 3, 1) # (b,c,h,w) -> (b, h, w, c)
        quantize, diff, embed_ind = self.quantize(x) # quantize: (b, h, w, c*k)
        quantize = quantize.permute(0, 3, 1, 2) # (b, h, w, c*k) -> (b,c*k,h,w)
        diff = diff.unsqueeze(0)
        x = self.dec(quantize)
        return x, diff, embed_ind

class VQVAE_topk(nn.Module):
    def __init__(
        self, in_channel=3, out_channel=3, channel=128, n_res_block=2,
        n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, k=1):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1) #
        self.quantize_t = enc_quan_dec_topk(channel, embed_dim, n_embed, k) #
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        ) #
        # self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = enc_quan_dec_topk(embed_dim+channel, embed_dim, n_embed, k) #
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        ) #
        self.dec = Decoder(embed_dim+embed_dim, out_channel, channel,
            n_res_block, n_res_channel, stride=4,) #

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input) #
        dec = self.decode(quant_t, quant_b) #

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input) #
        enc_t = self.enc_t(enc_b) #
        # quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) #
        quant_t, diff_t, id_t = self.quantize_t(enc_t)
        # quant_t = quant_t.permute(0, 3, 1, 2)
        # diff_t = diff_t.unsqueeze(0)
        dec_t = self.dec_t(quant_t) #
        enc_b = torch.cat([dec_t, enc_b], 1) #
        # quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(enc_b) #
        # quant_b = quant_b.permute(0, 3, 1, 2)
        # diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) #
        quant = torch.cat([upsample_t, quant_b], 1) #
        dec = self.dec(quant) # 4x upsample

        return dec

    def decode_code(self, code_t, code_b):
        # quant_t = self.quantize_t.embed_code(code_t)
        # quant_t = quant_t.permute(0, 3, 1, 2)
        # quant_b = self.quantize_b.embed_code(code_b)
        # quant_b = quant_b.permute(0, 3, 1, 2)
        #
        # dec = self.decode(quant_t, quant_b)
        #
        # return dec
        pass #


def get_vqvae_topk(in_channel=3, out_channel=3,
                embed_dim=64, n_embed=512, k=1,
                channel=128, n_res_block=2, n_res_channel=32, decay=0.99
              ):

    return VQVAE_topk(in_channel=in_channel, out_channel=out_channel, channel=channel,
        n_res_block=n_res_block, n_res_channel=n_res_channel,
        embed_dim=embed_dim, n_embed=n_embed, decay=decay, k=k)

def test_vqvae_topk():
    in_channel = 9*3
    out_channel = 3
    net = get_vqvae_topk(in_channel, out_channel)
    b, c, h, w = 2, 9*3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)
    '''
                               Totals
    Total params             1.442371M
    Trainable params         1.442371M
    Non-trainable params           0.0
    Mult-Adds             6.614417408G
    '''

# =============================================================== #

class enc_quan_dec_res_topk(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_res_topk, self).__init__()
        self.quan = enc_quan_dec_topk(in_c, embed_dim, n_embed, k=k)
        self.enc_x = nn.Conv2d(in_c, embed_dim, 1)

    def forward(self,x):
        out, diff, embed_ind = self.quan(x)
        out += self.enc_x(x)
        return out, diff, embed_ind


class VQVAE_topk_res(nn.Module):
    def __init__(
        self, in_channel=3, out_channel=3, channel=128, n_res_block=2,
        n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, k=1):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        # self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = enc_quan_dec_res_topk(channel, embed_dim, n_embed, k)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        # self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1) #
        self.quantize_b = enc_quan_dec_res_topk(embed_dim+channel, embed_dim, n_embed, k) #
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        ) #
        self.dec = Decoder(embed_dim+embed_dim, out_channel, channel,
            n_res_block, n_res_channel, stride=4,) #

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b) #

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        # quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) #
        quant_t, diff_t, id_t = self.quantize_t(enc_t) #
        # quant_t = quant_t.permute(0, 3, 1, 2) #
        # diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t) #
        enc_b = torch.cat([dec_t, enc_b], 1) #

        # quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(enc_b) #
        # quant_b = quant_b.permute(0, 3, 1, 2)
        # diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1) #
        dec = self.dec(quant) #

        return dec

    def decode_code(self, code_t, code_b):
        # quant_t = self.quantize_t.embed_code(code_t)
        # quant_t = quant_t.permute(0, 3, 1, 2)
        # quant_b = self.quantize_b.embed_code(code_b)
        # quant_b = quant_b.permute(0, 3, 1, 2)
        #
        # dec = self.decode(quant_t, quant_b)
        #
        # return dec
        pass


def get_vqvae_topk_res(in_channel=3, out_channel=3,
           channel=128,n_res_block=2, n_res_channel=32,
            embed_dim=64, n_embed=512, decay=0.99, k=1):

    return VQVAE_topk_res(in_channel=in_channel, out_channel=out_channel, channel=channel,
        n_res_block=n_res_block, n_res_channel=n_res_channel,
        embed_dim=embed_dim, n_embed=n_embed, decay=decay, k=k)

def test_vqvae_topk_res():
    in_channel = 9*3
    out_channel = 3
    net = get_vqvae_topk_res(in_channel, out_channel)
    b, c, h, w = 2, 9*3, 256, 256
    in_tensor = torch.randn(b, c, h, w)
    summary(net, in_tensor)
    '''
                               Totals
    Total params             1.442371M
    Trainable params         1.442371M
    Non-trainable params           0.0
    Mult-Adds             6.614417408G
    '''

# ================================================================== #

class middle_unet(nn.Module):
    def __init__(self, in_c=64, out_c=64, bn=False):
        super(middle_unet, self).__init__()
        self.O2F = ResBlock(in_c, in_c) # (b,c*k,h,w) -> (b,c*k,h,w)
        self.F20 = ResBlock(in_c, in_c)
        self.dec_x = nn.Conv2d(2*in_c, out_c, 1) # 1x1 conv for depth_control
        self.dec_y = nn.Conv2d(2*in_c, out_c, 1)

    def forward(self,zx, zy):
        x1 = torch.cat([zx, self.O2F(zy)], 1)  # (b,c*k,h,w) -> (b,c*2*k,h,w)
        y1 = torch.cat([zy, self.F20(zx)], 1)
        x1 = self.dec_x(x1)  # (b,c*2*k,h,w) -> (b,c,h,w)
        y1 = self.dec_y(y1)
        return x1, y1

class VQVAE_topk_twostream(nn.Module):
    def __init__(
        self, in_channel=(3,2), out_channel=(3,2), channel=128, n_res_block=2,
        n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, k=1):
        super().__init__()

        self.enc_b_1 = Encoder(in_channel[0], channel, n_res_block, n_res_channel, stride=4)
        self.enc_t_1 = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.enc_b_2 = Encoder(in_channel[1], channel, n_res_block, n_res_channel, stride=4)
        self.enc_t_2 = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        #
        self.quantize_t_1 = enc_quan_dec_topk(channel, embed_dim, n_embed, k)
        self.dec_t_1 = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_t_2 = enc_quan_dec_topk(channel, embed_dim, n_embed, k)
        self.dec_t_2 = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        #
        self.bridge_t = middle_unet(in_c=embed_dim, out_c=embed_dim)
        #
        self.quantize_b_1 = enc_quan_dec_topk(embed_dim+channel, embed_dim, n_embed, k) # bottom_level quantize
        self.upsample_t_1 = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec_1 = Decoder(embed_dim+embed_dim, out_channel[0], channel,
            n_res_block, n_res_channel, stride=4,) # 4x upsampling,
        self.quantize_b_2 = enc_quan_dec_topk(embed_dim + channel, embed_dim, n_embed, k)
        self.upsample_t_2 = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec_2 = Decoder(embed_dim + embed_dim, out_channel[1], channel,
                           n_res_block, n_res_channel, stride=4, )
        #
        self.bride_b = middle_unet(in_c=embed_dim, out_c=embed_dim)

    def forward(self, rgb, op):
        quan_tuple, diff = self.encode(rgb, op) # input: (rgb,op)
        dec_1, dec_2 = self.decode(quan_tuple) # dec

        return dec_1, dec_2, diff

    def encode(self, rgb, op):
        #
        enc_b_1 = self.enc_b_1(rgb) #
        enc_t_1 = self.enc_t_1(enc_b_1) #
        enc_b_2 = self.enc_b_2(op)  #
        enc_t_2 = self.enc_t_2(enc_b_2)  #
        #
        quant_t_1, diff_t_1, id_t_1 = self.quantize_t_1(enc_t_1) #
        quant_t_2, diff_t_2, id_t_2 = self.quantize_t_2(enc_t_2)  #
        #
        quant_t_1, quant_t_2 = self.bridge_t(quant_t_1, quant_t_2)
        #
        dec_t_1 = self.dec_t_1(quant_t_1) #
        enc_b_1 = torch.cat([dec_t_1, enc_b_1], 1) #
        dec_t_2 = self.dec_t_2(quant_t_2)  #
        enc_b_2 = torch.cat([dec_t_2, enc_b_2], 1)  #
        #
        quant_b_1, diff_b_1, id_b_1 = self.quantize_b_1(enc_b_1) #
        quant_b_2, diff_b_2, id_b_2 = self.quantize_b_2(enc_b_2)  #
        #
        quant_b_1, quant_b_2 = self.bride_b(quant_b_1, quant_b_2)
        #
        return (quant_t_1, quant_t_2, quant_b_1, quant_b_2), \
               diff_t_1 + diff_t_2 + diff_b_1 + diff_b_2 #

    def decode(self, input_tuple):
        #
        quant_t_1, quant_t_2, quant_b_1, quant_b_2 = input_tuple
        #
        upsample_t_1 = self.upsample_t_1(quant_t_1) #
        quant_1 = torch.cat([upsample_t_1, quant_b_1], 1) #
        dec_1 = self.dec_1(quant_1) # 4x upsample
        #
        upsample_t_2 = self.upsample_t_2(quant_t_2)  #
        quant_2 = torch.cat([upsample_t_2, quant_b_2], 1)  #
        dec_2 = self.dec_2(quant_2)  #

        return dec_1, dec_2

    def decode_code(self, code_t, code_b):
        # quant_t = self.quantize_t.embed_code(code_t)
        # quant_t = quant_t.permute(0, 3, 1, 2)
        # quant_b = self.quantize_b.embed_code(code_b)
        # quant_b = quant_b.permute(0, 3, 1, 2)
        #
        # dec = self.decode(quant_t, quant_b)
        #
        # return dec
        pass #

    def fixed_rgb_op_branch(self):
        fixed_layer_list = [
            self.enc_b_1, self.enc_t_1, self.enc_b_2, self.enc_t_2,
            self.quantize_t_1,self.dec_t_1, self.quantize_t_2, self.dec_t_2,
            self.quantize_b_1, self.upsample_t_1, self.dec_1, self.quantize_b_2,
            self.upsample_t_2, self.dec_2,
        ]
        for layer in fixed_layer_list:
            for p in layer.parameters():
                p.requires_grad = False

def get_twostream(in_channel=(3,2), out_channel=(3,2), channel=128, n_res_block=2,
        n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, k=1):
    return VQVAE_topk_twostream(in_channel=in_channel, out_channel=out_channel,
        channel=channel, n_res_block=n_res_block,
        n_res_channel=n_res_channel, embed_dim=embed_dim,
        n_embed=n_embed, decay=decay, k=k)

def test_get_twostream():
    num_his = 9, 8
    channel = 3, 2
    in_c = num_his[0] * channel[0], num_his[1] * channel[1]
    out_c = channel
    net = get_twostream(in_c, out_c)
    b, c1, h, w = 2, num_his[0] * channel[0], 256, 256
    b, c2, h, w = 2, num_his[1] * channel[1], 256, 256
    rgb,op = torch.randn(b, c1, h, w), torch.randn(b, c2, h, w)
    summary(net, rgb, op)
    # print("ret: ", dec_1.size(), dec_2.size)
    dec_1, dec_2, diff = net(rgb,op)
    print("ret: ", dec_1.size(), dec_2.size())

    '''
                                 Totals                  
    Total params              3.028613M                  
    Trainable params          3.028613M                 
    Non-trainable params            0.0            
    Mult-Adds             13.363052544G
    '''

# =================================================================== #

class test_load_model_ckpt():

    def __init__(self, model, rgb_model_path, op_model_path, device):
        self.model = model
        self.rgb_model_path = rgb_model_path
        self.op_model_path = op_model_path
        self.device = device

    def test_1(self):
        vqvae_topk = get_vqvae_topk(in_channel=3, out_channel=3, channel=128,
                  n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=256, decay=0.99, k=2)
        twostream = get_twostream(in_channel=(3,2), out_channel=(3,2), channel=128,
                  n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=256, decay=0.99, k=2)
        print("vqvae_topk",vqvae_topk.state_dict().keys())
        print("\n")
        print("vqvae_topk",twostream.state_dict().keys())
        pass

    def test_2(self):
        rgb_vqvae_topk = get_vqvae_topk(in_channel=3, out_channel=3, channel=128,
                  n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, k=2)
        twostream = get_twostream(in_channel=(3,2), out_channel=(3,2), channel=128,
                  n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99, k=2)
        pretrained_key_list = ['enc_b', 'enc_t', 'quantize_t', 'dec_t', 'quantize_b',
                                    'upsample_t', 'dec']
        rgb_part_key_list = ["{}_1".format(s) for s in pretrained_key_list]
        op_part_key_list = ["{}_2".format(s) for s in pretrained_key_list]
        #
        def test_warpper_key():
            key = "enc_b.blocks.0.weight" # from pretrain_dict: xxx.yyy
            start,end = key.split('.', 1)
            rgb_part_key = start + '_.' + end
            op_part_key = start + '_2.' + end
        #
        def get_wapper_key(key, add_s):
            start, end = key.split('.', 1)
            wapped_key = start + '_{}.'.format(add_s) + end

            return wapped_key

    def test_3(self):
        model = self.model
        rgb_model_path = self.rgb_model_path
        op_model_path = self.op_model_path
        device = self.device

        rgb_pretrained_dict = torch.load(rgb_model_path,map_location=device)
        op_pretrained_dict = torch.load(op_model_path, map_location=device)
        #
        model_dict = model.state_dict()
        #
        def get_wapper_key(key, add_s):
            start, end = key.split('.', 1)
            wapped_key = start + '_{}.'.format(add_s) + end

            return wapped_key
        #
        rgb_state_dict = {get_wapper_key(k, '1'): v for k, v in rgb_pretrained_dict.items() if
                          get_wapper_key(k, '1') in model_dict.keys()}
        op_state_dict = {get_wapper_key(k, '2'): v for k, v in op_pretrained_dict.items() if
                         get_wapper_key(k, '2') in model_dict.keys()}
        #
        model_dict.update(rgb_state_dict)
        model_dict.update(op_state_dict)
        model.load_state_dict(model_dict)

        '''
        for i in range(65):
            dict_new[ new_list[i] ] = dict_trained[ trained_list[i] ]

        net.load_state_dict(dict_new)
        '''

    def test_4(self):
        model = self.model
        # optimizer = self.optimizer  # onlf for generator
        model.fixed_rgb_op_branch() #
        torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        for p in model.parameters():
            print("p: ", p)

def test_load_model():
    model =get_twostream(in_channel=(3*9,2*8), out_channel=(3*1,2*1), channel=128,
                  n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=256, decay=0.99, k=2)
    lr_g = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_g)
    rgb_model_path = ""
    op_model_path = ""
    device = torch.device("cpu")
    helper = test_load_model_ckpt(model, rgb_model_path, op_model_path, device)
    #
    helper.test_3() #
    #
    helper.test_4()




if __name__ == '__main__':
    # test_vqvae()
    # test_vqvae_topk_res()
    # test_get_twostream()
    test_load_model()

    # python -m pyt_vad_topk_mem_cons.models.vqvae

