# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------


import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import einsum, nn
from functools import partial
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import trunc_normal_
from timm.models.registry import register_model
from timm.models.layers import DropPath, Mlp

import warnings
warnings.filterwarnings("ignore")
from torch.nn import functional as F
from torchvision import models


__all__ = [
    "mpvit_tiny",
    "mpvit_xsmall",
    "mpvit_small",
    "mpvit_base",
]


def _cfg_mpvit(url="", **kwargs):  #url: 一个可选参数，用于指定预训练模型的下载链接。
    """configuration of mpvit."""
    return {
        "url": url,
        #"num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        #"classifier": "head",
        **kwargs,
    }

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.v1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.v2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.v3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 256, 3, padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 384, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.init_param()

    def forward(self, x1, x2, x3):
        x1 = self.v1(x1)
        x2 = self.v2(x2)
        x3 = self.v3(x3)
        x = x1 + x2 + x3
        y1 = self.stage1(x)
        y2 = self.stage2(x)
        y3 = self.stage3(x)
        y4 = self.stage4(x)
        y = torch.cat((y1, y2, y3), dim=1) + y4
        y = self.res(y)
        return y

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a.MLP) class."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """foward function"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    """Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        self.bn = norm_layer(out_ch)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
        )

    def forward(self, x):
        """foward function"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """Depthwise Separable Convolution with BN module."""
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):
        """
        foward function
        """
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""
    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""
    def __init__(self, embed_dim, num_path=4, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=2 if isPool and idx == 0 else 1,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        """foward function"""
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.

    Note: This module is similar to the conditional position encoding in CPVT.
    """
    def __init__(self, dim, k=3):
        """init function"""
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding
    class."""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        """foward function"""
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.
        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        factor_att = einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    """Multi-Head Convolutional self-Attention block."""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        shared_cpe=None,
        shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        """foward function"""
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x



class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""
    def __init__(
        self,
        dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3, 
        drop_path_list=[],
        qk_scale=None,
        crpe_window={
            3: 2,
            5: 3,
            7: 3
        },

    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads,
                                  h=num_heads,
                                  window=crpe_window)
        self.MHCA_layers = nn.ModuleList([
            MHCABlock(
                dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_list[idx],
                qk_scale=qk_scale,
                shared_cpe=self.cpe,
                shared_crpe=self.crpe,
            ) for idx in range(self.num_layers)
        ])


    def forward(self, x, size):
        """foward function"""
        H, W = size
        B = x.shape[0]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        drop_path_list=[],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([
            MHCAEncoder(
                embed_dim,
                num_layers,
                num_heads,
                mlp_ratio,
                drop_path_list=drop_path_list,
            ) for _ in range(num_path)
        ])
        

        self.InvRes = ResBlock(in_features=embed_dim, out_features=embed_dim)
        self.aggregate = Conv2d_BN(embed_dim * (num_path + 1),
                                   out_embed_dim,
                                   act_layer=nn.Hardswish)

    def forward(self, inputs):
        """foward function"""
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out



def dpr_generator(drop_path_rate, num_layers, num_stages):
    """Generate drop path rate list following linear decay rule."""
    dpr_list = [
        x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
    ]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur:cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)

        # 动态注册A0到设备（无需硬编码）
        self.register_buffer('A0', torch.eye(hide_channel))  # 自动跟随模型所在的设备

        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttMeanMax, self).__init__()
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  #特征进来先拷贝了一份
        n, c, h, w = x.size()

        # Mean pooling branch
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)  #拼接h方向和w方向的平均值
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # Max pooling branch
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # Apply attention
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # Expand to original shape
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # Combine outputs
        attention_mean = identity * x_w_mean * x_h_mean
        attention_max = identity * x_w_max * x_h_max

        # Sum the attention outputs
        return attention_mean + attention_max

#MaskAttention
class MaskAttention(nn.Module):
    def __init__(self, channels, size):
        super(MaskAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm([channels])

    def forward(self, x):
        b, c, h, w = x.size()

        # 检查通道数是否匹配
        if c != self.channels:
            raise ValueError(f"Input channel size {c} does not match initialized channel size {self.channels}.")

        # 重塑并转置输入
        x = x.view(b, c, h * w).permute(0, 2, 1)

        # 计算 Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5)

        # 动态创建 mask
        binary_mask = torch.randint(0, 2, (b, h, w), device=x.device)
        binary_mask = binary_mask.view(b, -1)
        processed_mask = torch.where(
            binary_mask > 0.5,
            torch.tensor(0.0, device=x.device),
            torch.tensor(-float('inf'), device=x.device)
        )
        mask = processed_mask.unsqueeze(1).expand(-1, h * w, -1)

        # 确保 mask 与 scores 尺寸匹配
        if mask.size() != scores.size():
            # 使用插值调整尺寸
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=scores.size()[1:],
                mode='nearest'
            ).squeeze(0)

        # 应用 mask
        scores = scores + mask

        # 计算注意力权重和输出
        attention_weight = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weight, V)
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        # 重塑为原始尺寸
        return attention_output.view(b, c, h, w)


class AttentionFusion(nn.Module):
    def __init__(self, channels=288):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f3, f4):
    
        f4_up = F.interpolate(f4, size=f3.size()[2:], mode='bilinear', align_corners=True)
        concat = torch.cat([f3, f4_up], dim=1)
        attention = self.attention(concat)
        fused = f3 * attention + f4_up * (1 - attention)

        return fused

class MPViT(nn.Module):
    """Multi-Path ViT class."""
    def __init__(
        self,
        num_stages=4,
        num_layers=[1, 1, 1, 1],
        mlp_ratios=[8, 8, 4, 4],
        in_chans=3,
        num_classes=1000,
        num_path=[4, 4, 4, 4],
        embed_dims=[64, 128, 256, 512],
        num_heads=[8, 8, 8, 8],
        drop_path_rate=0.0,
        out_features=None,
        norm="FrozenBN",
        **kwargs,
    ):
        super().__init__()

        self.num_stages = num_stages

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=False if idx == 0 else True,
            ) for idx in range(self.num_stages)
        ])

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList([
            MHCA_stage(
                embed_dims[idx],
                embed_dims[idx + 1]
                if not (idx + 1) == self.num_stages else embed_dims[idx],
                num_layers[idx],
                num_heads[idx],
                mlp_ratios[idx],
                num_path[idx],
                drop_path_list=dpr[idx],
            ) for idx in range(self.num_stages)
        ])


        # 添加回归头
        self.res = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(288, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )
       #【8, 1, 32, 32】

        # 初始化通道数调整层
        self.apply(self._init_weights)

        self.EMA = EMA(channels=64, factor=8)
        self.AGCA = AGCA(in_channel=216, ratio=4)
        self.coord_att = CoordAttMeanMax(inp=288, oup=288)
        self.maskattention = MaskAttention(channels=288, size=(8, 8))

        self.fusion = AttentionFusion(channels=288)

    def _init_weights(self, m):
        """initialization"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """forward feature function"""

        # x's shape : [B, C, H, W]

        x = self.stem(x)  # Shape : [B, 64, H/4, W/4]

        x = self.EMA(x)

        features = []

        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)   # 补丁嵌入
            x = self.mhca_stages[idx](att_inputs)          # 多头交叉注意力

            # if idx == 2:
            #     x = self.AGCA(x)

            # if idx == 4:
            #     x = self.maskattention(x)

            features.append(x)                             # 保存当前阶段特征

        return features

    def forward(self, x):
        """foward function"""
        #print(f"x没经过self.forward_features的通道数为：", x.shape)

        features = self.forward_features(x)
        #print(f"features[-1]:", features[-1].shape)

        # 打印所有阶段的形状
        # for i, feat in enumerate(features):
        #     print(f"Stage {i} shape: {feat.shape}")  # e.g., [B, C, H, W] 或 [B, num_patches, D]]

        if len(features) >= 2:
            f3 = features[-2]
            f4 = self.maskattention(features[-1])
        fused = self.fusion(f3, f4)

        out = self.coord_att(fused)
        mu = fused + out
        mu = self.res(mu)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


@register_model
def mpvit_tiny(pretrained=True, **kwargs):
    """mpvit_tiny :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 96, 176, 216]
    - MLP_ratio : 2
    Number of params: 5843736
    FLOPs : 1654163812
    Activations : 16641952
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    if pretrained:
        checkpoint = torch.load('/home/ln/A/CJQ/CCTrans-main/pre_weight/mpvit_tiny.pth') #预训练权重路径
        # print(model)
        model.load_state_dict(checkpoint, strict=True)
        print("load transformer pretrained")
    return model


@register_model
def mpvit_xsmall(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


@register_model
def mpvit_small(pretrained=True,**kwargs):
    """mpvit_small :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 6, 3]
    - #channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()


    if pretrained:
        checkpoint = torch.load('/home/ln/A/CJQ/CCTrans-main/pre_weight/mpvit_small.pth')  # 预训练权重路径
        # for key, value in checkpoint.items():
        #     print(key)

        # print(model)
        model.load_state_dict(checkpoint['model'], strict=False)
        print("load transformer pretrained")

    return model


@register_model
def mpvit_base(**kwargs):
    """mpvit_base :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 8, 3]
    - #channels : [128, 224, 368, 480]
    MLP_ratio : 4
    Number of params: 74845976
    FLOPs : 16445326240
    Activations : 60204392
    """

    model = MPViT(
        img_size=224,
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


if __name__ == "__main__":
    model = mpvit_small()
    model.eval()
    inputs = torch.randn(1, 3, 224, 224)
    model(inputs)


    #from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis

    #flops = FlopCountAnalysis(model, inputs)
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #acts = ActivationCountAnalysis(model, inputs)

    # print(f"total flops : {flops.total()}")
    # print(f"total activations: {acts.total()}")
    # print(f"number of parameter: {param}")
