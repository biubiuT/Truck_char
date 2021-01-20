import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,#每个尺度的输入通道数, 也是 backbone 的输出通道数
                 out_channels,#fpn 的输出通道数, 所有尺度的输出通道数相同, 都是一个值
                 num_outs,#输出 stage 的个数.(可以附加额外的层, num_outs 不一定等于 in_channels)
                 start_level=0,#使用 backbone 的起始 stage 索引, 默认为 0.
                 end_level=-1,#使用 backbone 的终止 stage 索引,默认为 -1, 代表到最后一层(包括)全使用
                 add_extra_convs=False,# 可以是 bool 或 str,bool 代表是否添加额外的层.(默认值: False),str  需要指定 extra convs 的输入的 feature map 的来源
                 extra_convs_on_inputs=True,#True  等同于 `add_extra_convs='on_input,False 等同于 `add_extra_convs='on_output
                 relu_before_extra_convs=False,#是否在 extra conv 前使用 relu
                 no_norm_on_lateral=False,#是否对 lateral 使用 bn
                 conv_cfg=None,#构建 conv 层的 config 字典
                 norm_cfg=None,#构建  bn  层的 config 字典
                 activation=None,
                **kwargs):#构建 activation  层的 config 字典
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels# [256, 512, 1024, 2048]
        self.out_channels = out_channels# 256
        self.num_ins = len(in_channels)# 4
        self.num_outs = num_outs# 5
        self.activation = activation# False
        self.relu_before_extra_convs = relu_before_extra_convs# False
        self.no_norm_on_lateral = no_norm_on_lateral# False
        self.fp16_enabled = False# False

        if end_level == -1:#end_level 是对 backbone 输出的尺度中使用的最后一个尺度的索引,如果是 -1 表示使用 backbone 最后一个 feature map, 作为最终的索引
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level#因为可能还有 extra conv 所以存在 num_outs > num_ins - start_level 的情况
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level#如果 end_level < inputs, 说明不使用 backbone 全部的尺度, 并且不会提供额外的层.
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.pam = _PositionAttentionModule(out_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):# 构建 lateral conv 和 fpn conv
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(#输出卷积: 3×3, C=256, P=1
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)#lateral_convs就是一个list，list里面存的就是1x1的卷积
            self.fpn_convs.append(fpn_conv)#fpn_convs也是一个list，存的是最后的3x3卷积

        # add extra conv layers (e.g., RetinaNet)   add_extra_convs 为 True 或 str 时才添加 extra_convs
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(# extra conv 是 3x3 步长为 2, padding 为 1 的卷积
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')# 使用 xavier 初始化卷积层



    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)#进行1x1卷积(lateral_conv):在backbone阶段每个block输出的featuremap经过1x1的卷积
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        # 自上至下将 laterals 里面的结果更新为经过 top-down 的结果.
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(#计算下层 feature map 大小,与下一个block输出进行插值后相加
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            (self.pam(self.fpn_convs[i](laterals[i]))+self.cam(self.fpn_convs[i](laterals[i]))) for i in range(used_backbone_levels)#最后过3x3的卷积(fpn_conv),输出outs，这里的outs同样也是一个list
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out
