import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
import torch.nn.functional as F


@HEADS.register_module
class TCMaskHead(nn.Module):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 fc_conv=True,
                 training = True,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(TCMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.training = training
        self.relu = nn.ReLU(inplace=True)
        self.fc_conv = fc_conv

        self.convs = nn.ModuleList()
        #self.fc_convs = nn.ModuleList()
        #self.fc_list = nn.ModuleList()
        for i in range(self.num_convs-1):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        padding = (self.conv_kernel_size - 1) // 2
        self.fc_conv_out_channels=(self.conv_out_channels)//2
        self.convs_last=nn.Conv2d(self.conv_out_channels, self.conv_out_channels, self.conv_kernel_size, 1, padding=padding, dilation=1,groups=1, bias=False)
        self.fc_convs_1=nn.Conv2d(self.conv_out_channels, self.conv_out_channels, self.conv_kernel_size, 1, padding=padding, dilation=1,groups=1, bias=False)
        self.fc_convs_2=nn.Conv2d(self.conv_out_channels, self.fc_conv_out_channels, self.conv_kernel_size, 1, padding=padding, dilation=1,groups=1, bias=False)
        fc_in_channels = self.fc_conv_out_channels * self.roi_feat_area
        linear_channel = self.roi_feat_area*4
        self.mask_fc = nn.Linear(fc_in_channels, linear_channel)#fc
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        for m in [self.mask_fc]:
            if m is None:
                continue
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            xc_3 = conv(x)
        x = self.convs_last(xc_3)
        if self.upsample is not None:
            xc = self.upsample(x)
            if self.upsample_method == 'deconv':
                xc = self.relu(xc)
        mask_pred_conv = self.conv_logits(xc)
        #print('mask_pred_conv.shape:',mask_pred_conv.shape)
        if self.fc_conv:
            xf = self.fc_convs_1(xc_3)
            xf = self.fc_convs_2(xf)
            xf = xf.view(xf.size(0), -1)
            xf = self.relu(self.mask_fc(xf))
            xf = xf.view(-1, 1, self.roi_feat_size[0]*2, self.roi_feat_size[1]*2)
            xf = xf.repeat(1, self.num_classes, 1, 1)
        if not self.training:
            x = F.sigmoid(x)
        #mask_pred = mask_pred_conv + xf
        return mask_pred_conv

    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred',))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, img_shape, scale_factor,
                      rescale, return_rect=False):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().detach().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        if return_rect:
            rects = []
        else:
            cls_segms = [[] for _ in range(self.num_classes - 1)]

        bboxes = det_bboxes.cpu().detach().numpy()[:, :4]
        labels = det_labels.cpu().detach().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h, img_w = img_shape[:2]
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)
            try:
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            except:
                print(bbox, img_h, img_w)
                exit()

            if return_rect:
                cnt = np.stack(np.where(im_mask == 1)).T
                rect = cv2.boxPoints(cv2.minAreaRect(cnt))
                rect = np.array(rect)[:, ::-1].reshape(-1)
                rects.append(rect)

            else:
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[label - 1].append(rle)

        if return_rect:
            return rects
        return cls_segms
