import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
import Polygon as plg

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, poly_nms, tensor2imgs
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from pytorch_transformers import BertTokenizer, BertConfig, BertModel
from mmcv.image import imread, imwrite
from PIL import Image, ImageDraw, ImageFont


@DETECTORS.register_module
class Truck_char(BaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 text_bbox_roi_extractor,
                 text_bbox_head,
                 text_mask_roi_extractor,
                 text_mask_head,
                 char_bbox_roi_extractor,
                 char_bbox_head,
                 crm_cfg,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Truck_char, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.rpn_head = builder.build_head(rpn_head)

        # text detection module
        self.text_bbox_roi_extractor = builder.build_roi_extractor(text_bbox_roi_extractor)
        self.text_bbox_head = builder.build_head(text_bbox_head)
        self.text_mask_roi_extractor = builder.build_roi_extractor(text_mask_roi_extractor)
        self.text_mask_head = builder.build_head(text_mask_head)

        # character-based recognition module
        self.char_bbox_roi_extractor = builder.build_roi_extractor(char_bbox_roi_extractor)
        self.char_bbox_head = builder.build_head(char_bbox_head)
        self.crm_cfg = crm_cfg
        self.label2char = mmcv.load(crm_cfg.char_dict_file)['label2char']

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_lm_cfg(self):
        return hasattr(self, 'lm_cfg') and self.lm_cfg is not None

    def init_weights(self, pretrained=None):
        super(Truck_char, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()
        self.rpn_head.init_weights()

        self.text_bbox_roi_extractor.init_weights()
        self.text_bbox_head.init_weights()
        self.text_mask_roi_extractor.init_weights()
        self.text_mask_head.init_weights()

        self.char_bbox_roi_extractor.init_weights()
        self.char_bbox_head.init_weights()


    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)

        return x

    def forward_dummy(self, img):
        pass

    def get_text_content(self, rects, char_bboxes, char_labels, class_names):
        rects = [np.array(rect).reshape(-1, 2) for rect in rects]
        char_bboxes = char_bboxes.cpu().numpy()
        char_labels = char_labels.cpu().numpy()
        char_rects, char_scores, char_heights = [], [], []
        for char_bbox in char_bboxes:
            char_rect = [char_bbox[0], char_bbox[1],
                         char_bbox[2], char_bbox[1],
                         char_bbox[2], char_bbox[3],
                         char_bbox[0], char_bbox[3]]
            char_heights.append(char_bbox[3]-char_bbox[1])
            char_rect = plg.Polygon(np.array(char_rect).reshape(-1, 2))
            char_rects.append(char_rect)
            char_scores.append(char_bbox[-1])
        char_scores = np.array(char_scores)
        char_heights = np.array(char_heights)
        char_centers = (char_bboxes[:, :2] + char_bboxes[:, 2:4]) / 2.0

        x_rank = np.argsort(char_centers[:, 0], 0)
        char_rects_x = [char_rects[i] for i in x_rank]
        char_heights_x = char_heights[x_rank]
        char_scores_x = char_scores[x_rank]
        char_labels_x = char_labels[x_rank]

        y_rank = np.argsort(char_centers[:, 1], 0)
        char_rects_y = [char_rects[i] for i in y_rank]
        char_heights_y = char_heights[y_rank]
        char_scores_y = char_scores[y_rank]
        char_labels_y = char_labels[y_rank]

        def get_intersection(pa, pb):
            pInt = pa & pb
            if len(pInt) == 0:
                return 0
            return pInt.area()

        def assign_char(rect_):
            h, w = (np.max(rect_, 0) - np.min(rect_, 0)).tolist()
            if h <= w:
                char_rects_ = char_rects_y
                char_scores_ = char_scores_y
                char_labels_ = char_labels_y
                char_heights_ = char_heights_y
            else:
                char_rects_ = char_rects_x
                char_scores_ = char_scores_x
                char_labels_ = char_labels_x
                char_heights_ = char_heights_x
            label_ = []
            score_ = []
            height_ = []
            rect_ = plg.Polygon(rect_)
            for i, char_score_, char_rect_, char_height_ in zip(char_labels_, char_scores_, char_rects_, char_heights_):
                inter = get_intersection(rect_, char_rect_)
                iou = inter / (char_rect_.area() + 1e-6)
                if iou > self.crm_cfg.char_assign_iou:
                    label_.append(i)
                    score_.append(char_score_)
                    height_.append(char_height_)
            score_h = 1
            for i in range(len(height_)-1):
                distence = min(2*abs(height_[i]-height_[i+1])/(height_[i]+height_[i+1]),1)#add
                score_hi = 1/(1+distence)
                score_h=min(score_h,score_hi)
            M_value = 0
            number = range(15,25)
            province = [1937,1108]
            letter = range(38,64)
            length = len(label_)
            if length>=2:
                if (label_[0] in province) and (label_[1] in letter):
                    for j in range(2,length):
                            if (label_[j] in number) or (label_[j] in letter):
                                continue
                            else:
                                M_value = 1
                else:
                    M_value = 1
            else:
                M_value = 1
            socre_relu = score_h*(0.8**(abs(length-7)+M_value))
            label_ = np.array(label_)
            score_ = np.array(score_)
            return label_, score_, socre_relu

        texts = []
        text_scores = []
        score_re = []
        for rect in rects:
            chars, scores, socre_relu = assign_char(rect)
            text = ''
            text_score = []
            for char, score in zip(chars, scores):
                char = int(char)
                text += class_names[char]
                text_score.append(score)
            texts.append(text)
            score_re.append(socre_relu)#add
            text_scores.append(np.mean(text_score) if len(text_score) > 0 else 0.0)
        text_scores = np.array(text_scores)
        score_re = np.array(score_re,dtype='float32')#add

        return texts, text_scores, score_re

    def forward_train(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None):
        losses = self.forward_train_vis(img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, proposals)
        return losses

    def forward_train_vis(self, img, img_meta, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None,
                          proposals=None):
        losses = dict()

        # separate gt
        num_imgs = img.size(0)
        text_gt_bboxes = []
        text_gt_labels = []
        char_gt_bboxes = []
        char_gt_labels = []
        for img_i in range(num_imgs):
            text_num = gt_masks[img_i].shape[0]
            # text line gt
            text_gt_bboxes.append(gt_bboxes[img_i][:text_num])
            text_gt_labels.append(gt_labels[img_i][:text_num])
            # character gt
            char_gt_bboxes.append(gt_bboxes[img_i][text_num:])
            char_gt_labels.append(gt_labels[img_i][text_num:])

        x = self.extract_feat(img)

        # RPN forward and loss
        rpn_outs = self.rpn_head(x)
        stage_num = len(rpn_outs[0])
        # text line proposals
        text_rpn_outs = ([], [])
        for stage_i in range(stage_num):
            text_rpn_outs[0].append(rpn_outs[0][stage_i])
            text_rpn_outs[1].append(rpn_outs[1][stage_i])
        text_rpn_loss_inputs = text_rpn_outs + (text_gt_bboxes, img_meta, self.train_cfg.rpn)
        text_rpn_losses = self.rpn_head.loss(*text_rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, type='text')
        losses.update(text_rpn_losses)
        text_proposal_inputs = text_rpn_outs + (img_meta, self.train_cfg.text_rpn_proposal)
        text_proposal_list = self.rpn_head.get_bboxes(*text_proposal_inputs, type='text')
        # character proposals
        char_rpn_outs = ([], [])
        for stage_i in range(stage_num):
            char_rpn_outs[0].append(rpn_outs[2][stage_i])
            char_rpn_outs[1].append(rpn_outs[3][stage_i])
        char_rpn_loss_inputs = char_rpn_outs + (char_gt_bboxes, img_meta, self.train_cfg.rpn)
        char_rpn_losses = self.rpn_head.loss(*char_rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, type='char')
        losses.update(char_rpn_losses)
        char_proposal_inputs = char_rpn_outs + (img_meta, self.train_cfg.char_rpn_proposal)
        char_proposal_list = self.rpn_head.get_bboxes(*char_proposal_inputs, type='char')

        # assign gts and sample proposals
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
            self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        text_sampling_results, char_sampling_results = [], []
        for i in range(num_imgs):
            # sample text line proposals
            text_assign_result = bbox_assigner.assign(
                text_proposal_list[i], text_gt_bboxes[i], gt_bboxes_ignore[i], text_gt_labels[i])
            text_sampling_result = bbox_sampler.sample(
                text_assign_result, text_proposal_list[i], text_gt_bboxes[i], text_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            text_sampling_results.append(text_sampling_result)
            # sample character proposals
            char_assign_result = bbox_assigner.assign(
                char_proposal_list[i], char_gt_bboxes[i], gt_bboxes_ignore[i], char_gt_labels[i])
            char_sampling_result = bbox_sampler.sample(
                char_assign_result, char_proposal_list[i], char_gt_bboxes[i], char_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            char_sampling_results.append(char_sampling_result)

        # text detection module
        text_rois = bbox2roi([res.bboxes for res in text_sampling_results])
        text_bbox_feats = self.text_bbox_roi_extractor(x[:self.text_bbox_roi_extractor.num_inputs], text_rois)
        text_cls_score, text_bbox_pred = self.text_bbox_head(text_bbox_feats)
        text_bbox_targets = self.text_bbox_head.get_target(
            text_sampling_results, text_gt_bboxes, text_gt_labels, self.train_cfg.rcnn)
        text_loss_bbox = self.text_bbox_head.loss(text_cls_score, text_bbox_pred, *text_bbox_targets, type='text')
        losses.update(text_loss_bbox)
        pos_rois = bbox2roi([res.pos_bboxes for res in text_sampling_results])
        text_mask_feats = self.text_mask_roi_extractor(x[:self.text_mask_roi_extractor.num_inputs], pos_rois)
        mask_pred = self.text_mask_head(text_mask_feats)
        mask_targets = self.text_mask_head.get_target(text_sampling_results, gt_masks, self.train_cfg.rcnn)
        pos_labels = torch.cat([res.pos_gt_labels for res in text_sampling_results])
        loss_mask = self.text_mask_head.loss(mask_pred, mask_targets, pos_labels)
        losses.update(loss_mask)

        # character-based recognition module
        char_rois = bbox2roi([res.bboxes for res in char_sampling_results])
        char_bbox_feats = self.char_bbox_roi_extractor(x[:self.char_bbox_roi_extractor.num_inputs], char_rois)
        char_cls_score, char_bbox_pred = self.char_bbox_head(char_bbox_feats)  # the input may be a tuple
        char_bbox_targets = self.char_bbox_head.get_target(
            char_sampling_results, char_gt_bboxes, char_gt_labels, self.train_cfg.rcnn)
        char_loss_bbox = self.char_bbox_head.loss(char_cls_score, char_bbox_pred, *char_bbox_targets, type='char')
        losses.update(char_loss_bbox)

        # print(losses)
        # exit()

        return losses


    def simple_test_rpn(self, x, img_meta, text_rpn_test_cfg, char_rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        stage_num = len(rpn_outs[0])
        text_rpn_outs = ([], [])
        for stage_i in range(stage_num):
            text_rpn_outs[0].append(rpn_outs[0][stage_i])
            text_rpn_outs[1].append(rpn_outs[1][stage_i])
        text_proposal_inputs = text_rpn_outs + (img_meta, text_rpn_test_cfg)
        text_proposal_list = self.rpn_head.get_bboxes(*text_proposal_inputs, type='text')
        char_rpn_outs = ([], [])
        for stage_i in range(stage_num):
            char_rpn_outs[0].append(rpn_outs[2][stage_i])
            char_rpn_outs[1].append(rpn_outs[3][stage_i])
        char_proposal_inputs = char_rpn_outs + (img_meta, char_rpn_test_cfg)
        char_proposal_list = self.rpn_head.get_bboxes(*char_proposal_inputs, type='char')

        return text_proposal_list, char_proposal_list

    def simple_test_text_bboxes(self, x, img_meta, proposals, text_rcnn_test_cfg, rescale=False):
        rois = bbox2roi(proposals)
        roi_feats = self.text_bbox_roi_extractor(x[:len(self.text_bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.text_bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels, nms_inds = self.text_bbox_head.get_det_bboxes(  # this place can be refine
            rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=text_rcnn_test_cfg)

        return det_bboxes, det_labels

    def simple_test_text_mask(self, x, img_meta, det_bboxes, det_labels, rescale=False):
        ori_shape = img_meta[0]['ori_shape']
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.text_mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.text_mask_roi_extractor(x[:len(self.text_mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.text_mask_head(mask_feats)
            segm_result = self.text_mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.text_rcnn, ori_shape, img_shape, scale_factor,
                rescale=rescale, return_rect=True)

        return segm_result

    def simple_test_char_bboxes(self, x, img_meta, proposals, char_rcnn_test_cfg, rescale=False):
        rois = bbox2roi(proposals)
        roi_feats = self.char_bbox_roi_extractor(x[:self.char_bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.char_bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels, nms_inds = self.char_bbox_head.get_det_bboxes(
            rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=rescale, cfg=char_rcnn_test_cfg)

        return det_bboxes, det_labels

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        with torch.no_grad():
            x = self.extract_feat(img)

            # two-stream rpn
            rpn_outs = self.simple_test_rpn(x, img_meta, self.test_cfg.text_rpn, self.test_cfg.char_rpn) \
                if proposals is None else proposals
            # text line proposals
            text_proposal_list = rpn_outs[0]
            # character proposals
            char_proposal_list = rpn_outs[1]

            # text detection module
            text_det_bboxes, text_det_labels = self.simple_test_text_bboxes(
                x, img_meta, text_proposal_list, self.test_cfg.text_rcnn, rescale=rescale)
            rects = self.simple_test_text_mask(x, img_meta, text_det_bboxes, text_det_labels, rescale=rescale)

            # character-based recognition module
            char_det_bboxes, char_det_labels = self.simple_test_char_bboxes(
                x, img_meta, char_proposal_list, self.test_cfg.char_rcnn, rescale=rescale)
            char_bbox_results = bbox2result(char_det_bboxes, char_det_labels, self.char_bbox_head.num_classes)

            # match-assemble algorithm
            texts, text_scores, score_re = self.get_text_content(rects, char_det_bboxes, char_det_labels, self.label2char)
            scores = ((1 - 0.3) * text_det_bboxes[:, 4]).cpu().numpy() + 0.3 * score_re
            rects, scores, inds = poly_nms(rects, scores, self.test_cfg.poly_iou, 0.0, return_ind=True)
                
            texts = [texts[i] for i in inds]

            return rects, scores, char_bbox_results, texts

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        pass

    def imshow_det_bboxes(self,
                          img,
                          bboxes,
                          texts,
                          out_file):
        # draw text
        def change_cv2_draw(image, strs, local, sizes, color):
            cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)
            font = ImageFont.truetype('resource/simsun.ttc', sizes, encoding="utf-8")
            draw.text(local, strs, color, font=font)
            image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            return image

        img = imread(img)

        bbox_color = (255, 0, 0)
        text_color = (0, 0, 255)
        for i in range(len(texts)):
            bbox = bboxes[i].astype(np.int32)
            text = texts[i]
            cv2.drawContours(img, [bbox], -1, bbox_color, 2)
            tl = np.min(bbox, 0)
            img = change_cv2_draw(img, text, (tl[0], tl[1]), 20, text_color)
        imwrite(img, out_file)

    def show_result(self, data, result, score_thr=0.5):
        rects, scores, char_bbox_results, texts = result

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)
        img_name = img_metas[0]['filename'].split('/')[-1]

        vis_dir = './vis/'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        h, w, _ = img_metas[0]['img_shape']
        img_show = imgs[0][:h, :w, :]

        rects = np.array(rects)
        scores = np.array(scores)
        ind = scores > score_thr
        rects = rects[ind, :, :]
        texts = [texts[i] for i in range(len(texts)) if ind[i]]
        self.imshow_det_bboxes(
            img_show,
            rects,
            texts,
            out_file=vis_dir + img_name)
