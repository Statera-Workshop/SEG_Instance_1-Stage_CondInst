"""
FCOS builder (Image task only).

Paper: Tian, Shen, Chen, He, "FCOS: Fully Convolutional One-Stage Object Detection" (ICCV 2019) and the
       improved version "FCOS: A Simple and Strong Anchor-free Object Detector" (TPAMI 2020, arXiv 2006.09214).
Code : detectron2 modeling/meta_arch/fcos.py (FCOS, FCOSHead) + configs/common/models/fcos.py; detectron2 is the
       official-equivalent source approved on 2026-10-04 (the authors' tianzhi0549/FCOS needs maskrcnn-benchmark
       C++ ops). Backbone / FPN classes shared with detectors/retinanet (torchvision ResNet-50, detectron2 FPN).

    backbone : ResNet-50 (res3-res5)                       neck : FPN P3-P5, P6/P7 from P5 (fcos.py config)
    head     : RetinaNetHead with GN, A = 1, + centerness on the box tower (FCOSHead)
    targets  : one point per location (anchor = stride x stride box), center sampling radius 1.5,
               scale ranges [0, 64], [64, 128], [128, 256], [256, 512], [512, inf] (4x / 8x stride), smallest area wins
    loss     : focal (0.25, 2) + GIoU on ltrb / stride + BCE centerness, each / EMA(#pos) (initial 300)
    score    : sqrt(sigmoid(cls) * sigmoid(ctr))  ->  raw objectness = sqrt(ctr), class prob = sqrt(cls),
               so the shared score obj * cls equals the FCOS score exactly.

[Unified] as RetinaNet (detectors/retinanet/build.py): no ImageNet weights, trainable BN, common init (conv
weights only, GN / prior bias kept), no mean / std, 608 input, Adam 1e-4, common post-processing (instead of
score 0.2, top-1000, NMS 0.6), PreciseBN before evaluation (common/precise_bn.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.retinanet.box_ops import AnchorGenerator
from detectors.retinanet.fpn import build_retinanet_fpn
from detectors.retinanet.loss import sigmoid_focal_loss
from detectors.retinanet.resnet import build_resnet
from detectors.retinanet.retinanet_head import permute_to_N_HWA_K

STRIDES = (8, 16, 32, 64, 128)


class FCOSHead(nn.Module):
    """detectron2 RetinaNetHead(norm="GN", num_anchors=1) + FCOSHead.ctrness"""

    def __init__(self, in_channels, num_classes, num_convs=4, prior_prob=0.01, norm="GN"):
        super().__init__()
        cls_subnet, bbox_subnet = [], []
        for _ in range(num_convs):
            cls_subnet += [nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.GroupNorm(32, in_channels), nn.ReLU()]
            bbox_subnet += [nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.GroupNorm(32, in_channels), nn.ReLU()]
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, 1, 1)
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        torch.nn.init.constant_(self.cls_score.bias, -math.log((1 - prior_prob) / prior_prob))
        self.ctrness = nn.Conv2d(in_channels, 1, 3, 1, 1)
        torch.nn.init.normal_(self.ctrness.weight, std=0.01)
        torch.nn.init.constant_(self.ctrness.bias, 0)

    def forward(self, features):
        logits, bbox_reg, ctrness = [], [], []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_feature = self.bbox_subnet(feature)
            bbox_reg.append(self.bbox_pred(bbox_feature))
            ctrness.append(self.ctrness(bbox_feature))
        return logits, bbox_reg, ctrness


class Box2BoxTransformLinear:
    """detectron2 Box2BoxTransformLinear(normalize_by_size=True): ltrb distances / anchor size."""

    def get_deltas(self, src_boxes, target_boxes):
        src_ctr_x = 0.5 * (src_boxes[:, 0] + src_boxes[:, 2])
        src_ctr_y = 0.5 * (src_boxes[:, 1] + src_boxes[:, 3])
        deltas = torch.stack((src_ctr_x - target_boxes[:, 0], src_ctr_y - target_boxes[:, 1],
                              target_boxes[:, 2] - src_ctr_x, target_boxes[:, 3] - src_ctr_y), dim=1)
        stride_w = src_boxes[:, 2] - src_boxes[:, 0]
        stride_h = src_boxes[:, 3] - src_boxes[:, 1]
        return deltas / torch.stack([stride_w, stride_h, stride_w, stride_h], dim=1)

    def apply_deltas(self, deltas, boxes):
        deltas = F.relu(deltas)
        boxes = boxes.to(deltas.dtype)
        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        stride_w = boxes[:, 2] - boxes[:, 0]
        stride_h = boxes[:, 3] - boxes[:, 1]
        deltas = deltas * torch.stack([stride_w, stride_h, stride_w, stride_h], dim=1)
        return torch.stack((ctr_x - deltas[:, 0], ctr_y - deltas[:, 1], ctr_x + deltas[:, 2], ctr_y + deltas[:, 3]), 1)


def giou_loss(boxes1, boxes2, reduction="sum", eps=1e-7):
    """fvcore.nn.giou_loss"""
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"
    xkis1, ykis1 = torch.max(x1, x1g), torch.max(y1, y1g)
    xkis2, ykis2 = torch.min(x2, x2g), torch.min(y2, y2g)
    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)
    xc1, yc1 = torch.min(x1, x1g), torch.min(y1, y1g)
    xc2, yc2 = torch.max(x2, x2g), torch.max(y2, y2g)
    area_c = (xc2 - xc1) * (yc2 - yc1)
    loss = 1 - (iouk - ((area_c - unionk) / (area_c + eps)))
    return loss.sum() if reduction == "sum" else loss.mean()


class FCOS(nn.Module):
    def __init__(self, num_classes, depth=50, center_sampling_radius=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = build_resnet(depth)
        self.fpn = build_retinanet_fpn(self.backbone.out_channels, 256, top_block_in="p5")
        self.head = FCOSHead(256, num_classes)
        self.anchor_generator = AnchorGenerator([[s] for s in STRIDES], STRIDES, aspect_ratios=(1.0,))
        self.box2box_transform = Box2BoxTransformLinear()
        self.center_sampling_radius = center_sampling_radius
        self.focal_loss_alpha, self.focal_loss_gamma = 0.25, 2.0
        self.loss_normalizer = 300.0           # detectron2 FCOS: _ema_update(..., initial_value=300)
        self.precise_bn = True                 # [Unified] common/precise_bn.py (BN ResNet-50 + FPN, as RetinaNet)
        self.precise_bn_images = 1600
        self.precise_bn_batch_size = 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    @torch.no_grad()
    def _match_anchors(self, gt_boxes, anchors, num_anchors_per_level):
        anchor_centers = 0.5 * (anchors[:, :2] + anchors[:, 2:])
        anchor_sizes = anchors[:, 2] - anchors[:, 0]
        lower_bound = anchor_sizes * 4
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound[-num_anchors_per_level[-1]:] = float("inf")
        gt_centers = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
        center_dists = (anchor_centers[None, :, :] - gt_centers[:, None, :]).abs_()
        sampling_regions = self.center_sampling_radius * anchor_sizes[None, :]
        match_quality_matrix = center_dists.max(dim=2).values < sampling_regions
        x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)                        # pairwise_point_box_distance
        x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2).permute(1, 0, 2)   # (M, R, 4)
        match_quality_matrix &= pairwise_dist.min(dim=2).values > 0
        pairwise_dist = pairwise_dist.max(dim=2).values
        match_quality_matrix &= (pairwise_dist > lower_bound[None, :]) & (pairwise_dist < upper_bound[None, :])
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        match_quality_matrix = match_quality_matrix.to(torch.float32)
        match_quality_matrix *= 1e8 - gt_areas[:, None]
        return match_quality_matrix

    @torch.no_grad()
    def label_anchors(self, anchors, num_anchors_per_level, targets):
        gt_labels, matched_gt_boxes = [], []
        for t in targets:
            if len(t["labels"]) > 0:
                match_quality, matched_idxs = self._match_anchors(t["boxes"], anchors, num_anchors_per_level).max(dim=0)
                matched_idxs[match_quality < 1e-5] = -1
                matched_gt_boxes_i = t["boxes"][matched_idxs.clip(min=0)]
                gt_labels_i = t["labels"][matched_idxs.clip(min=0)].clone()
                gt_labels_i[matched_idxs < 0] = self.num_classes
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors)
                gt_labels_i = torch.full((len(anchors),), self.num_classes, dtype=torch.long, device=anchors.device)
            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def losses(self, anchors, pred_logits, gt_labels, pred_deltas, gt_boxes, pred_ctr):
        gt_labels = torch.stack(gt_labels)
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        self.loss_normalizer = 0.9 * self.loss_normalizer + 0.1 * max(num_pos_anchors, 1)
        normalizer = self.loss_normalizer
        gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[:, :, :-1]
        loss_cls = sigmoid_focal_loss(pred_logits, gt_labels_target.to(pred_logits.dtype),
                                      self.focal_loss_alpha, self.focal_loss_gamma, "sum")
        pred_boxes = torch.stack([self.box2box_transform.apply_deltas(k, anchors) for k in pred_deltas])
        loss_box_reg = giou_loss(pred_boxes[pos_mask], torch.stack(gt_boxes)[pos_mask], reduction="sum")
        reg_targets = torch.stack([self.box2box_transform.get_deltas(anchors, m) for m in gt_boxes])
        left_right, top_bottom = reg_targets[:, :, [0, 2]], reg_targets[:, :, [1, 3]]
        ctrness_targets = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                                     * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr[pos_mask], ctrness_targets[pos_mask], reduction="sum")
        losses = {"loss_fcos_cls": loss_cls / normalizer, "loss_fcos_loc": loss_box_reg / normalizer,
                  "loss_fcos_ctr": loss_ctr / normalizer}
        losses["losses"] = sum(losses.values())
        losses["num_pos"] = torch.tensor(num_pos_anchors / len(gt_boxes))
        return losses, pos_mask.view(-1), gt_labels[pos_mask]

    def forward(self, x, targets=None):
        feats = self.fpn(self.backbone(x))
        logits, bbox_reg, ctrness = self.head(feats)
        pred_logits = torch.cat([permute_to_N_HWA_K(t, self.num_classes) for t in logits], 1)
        pred_deltas = torch.cat([permute_to_N_HWA_K(t, 4) for t in bbox_reg], 1)
        pred_ctr = torch.cat([permute_to_N_HWA_K(t, 1) for t in ctrness], 1).squeeze(2)
        anchors_lv = self.anchor_generator([f.shape[-2:] for f in feats], x.device)
        anchors = torch.cat(anchors_lv, 0)
        B, R = pred_deltas.shape[:2]
        boxes = self.box2box_transform.apply_deltas(pred_deltas.reshape(-1, 4), anchors.repeat(B, 1)).view(B, R, 4)
        raw = to_raw(boxes, pred_logits.sigmoid().sqrt(), pred_ctr.sigmoid().sqrt().unsqueeze(-1))
        if targets is None:
            return raw
        gt_labels, gt_boxes = self.label_anchors(anchors, [len(a) for a in anchors_lv],
                                                 split_targets(targets, B, x.shape[2]))
        loss_dict, pos_mask, fg_labels = self.losses(anchors, pred_logits, gt_labels, pred_deltas, gt_boxes, pred_ctr)
        cls_flat = pred_logits.detach().reshape(-1, self.num_classes)
        self.stats.update(loss_dict, cls_flat.sigmoid().max(-1).values, cls_flat, pos_mask, fg_labels)
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        raise NotImplementedError("FCOS has no Darknet weights; train from scratch.")
