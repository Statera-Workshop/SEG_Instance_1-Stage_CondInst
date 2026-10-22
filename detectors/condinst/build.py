"""
CondInst (ResNet-50-FPN) builder - instance segmentation with dynamic (conditional) mask heads. Seg task.

Paper: Tian, Shen & Chen, "Conditional Convolutions for Instance Segmentation" (ECCV 2020, arXiv 2003.05664).
Code : AdelaiDet (aim-uofa, the authors' code; detectron2-based, approved source family) adet/modeling/condinst/
       condinst.py, dynamic_mask_head.py, mask_branch.py, configs/CondInst/Base-CondInst.yaml; reimplemented here
       on top of the detection stage's FCOS (detectors/fcos, unchanged box part).

    detector     FCOS R50-FPN P3-P7 (detectors/fcos.FCOS: GN head, centerness, center sampling, focal + GIoU)
    controller   conv 3x3 on the FCOS box tower -> 169 parameters per location (std 0.01 init, bias 0)
    mask branch  P3, P4, P5 -> conv 3x3 128 + BN + ReLU each, upsampled to P3 (aligned bilinear) and summed
                 -> 4 x (conv 3x3 128 + BN + ReLU) -> conv 1x1 8 channels (stride 8)
    mask head    per instance: [8 mask features, 2 relative coordinates / size of interest] -> 1x1 conv 8 + ReLU
                 -> 1x1 conv 8 + ReLU -> 1x1 conv 1 (weights 80 + 64 + 8, biases 8 + 8 + 1 = 169), sizes of
                 interest 64 / 128 / 256 / 512 / 1024 for P3-P7; logits upsampled 2x (aligned bilinear) to stride 4
    loss         FCOS losses + dice loss on sigmoid masks vs ground truth sampled at stride 4 (offset 2),
                 averaged over the positive locations (max 500 sampled per batch, MAX_PROPOSALS), weight 1
    inference    the controller parameters of each kept detection build its mask (dynamic head on the shared
                 mask features), sigmoid, cropped with the box, threshold 0.5 (tasks/seg_task.py)

Raw contract: FCOS rows [B, R, 5 + C] (one per location); aux = mask features + parameters + locations.

[Unified] as FCOS (no ImageNet weights, trainable BN + PreciseBN, common init, no mean / std, 608 input, Adam 1e-4,
common post-processing); semantic auxiliary loss off (Base-CondInst default); instance locations are the FCOS
anchor centres of the framework (detectron2 anchor offset 0), mask-feature locations x * 8 + 4 (AdelaiDet
compute_locations); the predicted mask is also cropped with its box at inference (common seg post-processing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors.fcos.build import FCOS
from detectors.retinanet.retinanet_head import permute_to_N_HWA_K
from detectors.anchor_free.common import to_raw
from detectors.yolo11_seg.build import split_seg_targets

CH, LAYERS, IN_CH = 8, 3, 10
WEIGHT_NUMS, BIAS_NUMS = [IN_CH * CH, CH * CH, CH], [CH, CH, 1]
NUM_PARAMS = sum(WEIGHT_NUMS) + sum(BIAS_NUMS)                       # 169
SOI = (64, 128, 256, 512, 1024)
MASK_STRIDE, MASK_OUT_STRIDE, MAX_PROPOSALS = 8, 4, 500


def aligned_bilinear(t, factor):
    """AdelaiDet adet/utils/comm.aligned_bilinear."""
    if factor == 1:
        return t
    h, w = t.shape[2:]
    t = F.pad(t, (0, 1, 0, 1), mode="replicate")
    oh, ow = factor * h + 1, factor * w + 1
    t = F.interpolate(t, size=(oh, ow), mode="bilinear", align_corners=True)
    t = F.pad(t, (factor // 2, 0, factor // 2, 0), mode="replicate")
    return t[:, :, :oh - 1, :ow - 1]


def dice_loss(x, target, eps=1e-5):
    """AdelaiDet dice_coefficient: 1 - 2 |x t| / (|x|^2 + |t|^2), per instance."""
    x, target = x.flatten(1), target.flatten(1)
    return 1.0 - 2 * (x * target).sum(1) / ((x * x).sum(1) + (target * target).sum(1) + eps)


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MaskBranch(nn.Module):
    def __init__(self, cin=256, c=128, out=CH):
        super().__init__()
        self.refine = nn.ModuleList(conv_bn_relu(cin, c) for _ in range(3))
        self.tower = nn.Sequential(*[conv_bn_relu(c, c) for _ in range(4)], nn.Conv2d(c, out, 1))

    def forward(self, p3, p4, p5):
        x = self.refine[0](p3)
        for k, f in ((1, p4), (2, p5)):
            y = self.refine[k](f)
            fac = x.shape[-1] // y.shape[-1]
            y = aligned_bilinear(y, fac)[:, :, :x.shape[2], :x.shape[3]]
            x = x + y
        return self.tower(x)


class CondInst(FCOS):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.controller = nn.Conv2d(256, NUM_PARAMS, 3, padding=1)
        nn.init.normal_(self.controller.weight, std=0.01)
        nn.init.constant_(self.controller.bias, 0)
        self.mask_branch = MaskBranch()

    # ------------------------------------------------------------ dynamic head
    def mask_logits(self, mask_feats, params, inst_locs, inst_soi):
        """mask_feats [8, h, w] (one image), params [N, 169], inst_locs [N, 2] px, inst_soi [N] -> [N, 2h, 2w]."""
        _, h, w = mask_feats.shape
        dev = mask_feats.device
        ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
        locs = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).float() * MASK_STRIDE + MASK_STRIDE // 2
        rel = (inst_locs[:, None, :] - locs[None]) / inst_soi[:, None, None]          # AdelaiDet: instance - grid
        n = len(params)
        x = torch.cat([rel.permute(0, 2, 1), mask_feats.reshape(1, CH, -1).expand(n, -1, -1)], 1)   # [N, 10, hw]
        parts = torch.split(params, WEIGHT_NUMS + BIAS_NUMS, 1)
        ws, bs = parts[:LAYERS], parts[LAYERS:]
        cin = IN_CH
        for k in range(LAYERS):
            cout = CH if k < LAYERS - 1 else 1
            x = torch.bmm(ws[k].reshape(n, cout, cin), x) + bs[k].reshape(n, cout, 1)
            if k < LAYERS - 1:
                x = F.relu(x)
            cin = cout
        return aligned_bilinear(x.reshape(n, 1, h, w), MASK_STRIDE // MASK_OUT_STRIDE)[:, 0]

    # ------------------------------------------------------------ forward
    def forward(self, x, targets=None):
        feats = self.fpn(self.backbone(x))
        h = self.head                                          # FCOSHead.forward, keeping the box tower
        tower = [h.bbox_subnet(f) for f in feats]
        logits = [h.cls_score(h.cls_subnet(f)) for f in feats]
        bbox_reg = [h.bbox_pred(t) for t in tower]
        ctrness = [h.ctrness(t) for t in tower]
        params = torch.cat([permute_to_N_HWA_K(self.controller(t), NUM_PARAMS) for t in tower], 1)  # [B, R, 169]
        mask_feats = self.mask_branch(*feats[:3])
        pred_logits = torch.cat([permute_to_N_HWA_K(t, self.num_classes) for t in logits], 1)
        pred_deltas = torch.cat([permute_to_N_HWA_K(t, 4) for t in bbox_reg], 1)
        pred_ctr = torch.cat([permute_to_N_HWA_K(t, 1) for t in ctrness], 1).squeeze(2)
        anchors_lv = self.anchor_generator([f.shape[-2:] for f in feats], x.device)
        anchors = torch.cat(anchors_lv, 0)
        locs = 0.5 * (anchors[:, :2] + anchors[:, 2:])
        soi = torch.cat([torch.full((len(a),), float(s), device=x.device) for a, s in zip(anchors_lv, SOI)])
        B, R = pred_deltas.shape[:2]
        boxes = self.box2box_transform.apply_deltas(pred_deltas.reshape(-1, 4), anchors.repeat(B, 1)).view(B, R, 4)
        raw = to_raw(boxes, pred_logits.sigmoid().sqrt(), pred_ctr.sigmoid().sqrt().unsqueeze(-1))
        if targets is None:
            if self.training:                                   # PreciseBN forward
                return raw
            return raw, {"mask_feats": mask_feats, "params": params, "locs": locs, "soi": soi}

        tgt = split_seg_targets(targets, B, x.shape[2])
        gt_labels, gt_boxes = self.label_anchors(anchors, [len(a) for a in anchors_lv], tgt)
        loss_dict, pos_mask, fg_labels = self.losses(anchors, pred_logits, gt_labels, pred_deltas, gt_boxes, pred_ctr)
        loss_mask = self.mask_loss(anchors, [len(a) for a in anchors_lv], tgt, params, mask_feats, locs, soi)
        loss_dict["loss_mask"] = loss_mask
        loss_dict["losses"] = loss_dict["losses"] + loss_mask
        cls_flat = pred_logits.detach().reshape(-1, self.num_classes)
        self.stats.update(loss_dict, cls_flat.sigmoid().max(-1).values, cls_flat, pos_mask, fg_labels)
        return loss_dict["losses"], raw

    def mask_loss(self, anchors, n_per_level, tgt, params, mask_feats, locs, soi):
        """Positive locations (same FCOS assignment) -> dynamic masks -> dice vs gt at stride 4."""
        items = []                                             # (image, location rows, gt index)
        with torch.no_grad():
            for i, t in enumerate(tgt):
                if len(t["labels"]) == 0:
                    continue
                q, idx = self._match_anchors(t["boxes"], anchors, n_per_level).max(0)
                rows = torch.nonzero(q >= 1e-5).squeeze(1)
                if rows.numel():
                    items.append((i, rows, idx[rows]))
        total = sum(len(r) for _, r, _ in items)
        if total == 0:
            return params.sum() * 0.0 + mask_feats.sum() * 0.0
        keep = torch.randperm(total, device=params.device)[:MAX_PROPOSALS] if total > MAX_PROPOSALS else None
        losses, offset = [], 0
        start = MASK_OUT_STRIDE // 2
        for i, rows, gi in items:
            if keep is not None:
                sel = keep[(keep >= offset) & (keep < offset + len(rows))] - offset
                offset += len(rows)
                if sel.numel() == 0:
                    continue
                rows, gi = rows[sel], gi[sel]
            pred = self.mask_logits(mask_feats[i], params[i, rows], locs[rows], soi[rows]).sigmoid()
            gt = tgt[i]["masks"][:, start::MASK_OUT_STRIDE, start::MASK_OUT_STRIDE].float()[gi]
            losses.append(dice_loss(pred, gt[:, :pred.shape[1], :pred.shape[2]]))
        return torch.cat(losses).mean()

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        logit = self.mask_logits(aux["mask_feats"][b], aux["params"][b, rows], aux["locs"][rows], aux["soi"][rows])
        return {"kind": "full", "prob": logit.sigmoid()}


def build_model(task, num_classes):
    return CondInst(num_classes)
