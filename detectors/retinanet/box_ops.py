"""
Anchors, anchor matching and box coding for RetinaNet.

Code : detectron2
    modeling/anchor_generator.py  DefaultAnchorGenerator (generate_cell_anchors, _create_grid_offsets,
                                  _grid_anchors); OFFSET 0.0; ASPECT_RATIOS [0.5, 1.0, 2.0] (H/W)
    configs/Base-RetinaNet.yaml   SIZES [[x, x*2^(1/3), x*2^(2/3)] for x in 32, 64, 128, 256, 512]
    modeling/matcher.py           Matcher (IOU_THRESHOLDS [0.4, 0.5], IOU_LABELS [0, -1, 1],
                                  allow_low_quality_matches=True)
    modeling/box_regression.py    Box2BoxTransform (weights 1, 1, 1, 1; scale_clamp log(1000 / 16))
    structures/boxes.py           pairwise_iou
Paper: Lin et al. 2017, Sec. 4 "Anchors": areas 32^2..512^2 on P3..P7, 3 ratios x 3 scales = A = 9;
       IoU >= 0.5 foreground, [0, 0.4) background, [0.4, 0.5) ignored.
"""

import math

import torch

STRIDES = (8, 16, 32, 64, 128)
ANCHOR_SIZES = [[x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [32, 64, 128, 256, 512]]
ASPECT_RATIOS = (0.5, 1.0, 2.0)


def generate_cell_anchors(sizes, aspect_ratios=ASPECT_RATIOS):
    """detectron2 DefaultAnchorGenerator.generate_cell_anchors -> [A, 4] XYXY centred at 0."""
    anchors = []
    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
    return torch.tensor(anchors)


class AnchorGenerator:
    """detectron2 DefaultAnchorGenerator, offset 0.0: list of [Hi*Wi*A, 4] XYXY anchors."""

    def __init__(self, sizes=ANCHOR_SIZES, strides=STRIDES, offset=0.0, aspect_ratios=ASPECT_RATIOS):
        self.strides = strides
        self.offset = offset
        self.cell_anchors = [generate_cell_anchors(s, aspect_ratios) for s in sizes]
        self.num_cell_anchors = len(self.cell_anchors[0])

    def __call__(self, feat_sizes, device):
        anchors = []
        for (h, w), stride, base in zip(feat_sizes, self.strides, self.cell_anchors):
            shifts_x = torch.arange(self.offset * stride, w * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(self.offset * stride, h * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base.to(device).view(1, -1, 4)).reshape(-1, 4))
        return anchors
