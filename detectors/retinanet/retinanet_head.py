"""
RetinaNet classification / box subnets.

Code : detectron2 modeling/meta_arch/retinanet.py RetinaNetHead (NUM_CONVS 4, conv_dims 256, NORM "",
       PRIOR_PROB 0.01); detectron2 _transpose_dense_predictions (dense_detector.py) for the output layout.
Paper: Lin et al. 2017, Sec. 4 and Fig. 3: two FCN subnets of 4 x (3x3 conv 256 + ReLU) shared across
       pyramid levels, last conv K*A (sigmoid) / 4*A; Sec. 4.1 "Initialization": bias of the last
       classification conv = -log((1 - pi) / pi), pi = 0.01.

Init: detectron2 sets every head conv to N(0, 0.01), bias 0, then the prior bias. The framework's
common init (weights_init_normal) later re-draws the conv WEIGHTS from N(0, 0.02) ([Unified]); it does
not touch biases, so the prior bias of cls_score survives (checked in the unit test).
"""


def permute_to_N_HWA_K(tensor, K):
    """detectron2: (N, A*K, Hi, Wi) -> (N, Hi*Wi*A, K)"""
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    return tensor.reshape(N, -1, K)
