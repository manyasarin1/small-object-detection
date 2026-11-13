"""
M4 -- M3.1 + CPNet (Cascade Aggregation Pyramid + Ternary Cascade Perception Module)
replacing the plain FPN.
========================================================================
Everything is identical to M3.1 (ResNet50 backbone, 512x512 RGB, AdamW, LR 1e-4, 25 epochs,
positive_chip_prob 0.5, same 80/20 split, same 100 fixed test chips, same NMS, center-sampling
assignment with radius 1.5 x stride, FIXED decoding (best of the 60 real classes, background
excluded), classification loss normalized by the number of positive points, objectness +
centerness heads, background-prior bias init).

ONLY CHANGE: the plain FPN (1x1 lateral projection + 3x3 smoothing) is replaced by CPNet's
Cascade Aggregation Pyramid (ASPP + AN blocks) feeding a Ternary Cascade Perception Module
(FPM graph-conv channel attention -> SPM spatial normalization -> KPM dilated-conv keypoint
perception) at every pyramid level, before the same cls / bbox / objectness / centerness heads.

WHY: a plain FPN pools small, low-contrast, texture-camouflaged objects through the same
1x1-lateral + 3x3-smooth path as everything else. CPNet's cascaded multi-scale context (CAP)
and graph-based channel attention + keypoint-focused spatial refinement (TCPM) exist specifically
to recover that kind of weak signal. This is the project's actual novelty claim, and this is the
FIRST time it is trained for the full 25 epochs under the corrected (M3.1) loss/decoding pipeline
-- every earlier CPNet/CPNet+Gabor run in this project was a 5-epoch smoke test and is not a fair
comparison point. Gabor features are deliberately NOT added yet (that is M5); this isolates the
CPNet contribution on its own before stacking the texture-feature change on top of it.

SELF-CONTAINED: the CPNet module library (GraphConv, CBR, FeaturePerceptionModule,
SpatialPerceptionModule, KeyPointPerceptionModule, TernaryCascadePerceptionModule, ASPP, ANBlock,
CascadeAggregationPyramid) is copied in below verbatim from manyasarin/camo-detection-current, so
nothing needs to be pasted in. CPNetPyramid wraps CascadeAggregationPyramid (builds the 4-level
pyramid via ASPP + AN-block cascade fusion) + one TernaryCascadePerceptionModule per level
(FPM graph-conv channel attention -> SPM spatial norm -> KPM dilated-conv keypoint perception),
used as a drop-in replacement for the plain FPN -- it keeps M3.1's separate cls/bbox/obj/ctr heads
rather than the old CPNet class's baked-in plain heads, so centerness/objectness still work.

EVALUATION: re-evaluates the M3.1 checkpoint (and M0/M1/M2 if present) with the identical
evaluator, exactly as the M3.1 run did for M3 -- so the M4 number you get is directly comparable,
not just another standalone run.

Run with: python xview_m4_cpnet.py
"""

import os
import glob
import json
import math
import time
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.flop_counter import FlopCounterMode
import torchvision.models as models
import torchvision.ops as ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

# ---------------- M0-M3.1 settings (unchanged) ----------------
IMG_SIZE = 512
CHIP_SIZE = 512
BATCH_SIZE = 2
NUM_EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEST_FRACTION = 0.20
RANDOM_SEED = 42
TEST_SAMPLE_SEED = 12345
NUM_TEST_CHIPS = 100
IOU_MATCH_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.40
CENTER_RADIUS = 1.5
NUM_FPS_CHIPS = 50
LAMBDA_OBJ = 1.0
OBJ_FOCAL_ALPHA = 0.25
OBJ_PRIOR_PROB = 0.01
LAMBDA_CTR = 0.5
MAP_CONF_THRESHOLD = 0.01       # scores are a product of 3 probabilities -> AP floor is low
CONF_THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.50, 0.60]
BACKGROUND_PRIOR_PROB = 0.99     # companion of the per-positive-normalized classification loss

# Only needed if Kaggle Internet is OFF. Path to resnet50-11ad3fa6.pth.
IMAGENET_WEIGHTS_PATH = None

CHECKPOINT_PATH = "/kaggle/working/cpnet_xview_m4_checkpoint.pt"
VISUALIZATION_PATH = "/kaggle/working/xview_m4_prediction_result.png"
TUNED_VISUALIZATION_PATH = "/kaggle/working/xview_m4_prediction_tuned.png"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ============================================================
# ARCHITECTURE: ResNet50 + (plain FPN for M0-M3.1 | CPNet pyramid for M4) + bare heads
# M0Detector (cls, bbox) -> M2Detector (+objectness) -> M3Detector (+centerness) -> M4Detector (+CPNet pyramid)
# ============================================================

class ConvBNReLU(nn.Module):
    """Plain conv -> batchnorm -> relu. Used only by the plain FPN's smoothing step."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    """Standard Feature Pyramid Network (Lin et al., 2017). Kept so M0/M1/M2/M3/M3.1
    checkpoints still load correctly for the comparison table at the end."""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.smooth_convs = nn.ModuleList([ConvBNReLU(out_channels, out_channels) for _ in in_channels_list])

    def forward(self, feature_list):
        laterals = [l(x) for l, x in zip(self.lateral_convs, feature_list)]
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="nearest")
            laterals[i] = laterals[i] + upsampled
        return [s(x) for s, x in zip(self.smooth_convs, laterals)]


# ============================================================
# CPNet module library (verbatim from manyasarin/camo-detection-current)
# ============================================================

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adj = nn.Parameter(torch.FloatTensor(in_channels, in_channels))
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.adj)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.mean(dim=(2, 3))
        adj = torch.sigmoid(self.adj)
        deg = torch.sum(adj, dim=1, keepdim=True) + 1e-6
        deg_inv_sqrt = torch.pow(deg, -0.5)
        support = deg_inv_sqrt * adj * deg_inv_sqrt.t()
        out = torch.matmul(x_flat, self.weight)
        out = torch.matmul(support, out.t()).t()
        return out.view(B, self.out_channels, 1, 1)


class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FeaturePerceptionModule(nn.Module):
    """FPM: graph-conv channel attention."""
    def __init__(self, channels):
        super(FeaturePerceptionModule, self).__init__()
        self.cbr = CBR(channels, channels, kernel_size=3, padding=1)
        self.gcn = GraphConv(channels, channels)

    def forward(self, x):
        f_cbr = self.cbr(x)
        f_gap = torch.mean(f_cbr, dim=(2, 3), keepdim=True)
        f_gcn = self.gcn(f_gap)
        f_weight = torch.softmax(f_gcn, dim=1)
        f_weighted = f_cbr * f_weight
        return f_weighted + x


class SpatialPerceptionModule(nn.Module):
    """SPM: spatial normalization."""
    def __init__(self, channels):
        super(SpatialPerceptionModule, self).__init__()
        self.cbr = CBR(channels, channels, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        f_cbr = self.cbr(x)
        mean = f_cbr.mean(dim=1, keepdim=True)
        std = f_cbr.std(dim=1, keepdim=True) + 1e-6
        f_bn = (f_cbr - mean) / std

        w_gamma = self.gamma / (torch.sum(self.gamma) + 1e-6)
        w_gamma = w_gamma.view(1, -1, 1, 1)

        f_w_gamma = f_bn * w_gamma
        f_weight = torch.softmax(f_w_gamma, dim=1)
        f_weighted = f_cbr * f_weight
        return f_weighted + x


class KeyPointPerceptionModule(nn.Module):
    """KPM: dilated-conv keypoint perception."""
    def __init__(self, channels):
        super(KeyPointPerceptionModule, self).__init__()
        self.cbr_location = CBR(channels, channels, kernel_size=3, padding=1)
        self.dconv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.dconv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=4, dilation=2, bias=False)
        self.cbr_rf = CBR(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, x, f_kpm_cbr_in):
        f_pool_w = torch.mean(f_kpm_cbr_in, dim=3, keepdim=True)
        f_pool_h = torch.mean(f_kpm_cbr_in, dim=2, keepdim=True)
        f_w_x_h = f_pool_w * f_pool_h
        f_location = self.cbr_location(f_w_x_h)

        dconv3_out = self.dconv3(f_location)
        dconv5_out = self.dconv5(f_location)
        f_rf = torch.cat([dconv3_out, dconv5_out], dim=1)

        f_weight = torch.softmax(self.cbr_rf(f_rf), dim=1)
        f_weighted = f_kpm_cbr_in * f_weight
        return f_weighted + x


class TernaryCascadePerceptionModule(nn.Module):
    """TCPM: FPM -> SPM -> KPM, cascaded."""
    def __init__(self, channels):
        super(TernaryCascadePerceptionModule, self).__init__()
        self.fpm = FeaturePerceptionModule(channels)
        self.spm = SpatialPerceptionModule(channels)
        self.kpm = KeyPointPerceptionModule(channels)
        self.kpm_cbr = CBR(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        f_fpm = self.fpm(x)
        f_spm = self.spm(f_fpm)
        f_kpm_cbr_in = self.kpm_cbr(f_spm)
        f_kpm = self.kpm(f_spm, f_kpm_cbr_in)
        return f_kpm


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp_layers = nn.ModuleList()
        self.aspp_layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in rates[1:]:
            self.aspp_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = []
        for layer in self.aspp_layers:
            res.append(layer(x))
        gp = self.global_pooling(x)
        gp_up = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(gp_up)
        return self.project(torch.cat(res, dim=1))


class ANBlock(nn.Module):
    def __init__(self, channels):
        super(ANBlock, self).__init__()
        self.cbr_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.cbr_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_lower, f_aspp):
        f_cbr = self.cbr_down(f_lower)
        if f_cbr.shape[2:] != f_aspp.shape[2:]:
            f_cbr = F.interpolate(f_cbr, size=f_aspp.shape[2:], mode='bilinear', align_corners=True)
        return self.cbr_fuse(f_cbr + f_aspp)


class CascadeAggregationPyramid(nn.Module):
    """CAP: 1x1 projection -> top-down fusion -> per-level ASPP -> AN-block cascade fusion."""
    def __init__(self, in_channels_list, out_channels=256):
        super(CascadeAggregationPyramid, self).__init__()
        self.out_channels = out_channels
        self.proj_blocks = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.aspp_blocks = nn.ModuleList([
            ASPP(out_channels, out_channels) for _ in range(4)
        ])
        self.an_blocks = nn.ModuleList([
            ANBlock(out_channels) for _ in range(3)
        ])
        self.fusion_blocks = nn.ModuleList([
            CBR(out_channels, out_channels, kernel_size=3, padding=1) for _ in range(4)
        ])

    def forward(self, feature_list):
        proj_features = [proj(x) for proj, x in zip(self.proj_blocks, feature_list)]
        td_features = list(proj_features)
        for i in range(2, -1, -1):
            up = F.interpolate(td_features[i + 1], size=proj_features[i].shape[2:], mode='nearest')
            td_features[i] = proj_features[i] + up

        aspp_features = [aspp(x) for aspp, x in zip(self.aspp_blocks, td_features)]

        an_features = list(aspp_features)
        for i in range(1, 4):
            an_features[i] = self.an_blocks[i - 1](an_features[i - 1], aspp_features[i])

        out_features = [fuse(an) for fuse, an in zip(self.fusion_blocks, an_features)]
        return out_features


class CPNetPyramid(nn.Module):
    """
    M4's replacement for the plain FPN.

    Cascade Aggregation Pyramid (CAP) first builds the same 4-level pyramid as a normal FPN,
    but each level is produced through ASPP + AN blocks (multi-scale dilated context) instead of
    a single 3x3 conv. Each resulting level is then refined by one Ternary Cascade Perception
    Module (TCPM): FPM (graph-conv channel attention) -> SPM (spatial normalization) ->
    KPM (dilated-conv keypoint perception).
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.cap = CascadeAggregationPyramid(in_channels_list, out_channels)
        self.tcpm = nn.ModuleList([TernaryCascadePerceptionModule(out_channels) for _ in in_channels_list])

    def forward(self, feature_list):
        pyramid = self.cap(feature_list)          # list of out_channels-channel maps, one per level
        return [tcpm(level) for tcpm, level in zip(self.tcpm, pyramid)]


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, weights_path=None):
        super().__init__()
        if pretrained and weights_path:
            resnet = models.resnet50(weights=None)
            resnet.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class M0Detector(nn.Module):
    """M0/M1 head: plain FPN + one classification conv and one bbox conv per pyramid level."""
    def __init__(self, in_channels_list, out_channels=256, num_classes_total=61):
        super().__init__()
        self.fpn = FPN(in_channels_list, out_channels)
        self.cls_heads = nn.ModuleList([nn.Conv2d(out_channels, num_classes_total, 3, padding=1) for _ in range(4)])
        self.bbox_heads = nn.ModuleList([nn.Conv2d(out_channels, 4, 3, padding=1) for _ in range(4)])

    def forward(self, backbone_features):
        pyramid = self.fpn(backbone_features)
        cls_preds = [head(f) for head, f in zip(self.cls_heads, pyramid)]
        bbox_preds = [head(f) for head, f in zip(self.bbox_heads, pyramid)]
        return pyramid, cls_preds, bbox_preds


class M2Detector(M0Detector):
    """M2 head: M1 head + one objectness conv (256 -> 1) per pyramid level."""
    def __init__(self, in_channels_list, out_channels=256, num_classes_total=61):
        super().__init__(in_channels_list, out_channels, num_classes_total)
        self.obj_heads = nn.ModuleList([nn.Conv2d(out_channels, 1, 3, padding=1) for _ in range(4)])
        for head in self.obj_heads:
            nn.init.constant_(head.bias, -math.log((1 - OBJ_PRIOR_PROB) / OBJ_PRIOR_PROB))

    def forward(self, backbone_features):
        pyramid, cls_preds, bbox_preds = super().forward(backbone_features)
        obj_preds = [head(f) for head, f in zip(self.obj_heads, pyramid)]
        return pyramid, cls_preds, bbox_preds, obj_preds


class M3Detector(M2Detector):
    """M3 / M3.1 head: M2 head + one centerness conv (256 -> 1) per pyramid level."""
    def __init__(self, in_channels_list, out_channels=256, num_classes_total=61):
        super().__init__(in_channels_list, out_channels, num_classes_total)
        self.ctr_heads = nn.ModuleList([nn.Conv2d(out_channels, 1, 3, padding=1) for _ in range(4)])

    def forward(self, backbone_features):
        pyramid, cls_preds, bbox_preds, obj_preds = super().forward(backbone_features)
        ctr_preds = [head(f) for head, f in zip(self.ctr_heads, pyramid)]
        return pyramid, cls_preds, bbox_preds, obj_preds, ctr_preds


class M4Detector(M3Detector):
    """M4 head: identical cls/bbox/obj/ctr heads as M3.1, but the plain FPN inherited from
    M0Detector is replaced with CPNet's Cascade Aggregation Pyramid + Ternary Cascade
    Perception Module stack. ONLY CHANGE vs M3.1: self.fpn."""
    def __init__(self, in_channels_list, out_channels=256, num_classes_total=61):
        super().__init__(in_channels_list, out_channels, num_classes_total)
        self.fpn = CPNetPyramid(in_channels_list, out_channels)


class ResNetDetector(nn.Module):
    """ResNet50 backbone + the chosen head class (M0Detector / M2Detector / M3Detector / M4Detector)."""
    def __init__(self, head_cls, pretrained=True, out_channels=256, num_classes_total=61, weights_path=None):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained, weights_path=weights_path)
        self.model = head_cls([256, 512, 1024, 2048], out_channels, num_classes_total)

    def forward(self, x):
        return self.model(self.backbone(x))


def init_background_prior(model, p_background=BACKGROUND_PRIOR_PROB):
    """Standard companion of a positive-normalized focal loss: start with p(background) ~ 0.99
    at every location so the classifier isn't fighting its own init on top of the loss change."""
    num_real = NUM_REAL_CLASSES
    bias_bg = math.log(p_background * num_real / (1 - p_background))
    for head in model.model.cls_heads:
        nn.init.constant_(head.bias, 0.0)
        head.bias.data[0] = bias_bg


XVIEW_CLASS_REMAP = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10,
    11, 12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1, 23, 24, 25, -1, 26, 27, -1, 28, -1,
    29, 30, 31, 32, 33, 34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1, 46, 47,
    48, 49, -1, 50, 51, -1, 52, -1, -1, -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59,
]

XVIEW_CLASS_NAMES = [
    "Fixed-wing Aircraft", "Small Aircraft", "Cargo Plane", "Helicopter", "Passenger Vehicle",
    "Small Car", "Bus", "Pickup Truck", "Utility Truck", "Truck", "Cargo Truck", "Truck w/Box",
    "Truck Tractor", "Trailer", "Truck w/Flatbed", "Truck w/Liquid", "Crane Truck",
    "Railway Vehicle", "Passenger Car", "Cargo Car", "Flat Car", "Tank car", "Locomotive",
    "Maritime Vessel", "Motorboat", "Sailboat", "Tugboat", "Barge", "Fishing Vessel", "Ferry",
    "Yacht", "Container Ship", "Oil Tanker", "Engineering Vehicle", "Tower crane",
    "Container Crane", "Reach Stacker", "Straddle Carrier", "Mobile Crane", "Dump Truck",
    "Haul Truck", "Scraper/Tractor", "Front loader/Bulldozer", "Excavator", "Cement Mixer",
    "Ground Grader", "Hut/Tent", "Shed", "Building", "Aircraft Hangar", "Damaged Building",
    "Facility", "Construction Site", "Vehicle Lot", "Helipad", "Storage Tank",
    "Shipping container lot", "Shipping Container", "Pylon", "Tower",
]
NUM_REAL_CLASSES = len(XVIEW_CLASS_NAMES)
NUM_CLASSES_TOTAL = NUM_REAL_CLASSES + 1


def remap_type_id(raw_type_id):
    raw_type_id = int(raw_type_id)
    if raw_type_id < 0 or raw_type_id >= len(XVIEW_CLASS_REMAP):
        return None
    mapped = XVIEW_CLASS_REMAP[raw_type_id]
    return mapped if mapped != -1 else None


class XViewDataset(Dataset):
    def __init__(self, root_dir, chip_size=512, img_size=512, chips_per_image=4,
                 positive_chip_prob=0.8, seed=0, allowed_image_ids=None, deterministic=False):
        self.chip_size = chip_size
        self.img_size = img_size
        self.chips_per_image = chips_per_image
        self.positive_chip_prob = positive_chip_prob
        self.seed = seed
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self.allowed_image_ids = set(allowed_image_ids) if allowed_image_ids is not None else None

        geojson_path = self._find_geojson(root_dir)
        images_dir = self._find_images_dir(root_dir)
        print(f"[XViewDataset] Using geojson: {geojson_path}")
        print(f"[XViewDataset] Using image dir: {images_dir}")

        with open(geojson_path, "r") as f:
            data = json.load(f)

        self.annotations_by_image = {}
        skipped_invalid_class = 0
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            image_id = props.get("image_id")
            bounds = props.get("bounds_imcoords")
            raw_type_id = props.get("type_id")
            if not image_id or not bounds or raw_type_id is None:
                continue
            cls_idx = remap_type_id(raw_type_id)
            if cls_idx is None:
                skipped_invalid_class += 1
                continue
            try:
                xmin, ymin, xmax, ymax = [float(v) for v in bounds.split(",")]
            except ValueError:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            self.annotations_by_image.setdefault(image_id, []).append(
                {"box": (xmin, ymin, xmax, ymax), "label": cls_idx})

        self.image_paths = {}
        for image_id in list(self.annotations_by_image.keys()):
            candidate = os.path.join(images_dir, image_id)
            if os.path.exists(candidate):
                self.image_paths[image_id] = candidate
            else:
                del self.annotations_by_image[image_id]

        self.image_ids = sorted(self.image_paths.keys())

        if self.allowed_image_ids is not None:
            before = len(self.image_ids)
            self.image_ids = [i for i in self.image_ids if i in self.allowed_image_ids]
            print(f"[XViewDataset] Restricted to allowed_image_ids: {before} -> {len(self.image_ids)} images")

        print(f"[XViewDataset] {len(self.image_ids)} images with valid annotations found "
              f"({skipped_invalid_class} annotations skipped)")
        if len(self.image_ids) == 0:
            raise RuntimeError("No usable xView images found.")

    @staticmethod
    def _find_geojson(root_dir):
        matches = glob.glob(os.path.join(root_dir, "**", "*.geojson"), recursive=True)
        if not matches:
            raise RuntimeError("No .geojson file found.")
        train_matches = [m for m in matches if "train" in os.path.basename(m).lower()]
        return train_matches[0] if train_matches else matches[0]

    @staticmethod
    def _find_images_dir(root_dir):
        tif_matches = glob.glob(os.path.join(root_dir, "**", "*.tif"), recursive=True)
        if not tif_matches:
            raise RuntimeError("No .tif images found.")
        train_matches = [m for m in tif_matches if "train" in m.lower()]
        chosen = train_matches[0] if train_matches else tif_matches[0]
        return os.path.dirname(chosen)

    def __len__(self):
        return len(self.image_ids) * self.chips_per_image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx // self.chips_per_image]
        img_path = self.image_paths[image_id]
        anns = self.annotations_by_image[image_id]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        chip = self.chip_size

        if self.deterministic:
            rng = random.Random(self.seed + idx)
        else:
            rng = self.rng

        use_positive_chip = rng.random() < self.positive_chip_prob
        if use_positive_chip and anns:
            anchor = rng.choice(anns)
            axmin, aymin, axmax, aymax = anchor["box"]
            acx = (axmin + axmax) / 2
            acy = (aymin + aymax) / 2
            jitter_x = rng.uniform(-chip * 0.3, chip * 0.3)
            jitter_y = rng.uniform(-chip * 0.3, chip * 0.3)
            left = int(acx - chip / 2 + jitter_x)
            top = int(acy - chip / 2 + jitter_y)
        else:
            left = rng.randint(0, max(1, w - chip))
            top = rng.randint(0, max(1, h - chip))

        left = max(0, min(left, max(0, w - chip)))
        top = max(0, min(top, max(0, h - chip)))
        right = min(w, left + chip)
        bottom = min(h, top + chip)

        chip_img = img.crop((left, top, right, bottom))
        if chip_img.size != (chip, chip):
            padded = Image.new("RGB", (chip, chip))
            padded.paste(chip_img, (0, 0))
            chip_img = padded

        boxes = []
        labels = []
        for ann in anns:
            bxmin, bymin, bxmax, bymax = ann["box"]
            ixmin = max(bxmin, left)
            iymin = max(bymin, top)
            ixmax = min(bxmax, right)
            iymax = min(bymax, bottom)
            if ixmax > ixmin and iymax > iymin:
                boxes.append([ixmin - left, iymin - top, ixmax - left, iymax - top])
                labels.append(ann["label"])

        scale = self.img_size / chip
        chip_resized = chip_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        if boxes:
            boxes = [[c * scale for c in box] for box in boxes]

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        rgb_tensor = torch.from_numpy(np.array(chip_resized)).permute(2, 0, 1).float() / 255.0
        rgb_tensor = (rgb_tensor - IMAGENET_MEAN) / IMAGENET_STD

        return rgb_tensor, {"boxes": boxes_tensor, "labels": labels_tensor}


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# ============================================================
# M1 target assignment (unchanged): center sampling, level by max(l, r, t, b)
# ============================================================
SIZE_RANGES = {4: (0, 32), 8: (32, 64), 16: (64, 128), 32: (128, float("inf"))}


def get_targets(images, targets_list, strides):
    B = len(targets_list)
    device = images.device
    cls_targets_all = []
    reg_targets_all = []
    mask_targets_all = []

    for stride in strides:
        size_lo, size_hi = SIZE_RANGES[stride]
        H_l = images.shape[2] // stride
        W_l = images.shape[3] // stride
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H_l, device=device), torch.arange(W_l, device=device), indexing="ij")
        x_centers = (x_grid + 0.5) * stride
        y_centers = (y_grid + 0.5) * stride

        batch_cls = torch.zeros(B, H_l, W_l, dtype=torch.long, device=device)
        batch_reg = torch.zeros(B, H_l, W_l, 4, dtype=torch.float32, device=device)
        batch_mask = torch.zeros(B, H_l, W_l, dtype=torch.float32, device=device)
        assigned_area = torch.full((B, H_l, W_l), float("inf"), dtype=torch.float32, device=device)

        for b in range(B):
            gt_boxes = targets_list[b]["boxes"].to(device)
            gt_labels = targets_list[b]["labels"].to(device)
            for box, label in zip(gt_boxes, gt_labels):
                xmin, ymin, xmax, ymax = [float(v) for v in box]
                box_area = (xmax - xmin) * (ymax - ymin)

                l = x_centers - xmin
                t = y_centers - ymin
                r = xmax - x_centers
                bt = ymax - y_centers
                dists = torch.stack([l, t, r, bt], dim=-1)

                inside_box = dists.min(dim=-1).values > 0

                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                radius = CENTER_RADIUS * stride
                in_center = ((x_centers > max(xmin, cx - radius)) & (x_centers < min(xmax, cx + radius)) &
                             (y_centers > max(ymin, cy - radius)) & (y_centers < min(ymax, cy + radius)))

                max_dist = dists.max(dim=-1).values
                in_level = (max_dist >= size_lo) & (max_dist < size_hi)

                assign_here = inside_box & in_center & in_level & (box_area < assigned_area[b])
                if assign_here.any():
                    batch_cls[b, assign_here] = int(label) + 1
                    batch_mask[b, assign_here] = 1.0
                    batch_reg[b, assign_here] = box
                    assigned_area[b, assign_here] = box_area

        cls_targets_all.append(batch_cls)
        reg_targets_all.append(batch_reg)
        mask_targets_all.append(batch_mask)
    return cls_targets_all, reg_targets_all, mask_targets_all


# ============================================================
# Losses: M3.1's per-positive-normalized classification loss, unchanged objectness/centerness
# ============================================================

def softmax_focal_loss(logits, targets, gamma=2.0):
    """M3.1 fix: SUM over all locations (normalized by the positives in compute_losses),
    not .mean() -- the bug this fixes diluted the gradient on real objects by ~100x."""
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1 - pt) ** gamma) * log_pt
    return loss.sum()


def objectness_loss(obj_logits, pos_mask):
    logits = obj_logits.squeeze(1)
    targets = pos_mask.float()
    loss = ops.sigmoid_focal_loss(logits, targets, alpha=OBJ_FOCAL_ALPHA, gamma=2.0, reduction="sum")
    return loss / max(int(pos_mask.sum().item()), 1)


def centerness_loss(ctr_logits, target_boxes, x_centers, y_centers, pos_mask):
    boxes = target_boxes[pos_mask]
    px = x_centers[pos_mask]
    py = y_centers[pos_mask]
    l = px - boxes[:, 0]
    t = py - boxes[:, 1]
    r = boxes[:, 2] - px
    b = boxes[:, 3] - py
    target = torch.sqrt((torch.minimum(l, r) / torch.maximum(l, r)) *
                         (torch.minimum(t, b) / torch.maximum(t, b)))
    return F.binary_cross_entropy_with_logits(ctr_logits.squeeze(1)[pos_mask], target)


def compute_losses(cls_preds, bbox_preds, obj_preds, ctr_preds, images, targets_list, strides=(4, 8, 16, 32)):
    cls_targets, reg_targets, mask_targets = get_targets(images, targets_list, strides)
    num_pos_total = max(int(sum(float(m.sum()) for m in mask_targets)), 1)

    total_cls = 0.0
    total_bbox = 0.0
    total_obj = 0.0
    total_ctr = 0.0

    for l_idx, stride in enumerate(strides):
        pred_cls = cls_preds[l_idx]
        pred_reg = bbox_preds[l_idx]
        pred_obj = obj_preds[l_idx]
        pred_ctr = ctr_preds[l_idx]
        target_cls = cls_targets[l_idx]
        target_reg = reg_targets[l_idx]
        pos_mask = mask_targets[l_idx] == 1.0

        total_cls += softmax_focal_loss(pred_cls, target_cls, gamma=2.0)
        total_obj += objectness_loss(pred_obj, pos_mask)

        if pos_mask.any():
            B, _, H_l, W_l = pred_reg.shape
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H_l, device=images.device), torch.arange(W_l, device=images.device), indexing="ij")
            x_centers = ((x_grid + 0.5) * stride).unsqueeze(0).expand(B, -1, -1)
            y_centers = ((y_grid + 0.5) * stride).unsqueeze(0).expand(B, -1, -1)
            offsets = F.softplus(pred_reg.permute(0, 2, 3, 1)) * stride
            pred_boxes = torch.stack([
                x_centers - offsets[..., 0], y_centers - offsets[..., 1],
                x_centers + offsets[..., 2], y_centers + offsets[..., 3]], dim=-1)
            total_bbox += ops.complete_box_iou_loss(pred_boxes[pos_mask], target_reg[pos_mask], reduction="mean")
            total_ctr += centerness_loss(pred_ctr, target_reg, x_centers, y_centers, pos_mask)
        else:
            total_bbox += pred_reg.sum() * 0.0
            total_ctr += pred_ctr.sum() * 0.0

    n = len(strides)
    # M3.1 fix: classification loss normalized by positives, not by level count.
    return total_cls / num_pos_total, total_bbox / n, total_obj / n, total_ctr / n


class M4Trainer:
    def __init__(self, model, lr=1e-4, weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader, device):
        self.model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_bbox = 0.0
        epoch_obj = 0.0
        epoch_ctr = 0.0
        for batch_idx, (images, targets_list) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            _, cls_preds, bbox_preds, obj_preds, ctr_preds = self.model(images)
            loss_cls, loss_bbox, loss_obj, loss_ctr = compute_losses(
                cls_preds, bbox_preds, obj_preds, ctr_preds, images, targets_list)
            total_loss = loss_cls + loss_bbox + LAMBDA_OBJ * loss_obj + LAMBDA_CTR * loss_ctr
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            epoch_loss += total_loss.item()
            epoch_cls += loss_cls.item()
            epoch_bbox += loss_bbox.item()
            epoch_obj += loss_obj.item()
            epoch_ctr += loss_ctr.item()
            if batch_idx % 20 == 0:
                print(f"batch {batch_idx:04d} | loss {total_loss.item():.4f} "
                      f"(cls {loss_cls.item():.4f} / bbox {loss_bbox.item():.4f} / "
                      f"obj {loss_obj.item():.4f} / ctr {loss_ctr.item():.4f})")
        n = len(dataloader)
        return epoch_loss / n, epoch_cls / n, epoch_bbox / n, epoch_obj / n, epoch_ctr / n


# ============================================================
# Decoding: FIXED (best of the 60 real classes; background cannot win) x cls*obj*ctr score
# ============================================================

@torch.no_grad()
def decode_predictions(model, img_tensor, device, conf_thresh=0.05, iou_thresh=0.4):
    model.eval()
    x = img_tensor.unsqueeze(0).to(device)
    outputs = model(x)
    cls_preds, bbox_preds = outputs[1], outputs[2]
    obj_preds = outputs[3] if len(outputs) > 3 else None
    ctr_preds = outputs[4] if len(outputs) > 4 else None
    strides = [4, 8, 16, 32]
    all_boxes, all_scores, all_labels = [], [], []

    for l_idx, stride in enumerate(strides):
        pred_cls = cls_preds[l_idx][0]
        pred_reg = bbox_preds[l_idx][0]
        H_l, W_l = pred_cls.shape[1:]
        probs = torch.softmax(pred_cls, dim=0)
        # FIX: best of the 60 real classes (index 0 = background). The old argmax over all 61
        # outputs let background win and silently removed objects whose probability was spread
        # over similar classes.
        best_score, best_class = probs[1:].max(dim=0)
        best_class = best_class + 1
        if obj_preds is not None:
            best_score = best_score * torch.sigmoid(obj_preds[l_idx][0, 0])
        if ctr_preds is not None:
            best_score = best_score * torch.sigmoid(ctr_preds[l_idx][0, 0])
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H_l, device=device), torch.arange(W_l, device=device), indexing="ij")
        x_centers = (x_grid + 0.5) * stride
        y_centers = (y_grid + 0.5) * stride
        offsets = F.softplus(pred_reg.permute(1, 2, 0)) * stride
        boxes = torch.stack([
            x_centers - offsets[..., 0], y_centers - offsets[..., 1],
            x_centers + offsets[..., 2], y_centers + offsets[..., 3]], dim=-1)
        all_boxes.append(boxes.reshape(-1, 4))
        all_scores.append(best_score.reshape(-1))
        all_labels.append(best_class.reshape(-1))

    flat_boxes = torch.cat(all_boxes, dim=0)
    flat_scores = torch.cat(all_scores, dim=0)
    flat_labels = torch.cat(all_labels, dim=0)

    keep = (flat_labels != 0) & (flat_scores > conf_thresh)
    filtered_boxes = flat_boxes[keep]
    filtered_scores = flat_scores[keep]
    filtered_labels = flat_labels[keep]

    if len(filtered_boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))

    filtered_boxes[:, 0].clamp_(0, img_tensor.shape[-1])
    filtered_boxes[:, 2].clamp_(0, img_tensor.shape[-1])
    filtered_boxes[:, 1].clamp_(0, img_tensor.shape[-2])
    filtered_boxes[:, 3].clamp_(0, img_tensor.shape[-2])

    keep_idxs = ops.batched_nms(filtered_boxes, filtered_scores, filtered_labels, iou_threshold=iou_thresh)
    return (filtered_boxes[keep_idxs].cpu().numpy(), filtered_scores[keep_idxs].cpu().numpy(),
            filtered_labels[keep_idxs].cpu().numpy())


def get_display_rgb(img_tensor):
    rgb = img_tensor[:3].cpu().clone()
    rgb = rgb * IMAGENET_STD + IMAGENET_MEAN
    rgb = rgb.clamp(0, 1)
    return (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def run_predictions_and_visualize(model, dataset, device, output_path, conf_thresh, dataset_idx):
    img_tensor, targets = dataset[dataset_idx]
    boxes, scores, labels = decode_predictions(model, img_tensor, device, conf_thresh, NMS_IOU_THRESHOLD)
    img_np = get_display_rgb(img_tensor)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    gt_canvas = img_np.copy()
    gt_boxes = targets["boxes"].numpy()
    gt_labels = targets["labels"].numpy()
    for box, label in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(gt_canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    axes[0].imshow(gt_canvas)
    axes[0].set_title(f"Ground Truth | {len(gt_boxes)} objects")
    axes[0].axis("off")

    pred_canvas = img_np.copy()
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(pred_canvas, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.putText(pred_canvas, f"{score:.2f}", (xmin, max(12, ymin - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    axes[1].imshow(pred_canvas)
    axes[1].set_title(f"Predictions | {len(boxes)} boxes")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization: {output_path}")
    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"Predicted boxes: {len(boxes)}")


# ============================================================
# Evaluation (unchanged from M3.1)
# ============================================================

def make_fixed_test_indices(dataset, num_chips, seed):
    rng = random.Random(seed)
    n = len(dataset)
    num = min(num_chips, n)
    return rng.sample(range(n), num)


@torch.no_grad()
def evaluate_dataset(model, dataset, device, test_indices, conf_thresh, iou_match_thresh):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_gt = 0
    iou_sum = 0.0
    iou_count = 0

    for idx in test_indices:
        img_tensor, targets = dataset[idx]
        pred_boxes, pred_scores, pred_labels = decode_predictions(
            model, img_tensor, device, conf_thresh, NMS_IOU_THRESHOLD)
        gt_boxes = targets["boxes"].numpy()
        gt_labels = targets["labels"].numpy() + 1
        total_gt += len(gt_boxes)
        matched_gt = set()
        order = np.argsort(-pred_scores)

        for pi in order:
            p_box = pred_boxes[pi]
            p_label = pred_labels[pi]
            candidate_indices = [gi for gi in range(len(gt_boxes))
                                  if gt_labels[gi] == p_label and gi not in matched_gt]
            if not candidate_indices:
                total_fp += 1
                continue
            candidate_boxes = torch.tensor(gt_boxes[candidate_indices], dtype=torch.float32)
            ious = ops.box_iou(torch.tensor(p_box, dtype=torch.float32).unsqueeze(0), candidate_boxes)[0]
            best_iou, best_idx = ious.max(dim=0)
            if best_iou.item() >= iou_match_thresh:
                total_tp += 1
                matched_gt.add(candidate_indices[best_idx.item()])
                iou_sum += best_iou.item()
                iou_count += 1
            else:
                total_fp += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    mean_iou = iou_sum / max(iou_count, 1)

    print(f"\nEvaluation on {len(test_indices)} fixed chips")
    print(f"GT objects: {total_gt}")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Mean matched IoU: {mean_iou:.3f}")

    return {"precision": precision, "recall": recall, "f1": f1, "mean_iou": mean_iou,
            "true_positives": total_tp, "false_positives": total_fp, "ground_truth": total_gt}


@torch.no_grad()
def evaluate_coco(model, dataset, device, test_indices, conf_thresh=MAP_CONF_THRESHOLD):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.warn_on_many_detections = False
    for idx in test_indices:
        img_tensor, targets = dataset[idx]
        pred_boxes, pred_scores, pred_labels = decode_predictions(
            model, img_tensor, device, conf_thresh, NMS_IOU_THRESHOLD)
        preds = [{"boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                  "scores": torch.tensor(pred_scores, dtype=torch.float32),
                  "labels": torch.tensor(pred_labels, dtype=torch.long)}]
        gts = [{"boxes": targets["boxes"], "labels": targets["labels"] + 1}]
        metric.update(preds, gts)
    r = metric.compute()
    return {"map": float(r["map"]), "map_50": float(r["map_50"]), "map_75": float(r["map_75"]),
            "map_small": float(r["map_small"]), "map_medium": float(r["map_medium"]),
            "map_large": float(r["map_large"])}


def count_parameters_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gflops(model, device):
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def measure_fps(model, dataset, device, test_indices, conf_thresh):
    chips = [dataset[i][0] for i in test_indices[:NUM_FPS_CHIPS]]
    for img in chips[:5]:
        decode_predictions(model, img, device, conf_thresh, NMS_IOU_THRESHOLD)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for img in chips:
        decode_predictions(model, img, device, conf_thresh, NMS_IOU_THRESHOLD)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return len(chips) / (time.time() - start)


def full_report(name, model, dataset, device, test_indices):
    print(f"\n################ {name} ################")

    sweep = {}
    for threshold in CONF_THRESHOLDS:
        print(f"\nThreshold = {threshold}")
        sweep[threshold] = evaluate_dataset(model, dataset, device, test_indices, threshold, IOU_MATCH_THRESHOLD)

    print("\nThreshold | Precision | Recall | F1 | Mean IoU")
    for threshold in CONF_THRESHOLDS:
        r = sweep[threshold]
        print(f"{threshold:9.2f} | {r['precision']:9.3f} | {r['recall']:6.3f} | {r['f1']:4.3f} | {r['mean_iou']:8.3f}")

    best_thresh = max(sweep, key=lambda t: sweep[t]["f1"])
    best = sweep[best_thresh]
    print(f"\nBest F1 threshold: {best_thresh:.2f}")

    coco = evaluate_coco(model, dataset, device, test_indices)
    report = {"best_thresh": best_thresh, "precision": best["precision"], "recall": best["recall"],
              "f1": best["f1"], "mean_iou": best["mean_iou"],
              "fps": measure_fps(model, dataset, device, test_indices, best_thresh),
              "params_m": count_parameters_m(model), "gflops": count_gflops(model, device)}
    report.update(coco)
    return report


def print_comparison(reports):
    rows = [("mAP50:95", "map"), ("AP50", "map_50"), ("AP75", "map_75"),
            ("AP_small", "map_small"), ("AP_medium", "map_medium"), ("AP_large", "map_large"),
            ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("Mean IoU", "mean_iou"),
            ("Best conf thr", "best_thresh"), ("FPS", "fps"), ("Params (M)", "params_m"), ("GFLOPs", "gflops")]
    names = list(reports.keys())
    print("\n" + "=" * (22 + 12 * len(names)))
    print(f"{'Metric':<20}" + "".join(f"{n:>12}" for n in names))
    print("-" * (22 + 12 * len(names)))
    for label, key in rows:
        print(f"{label:<20}" + "".join(f"{reports[n][key]:>12.3f}" for n in names))
    print("=" * (22 + 12 * len(names)))
    print("AP values of -1 mean no ground-truth objects of that size exist in the test chips.")


def find_checkpoint(filename):
    for root in ("/kaggle/working", "/kaggle/input"):
        hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if hits:
            return hits[0]
    return None


# Earlier models re-evaluated with the SAME evaluator: (name, checkpoint file, head class).
# M3.1's checkpoint is included so the M4 run reports a direct, same-session comparison --
# exactly how M3.1 reported against M3.
PREVIOUS_MODELS = [
    ("M0", "cpnet_xview_m0_checkpoint.pt", M0Detector),
    ("M1", "cpnet_xview_m1_checkpoint.pt", M0Detector),
    ("M2", "cpnet_xview_m2_checkpoint.pt", M2Detector),
    ("M3.1", "cpnet_xview_m31_checkpoint.pt", M3Detector),
]


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable -- CPNet is too heavy to finish 25 epochs on CPU "
              "within Kaggle's session limit (see camouflage-detection notebook: it hit the "
              "12-hour CPU timeout at batch 1490 of epoch 1). Attach a GPU before running this.")

    input_root = "/kaggle/input"
    if not os.path.exists(input_root):
        raise FileNotFoundError("Attach the xView dataset.")

    probe = XViewDataset(root_dir=input_root, chip_size=CHIP_SIZE, img_size=IMG_SIZE,
                          chips_per_image=1, seed=RANDOM_SEED)
    all_ids = list(probe.image_ids)
    split_rng = random.Random(RANDOM_SEED)
    split_rng.shuffle(all_ids)
    n_test = max(1, int(len(all_ids) * TEST_FRACTION))
    test_ids = set(all_ids[:n_test])
    train_ids = set(all_ids[n_test:])
    print(f"Train images: {len(train_ids)}")
    print(f"Test images: {len(test_ids)}")
    assert train_ids.isdisjoint(test_ids)

    train_dataset = XViewDataset(root_dir=input_root, chip_size=CHIP_SIZE, img_size=IMG_SIZE,
                                  chips_per_image=4, positive_chip_prob=0.5, seed=RANDOM_SEED,
                                  allowed_image_ids=train_ids, deterministic=False)

    test_dataset = XViewDataset(root_dir=input_root, chip_size=CHIP_SIZE, img_size=IMG_SIZE,
                                 chips_per_image=2, positive_chip_prob=0.5, seed=TEST_SAMPLE_SEED,
                                 allowed_image_ids=test_ids, deterministic=True)

    print(f"Train chips: {len(train_dataset)}")
    print(f"Test chips: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available())

    model = ResNetDetector(M4Detector, pretrained=True, num_classes_total=NUM_CLASSES_TOTAL,
                            weights_path=IMAGENET_WEIGHTS_PATH).to(device)
    init_background_prior(model, p_background=BACKGROUND_PRIOR_PROB)
    trainer = M4Trainer(model, lr=LR, weight_decay=WEIGHT_DECAY)

    print("Input: 3-channel RGB")
    print(f"Input resolution: {IMG_SIZE} x {IMG_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}  (full length -- every earlier CPNet run in this project was 5 epochs)")
    print("Optimizer: AdamW")
    print(f"Learning rate: {LR}")
    print(f"Assignment: center sampling (radius {CENTER_RADIUS} x stride), level by max(l, r, t, b)")
    print("Pyramid: CPNet Cascade Aggregation Pyramid + Ternary Cascade Perception Module (replaces plain FPN)")
    print(f"Objectness: focal BCE, lambda {LAMBDA_OBJ}")
    print(f"Centerness: BCE on positives, lambda {LAMBDA_CTR}; score = class x objectness x centerness")
    print("Classification loss: per-positive normalized (M3.1 fix carried forward)")
    print("Decoding: fixed (best of 60 real classes; background excluded from the argmax)")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        avg_loss, avg_cls, avg_bbox, avg_obj, avg_ctr = trainer.train_epoch(train_loader, device)
        print(f"Epoch {epoch} summary:")
        print(f"  Total loss: {avg_loss:.4f}")
        print(f"  Classification loss: {avg_cls:.4f}")
        print(f"  Bounding-box loss: {avg_bbox:.4f}")
        print(f"  Objectness loss: {avg_obj:.4f}")
        print(f"  Centerness loss: {avg_ctr:.4f}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(), "loss": avg_loss}, CHECKPOINT_PATH)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    test_indices = make_fixed_test_indices(test_dataset, NUM_TEST_CHIPS, TEST_SAMPLE_SEED)
    print(f"Using {len(test_indices)} fixed test chips.")
    print("First indices:", test_indices[:10])

    visual_idx = test_indices[0]
    run_predictions_and_visualize(model, test_dataset, device, VISUALIZATION_PATH,
                                   conf_thresh=0.10, dataset_idx=visual_idx)

    reports = {}
    reports["M4"] = full_report("M4 (CPNet)", model, test_dataset, device, test_indices)

    run_predictions_and_visualize(model, test_dataset, device, TUNED_VISUALIZATION_PATH,
                                   conf_thresh=reports["M4"]["best_thresh"], dataset_idx=visual_idx)

    # Re-evaluate the earlier checkpoints with the SAME evaluator (no retraining) -- the same
    # same-run comparison pattern M3.1 used against M3.
    for name, filename, head_cls in PREVIOUS_MODELS:
        path = find_checkpoint(filename)
        if path is None:
            print(f"\n{name} checkpoint ({filename}) not found in /kaggle/working or /kaggle/input -- skipping {name}.")
            continue
        old_model = ResNetDetector(head_cls, pretrained=False, num_classes_total=NUM_CLASSES_TOTAL).to(device)
        old_model.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
        reports[name] = full_report(name, old_model, test_dataset, device, test_indices)

    ordered = {k: reports[k] for k in ("M0", "M1", "M2", "M3.1", "M4") if k in reports}
    print_comparison(ordered)

    print("\nCOMPLETE")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Visualization: {VISUALIZATION_PATH}")
    print(f"Tuned visualization: {TUNED_VISUALIZATION_PATH}")