"""
Step 1 of the post-M3.1 experiment plan: freeze and audit M3.1.
========================================================================
PURPOSE
  Produce a frozen, reproducible audit of the M3.1 checkpoint -- NO RETRAINING, and no
  change to the model, the dataset split, the decoding, or the metric implementation.
  Everything here is copied verbatim from train_m3_1.py so the numbers below match what
  that script would print; this file only adds reporting (full per-class AP instead of
  top 12, a saved precision/recall sweep, and color-coded qualitative examples) and
  writes everything to disk instead of only stdout.

WHAT THIS AUDITS (and does NOT change)
  - Eval pipeline: same XViewDataset, same fixed 80/20 image-level split (seed 42),
    same 100 fixed test chips (seed 12345), same confidence sweep.
  - Metric implementation: same torchmetrics MeanAveragePrecision usage, same label
    space (1..60, 0 reserved for background), same custom greedy precision/recall/F1.
  - Class mapping: same XVIEW_CLASS_REMAP / XVIEW_CLASS_NAMES.
  - Decoding: same fixed decode (probs[1:].max(dim=0), i.e. argmax restricted to the
    60 real classes, excluding background) x objectness x centerness.

OUTPUTS (written under /kaggle/working/audit_m3_1/, copy this folder out after running)
  - per_class_ap.csv            every class present in the test chips, not just top 12
  - ap_by_size.json             AP50:95 for small / medium / large (+ overall mAP/AP50/AP75)
  - precision_recall_sweep.csv  one row per confidence threshold (precision/recall/F1/IoU)
  - summary.json                headline numbers in one place
  - qualitative/*.png           N sample chips: GT (green) vs predictions color-coded
                                 TP = green, FP = red, FN (missed GT) = yellow

HOW TO RUN
  Attach as inputs: the xView dataset, and the M3.1 checkpoint
  (cpnet_xview_m31_checkpoint.pt, from the M3.1 notebook's output). Internet can be OFF --
  no ImageNet weights are needed, since this only loads the already-trained checkpoint.
  GPU is not required (this is eval-only over 100 test chips) but speeds it up.
  If the checkpoint is not found, the script raises immediately rather than training.

  python audit_m3_1.py
"""

import os
import csv
import glob
import json
import math
import random

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.ops as ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

# ---------------- unchanged settings, copied verbatim from train_m3_1.py ----------------
IMG_SIZE = 512
CHIP_SIZE = 512
TEST_FRACTION = 0.20
RANDOM_SEED = 42
TEST_SAMPLE_SEED = 12345
NUM_TEST_CHIPS = 100
IOU_MATCH_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.40
MAP_CONF_THRESHOLD = 0.01
CONF_THRESHOLDS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]

CHECKPOINT_FILENAME = "cpnet_xview_m31_checkpoint.pt"
OUTPUT_DIR = "/kaggle/working/audit_m3_1"
NUM_QUALITATIVE_EXAMPLES = 10

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ============================================================
# Architecture (copied verbatim from train_m3_1.py -- only M3Detector is needed, since
# M3.1 uses the same head class as M3; the change was the loss function during training,
# which has no effect on the frozen checkpoint's architecture).
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
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


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False, weights_path=None):
        super().__init__()
        # Eval-only: never need ImageNet weights, we load a trained checkpoint below.
        resnet = models.resnet50(weights=None)
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
    def __init__(self, in_channels_list, out_channels=256, num_classes_total=61):
        super().__init__(in_channels_list, out_channels, num_classes_total)
        self.obj_heads = nn.ModuleList([nn.Conv2d(out_channels, 1, 3, padding=1) for _ in range(4)])

    def forward(self, backbone_features):
        pyramid, cls_preds, bbox_preds = super().forward(backbone_features)
        obj_preds = [head(f) for head, f in zip(self.obj_heads, pyramid)]
        return pyramid, cls_preds, bbox_preds, obj_preds


class M3Detector(M2Detector):
    def __init__(self, in_channels_list, out_channels=256, num_classes_total=61):
        super().__init__(in_channels_list, out_channels, num_classes_total)
        self.ctr_heads = nn.ModuleList([nn.Conv2d(out_channels, 1, 3, padding=1) for _ in range(4)])

    def forward(self, backbone_features):
        pyramid, cls_preds, bbox_preds, obj_preds = super().forward(backbone_features)
        ctr_preds = [head(f) for head, f in zip(self.ctr_heads, pyramid)]
        return pyramid, cls_preds, bbox_preds, obj_preds, ctr_preds


class ResNetDetector(nn.Module):
    def __init__(self, head_cls, num_classes_total=61):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=False)
        self.model = head_cls([256, 512, 1024, 2048], 256, num_classes_total)

    def forward(self, x):
        return self.model(self.backbone(x))


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
    """Copied verbatim from train_m3_1.py (eval path only -- deterministic chipping)."""

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


# ============================================================
# Decoding -- copied verbatim from train_m3_1.py's fixed decode.
# ============================================================

@torch.no_grad()
def decode_predictions(model, img_tensor, device, conf_thresh=0.3, iou_thresh=0.4):
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


def make_fixed_test_indices(dataset, num_chips, seed):
    rng = random.Random(seed)
    n = len(dataset)
    num = min(num_chips, n)
    return rng.sample(range(n), num)


# ============================================================
# Audit reporting (new in this file -- everything below writes to disk)
# ============================================================

@torch.no_grad()
def evaluate_dataset(model, dataset, device, test_indices, conf_thresh, iou_match_thresh):
    """Same custom greedy precision/recall/F1 as train_m3_1.py."""
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


@torch.no_grad()
def full_per_class_ap(model, dataset, device, test_indices):
    """Every class present in the test chips (not just a top-N) -- written to CSV."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric.warn_on_many_detections = False
    gt_counts = {}
    for idx in test_indices:
        img_tensor, targets = dataset[idx]
        pb, ps, pl = decode_predictions(model, img_tensor, device, MAP_CONF_THRESHOLD, NMS_IOU_THRESHOLD)
        gt_labels = targets["labels"] + 1
        metric.update([{"boxes": torch.tensor(pb, dtype=torch.float32), "scores": torch.tensor(ps, dtype=torch.float32),
                        "labels": torch.tensor(pl, dtype=torch.long)}],
                      [{"boxes": targets["boxes"], "labels": gt_labels}])
        for l in gt_labels.tolist():
            gt_counts[l] = gt_counts.get(l, 0) + 1
    r = metric.compute()
    class_ap = dict(zip([int(c) for c in r["classes"].tolist()], [float(v) for v in r["map_per_class"].tolist()]))
    rows = []
    for cls_idx in sorted(gt_counts, key=gt_counts.get, reverse=True):
        rows.append({
            "class_id": cls_idx,
            "class_name": XVIEW_CLASS_NAMES[cls_idx - 1],
            "gt_count": gt_counts[cls_idx],
            "ap_50_95": round(class_ap.get(cls_idx, -1.0), 4),
        })
    with_ap = sum(1 for row in rows if row["ap_50_95"] > 0)
    return rows, with_ap, len(gt_counts)


def get_display_rgb(img_tensor):
    rgb = img_tensor[:3].cpu().clone()
    rgb = rgb * IMAGENET_STD + IMAGENET_MEAN
    rgb = rgb.clamp(0, 1)
    return (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def save_qualitative_tp_fp_fn(model, dataset, device, dataset_idx, conf_thresh, iou_match_thresh, out_path):
    """
    One chip, predictions matched against GT the same way evaluate_dataset does.
    Green = true positive, Red = false positive, Yellow (dashed-style thin box) = false
    negative (a GT object with no matching prediction).
    """
    img_tensor, targets = dataset[dataset_idx]
    pred_boxes, pred_scores, pred_labels = decode_predictions(model, img_tensor, device, conf_thresh, NMS_IOU_THRESHOLD)
    gt_boxes = targets["boxes"].numpy()
    gt_labels = targets["labels"].numpy() + 1

    matched_gt = set()
    pred_status = []  # "TP" or "FP" per prediction, aligned to pred_boxes order
    order = np.argsort(-pred_scores)
    status_by_pred_idx = {}
    for pi in order:
        p_box = pred_boxes[pi]
        p_label = pred_labels[pi]
        candidate_indices = [gi for gi in range(len(gt_boxes))
                              if gt_labels[gi] == p_label and gi not in matched_gt]
        if not candidate_indices:
            status_by_pred_idx[pi] = "FP"
            continue
        candidate_boxes = torch.tensor(gt_boxes[candidate_indices], dtype=torch.float32)
        ious = ops.box_iou(torch.tensor(p_box, dtype=torch.float32).unsqueeze(0), candidate_boxes)[0]
        best_iou, best_idx = ious.max(dim=0)
        if best_iou.item() >= iou_match_thresh:
            status_by_pred_idx[pi] = "TP"
            matched_gt.add(candidate_indices[best_idx.item()])
        else:
            status_by_pred_idx[pi] = "FP"

    img_np = get_display_rgb(img_tensor)
    canvas = img_np.copy()

    for gi, box in enumerate(gt_boxes):
        if gi not in matched_gt:
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 215, 0), 1)  # FN = yellow

    for pi, box in enumerate(pred_boxes):
        status = status_by_pred_idx.get(pi, "FP")
        color = (0, 200, 0) if status == "TP" else (220, 0, 0)  # TP = green, FP = red
        xmin, ymin, xmax, ymax = map(int, box)
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), color, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(canvas)
    n_tp = sum(1 for s in status_by_pred_idx.values() if s == "TP")
    n_fp = sum(1 for s in status_by_pred_idx.values() if s == "FP")
    n_fn = len(gt_boxes) - len(matched_gt)
    ax.set_title(f"TP={n_tp} (green)  FP={n_fp} (red)  FN={n_fn} (yellow)  @ conf={conf_thresh:.3f}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return n_tp, n_fp, n_fn


def find_checkpoint(filename):
    for root in ("/kaggle/working", "/kaggle/input"):
        hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if hits:
            return hits[0]
    return None


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "qualitative"), exist_ok=True)

    checkpoint_path = find_checkpoint(CHECKPOINT_FILENAME)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"{CHECKPOINT_FILENAME} not found under /kaggle/working or /kaggle/input. "
            "Attach the finished M3.1 checkpoint as an input -- this script never trains.")
    print(f"Loading frozen checkpoint: {checkpoint_path}")

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
    print(f"Test images: {len(test_ids)} (same 80/20 split as training, seed {RANDOM_SEED})")

    test_dataset = XViewDataset(root_dir=input_root, chip_size=CHIP_SIZE, img_size=IMG_SIZE,
                                 chips_per_image=2, positive_chip_prob=0.5, seed=TEST_SAMPLE_SEED,
                                 allowed_image_ids=test_ids, deterministic=True)
    test_indices = make_fixed_test_indices(test_dataset, NUM_TEST_CHIPS, TEST_SAMPLE_SEED)
    print(f"Test chips: {len(test_indices)} (same fixed set as training, seed {TEST_SAMPLE_SEED})")

    model = ResNetDetector(M3Detector, num_classes_total=NUM_CLASSES_TOTAL).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')} -- no training performed.")

    # ---- 1. precision/recall/F1 sweep ----
    print("\nRunning precision/recall/F1 sweep...")
    sweep_rows = []
    for threshold in CONF_THRESHOLDS:
        r = evaluate_dataset(model, test_dataset, device, test_indices, threshold, IOU_MATCH_THRESHOLD)
        sweep_rows.append({"conf_threshold": threshold, **r})
        print(f"  thr={threshold:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
    best_row = max(sweep_rows, key=lambda r: r["f1"])
    with open(os.path.join(OUTPUT_DIR, "precision_recall_sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"Best F1 = {best_row['f1']:.3f} at threshold {best_row['conf_threshold']:.3f}")

    # ---- 2. COCO mAP / AP-by-size ----
    print("\nRunning COCO mAP (overall + by size)...")
    coco = evaluate_coco(model, test_dataset, device, test_indices)
    with open(os.path.join(OUTPUT_DIR, "ap_by_size.json"), "w") as f:
        json.dump(coco, f, indent=2)
    print(json.dumps(coco, indent=2))

    # ---- 3. full per-class AP ----
    print("\nRunning full per-class AP (all classes present in the test chips)...")
    class_rows, with_ap, n_present = full_per_class_ap(model, test_dataset, device, test_indices)
    with open(os.path.join(OUTPUT_DIR, "per_class_ap.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "gt_count", "ap_50_95"])
        writer.writeheader()
        writer.writerows(class_rows)
    print(f"Classes with AP50:95 > 0: {with_ap} of {n_present} present in the test chips")

    # ---- 4. qualitative TP/FP/FN examples ----
    print(f"\nSaving {NUM_QUALITATIVE_EXAMPLES} qualitative TP/FP/FN examples...")
    qual_summary = []
    qual_indices = test_indices[:NUM_QUALITATIVE_EXAMPLES]
    for i, idx in enumerate(qual_indices):
        out_path = os.path.join(OUTPUT_DIR, "qualitative", f"example_{i:02d}_idx{idx}.png")
        n_tp, n_fp, n_fn = save_qualitative_tp_fp_fn(
            model, test_dataset, device, idx, best_row["conf_threshold"], IOU_MATCH_THRESHOLD, out_path)
        qual_summary.append({"example": i, "dataset_idx": idx, "tp": n_tp, "fp": n_fp, "fn": n_fn})
        print(f"  [{i}] idx={idx}  TP={n_tp} FP={n_fp} FN={n_fn}  -> {out_path}")

    # ---- 5. summary ----
    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": state.get("epoch", None),
        "best_conf_threshold": best_row["conf_threshold"],
        "precision": best_row["precision"],
        "recall": best_row["recall"],
        "f1": best_row["f1"],
        "mean_iou": best_row["mean_iou"],
        **coco,
        "classes_with_ap_gt_0": with_ap,
        "classes_present_in_test_chips": n_present,
        "qualitative_examples": qual_summary,
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCOMPLETE -- no training was performed.")
    print(f"All audit outputs written to: {OUTPUT_DIR}")
    print("Copy this folder out of /kaggle/working and add it to the repo under results/audit_m3_1/")
