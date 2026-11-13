"""
M1 -- M0 + corrected center-based target assignment.
========================================================================
Everything is identical to M0 (ResNet50 + plain FPN + bare head, 512x512 RGB,
AdamW, LR 1e-4, 25 epochs, positive_chip_prob 0.5, same 80/20 split, same
100 fixed test chips, same NMS, same loss).

ONLY CHANGE: get_targets().

M0 assignment:
    point inside GT box
    FPN level chosen by the box's max(width, height)

M1 assignment (FCOS style):
    point inside GT box
    AND point inside a center-sampling region (radius 1.5 * stride, clipped to box)
    FPN level chosen by max(l, r, t, b) of the point
    smallest-area GT wins if boxes overlap

The level ranges (SIZE_RANGES) are kept exactly as in M0. They already equal
8 * stride, which is the FCOS range for max(l, r, t, b).

The evaluation additionally reports COCO metrics (torchmetrics), parameters,
GFLOPs and FPS. If the M0 checkpoint exists it is evaluated with the SAME
evaluator, so the M0 / M1 comparison is apples-to-apples.

Run with:  python xview_m1_center_assignment.py
"""

import os
import glob
import json
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
CONF_THRESHOLDS = [0.30, 0.45, 0.50, 0.60]
IOU_MATCH_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.40
MAP_CONF_THRESHOLD = 0.05      # low threshold so the AP curve is not truncated
CENTER_RADIUS = 1.5            # M1: center-sampling radius in units of stride
NUM_FPS_CHIPS = 50

M0_CHECKPOINT_PATH = "/kaggle/working/cpnet_xview_m0_checkpoint.pt"   # optional, for comparison
CHECKPOINT_PATH = "/kaggle/working/cpnet_xview_m1_checkpoint.pt"
VISUALIZATION_PATH = "/kaggle/working/xview_m1_prediction_result.png"
TUNED_VISUALIZATION_PATH = "/kaggle/working/xview_m1_prediction_tuned.png"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# ============================================================
# ARCHITECTURE (unchanged from M0): ResNet50 + plain FPN + bare head
# ============================================================

class ConvBNReLU(nn.Module):
    """Plain conv -> batchnorm -> relu. Used only for FPN's smoothing step."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    """
    Standard Feature Pyramid Network (Lin et al., 2017).
    1x1 lateral projection -> top-down upsample+add -> 3x3 smoothing.
    """
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
    def __init__(self, pretrained=True):
        super().__init__()
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
    """Plain FPN + one classification conv and one bbox conv per pyramid level."""
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


class ResNetM0(nn.Module):
    def __init__(self, pretrained=True, out_channels=256, num_classes_total=61):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.model = M0Detector([256, 512, 1024, 2048], out_channels, num_classes_total)

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
# M1 CHANGE: center-based target assignment
# ============================================================
# Same level ranges as M0, but now applied to max(l, r, t, b) of each point
# instead of the box's max(width, height).
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

                # distances from every grid point to the four box sides
                l = x_centers - xmin
                t = y_centers - ymin
                r = xmax - x_centers
                bt = ymax - y_centers
                dists = torch.stack([l, t, r, bt], dim=-1)

                # 1) point must be inside the GT box
                inside_box = dists.min(dim=-1).values > 0

                # 2) point must be inside the center-sampling region (clipped to the box)
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                radius = CENTER_RADIUS * stride
                in_center = ((x_centers > max(xmin, cx - radius)) & (x_centers < min(xmax, cx + radius)) &
                             (y_centers > max(ymin, cy - radius)) & (y_centers < min(ymax, cy + radius)))

                # 3) FPN level chosen by max(l, r, t, b)
                max_dist = dists.max(dim=-1).values
                in_level = (max_dist >= size_lo) & (max_dist < size_hi)

                # 4) smallest-area GT wins if boxes overlap
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
# Loss, trainer, decoding (unchanged from M0)
# ============================================================

def softmax_focal_loss(logits, targets, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1 - pt) ** gamma) * log_pt
    return loss.mean()


def compute_losses(cls_preds, bbox_preds, images, targets_list, strides=(4, 8, 16, 32)):
    cls_targets, reg_targets, mask_targets = get_targets(images, targets_list, strides)
    total_cls = 0.0
    total_bbox = 0.0

    for l_idx, stride in enumerate(strides):
        pred_cls = cls_preds[l_idx]
        pred_reg = bbox_preds[l_idx]
        target_cls = cls_targets[l_idx]
        target_reg = reg_targets[l_idx]
        pos_mask = mask_targets[l_idx] == 1.0

        total_cls += softmax_focal_loss(pred_cls, target_cls, gamma=2.0)

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
        else:
            total_bbox += pred_reg.sum() * 0.0

    n = len(strides)
    return total_cls / n, total_bbox / n


class M1Trainer:
    def __init__(self, model, lr=1e-4, weight_decay=1e-4):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader, device):
        self.model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_bbox = 0.0
        for batch_idx, (images, targets_list) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            _, cls_preds, bbox_preds = self.model(images)
            loss_cls, loss_bbox = compute_losses(cls_preds, bbox_preds, images, targets_list)
            total_loss = loss_cls + loss_bbox
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            epoch_loss += total_loss.item()
            epoch_cls += loss_cls.item()
            epoch_bbox += loss_bbox.item()
            if batch_idx % 20 == 0:
                print(f"batch {batch_idx:04d} | loss {total_loss.item():.4f} "
                      f"(cls {loss_cls.item():.4f} / bbox {loss_bbox.item():.4f})")
        n = len(dataloader)
        return epoch_loss / n, epoch_cls / n, epoch_bbox / n


@torch.no_grad()
def decode_predictions(model, img_tensor, device, conf_thresh=0.3, iou_thresh=0.4):
    model.eval()
    x = img_tensor.unsqueeze(0).to(device)
    _, cls_preds, bbox_preds = model(x)
    strides = [4, 8, 16, 32]
    all_boxes, all_scores, all_labels = [], [], []

    for l_idx, stride in enumerate(strides):
        pred_cls = cls_preds[l_idx][0]
        pred_reg = bbox_preds[l_idx][0]
        H_l, W_l = pred_cls.shape[1:]
        probs = torch.softmax(pred_cls, dim=0)
        best_score, best_class = probs.max(dim=0)
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
# Evaluation
# ============================================================

def make_fixed_test_indices(dataset, num_chips, seed):
    rng = random.Random(seed)
    n = len(dataset)
    num = min(num_chips, n)
    return rng.sample(range(n), num)


@torch.no_grad()
def evaluate_dataset(model, dataset, device, test_indices, conf_thresh, iou_match_thresh):
    """Custom Precision / Recall / F1 at one confidence threshold (same as M0)."""
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
    """COCO-style AP via torchmetrics. Labels are 1..60 (0 = background, never predicted)."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
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
    """FLOPs for one 512x512 forward pass (a multiply-add counts as 2 FLOPs)."""
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def measure_fps(model, dataset, device, test_indices, conf_thresh=0.30):
    """End-to-end FPS: forward + decode + NMS, one chip at a time."""
    chips = [dataset[i][0] for i in test_indices[:NUM_FPS_CHIPS]]
    for img in chips[:5]:                      # warm-up
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
    """Runs every metric on one model and returns a flat dict."""
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
              "fps": measure_fps(model, dataset, device, test_indices),
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


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable.")

    input_root = "/kaggle/input"
    if not os.path.exists(input_root):
        raise FileNotFoundError("Attach xView dataset.")

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

    model = ResNetM0(pretrained=True, num_classes_total=NUM_CLASSES_TOTAL).to(device)
    trainer = M1Trainer(model, lr=LR, weight_decay=WEIGHT_DECAY)

    print("Input: 3-channel RGB")
    print(f"Input resolution: {IMG_SIZE} x {IMG_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("Optimizer: AdamW")
    print(f"Learning rate: {LR}")
    print(f"Assignment: center sampling (radius {CENTER_RADIUS} x stride), level by max(l, r, t, b)")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n========== Epoch {epoch}/{NUM_EPOCHS} ==========")
        avg_loss, avg_cls, avg_bbox = trainer.train_epoch(train_loader, device)
        print(f"Epoch {epoch} summary:")
        print(f"  Total loss: {avg_loss:.4f}")
        print(f"  Classification loss: {avg_cls:.4f}")
        print(f"  Bounding-box loss: {avg_bbox:.4f}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(), "loss": avg_loss}, CHECKPOINT_PATH)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    test_indices = make_fixed_test_indices(test_dataset, NUM_TEST_CHIPS, TEST_SAMPLE_SEED)
    print(f"Using {len(test_indices)} fixed test chips.")
    print("First indices:", test_indices[:10])

    visual_idx = test_indices[0]
    run_predictions_and_visualize(model, test_dataset, device, VISUALIZATION_PATH,
                                   conf_thresh=0.30, dataset_idx=visual_idx)

    reports = {}
    reports["M1"] = full_report("M1", model, test_dataset, device, test_indices)

    run_predictions_and_visualize(model, test_dataset, device, TUNED_VISUALIZATION_PATH,
                                   conf_thresh=reports["M1"]["best_thresh"], dataset_idx=visual_idx)

    # Evaluate the existing M0 checkpoint with the SAME evaluator (no retraining).
    if os.path.exists(M0_CHECKPOINT_PATH):
        m0_model = ResNetM0(pretrained=False, num_classes_total=NUM_CLASSES_TOTAL).to(device)
        m0_model.load_state_dict(torch.load(M0_CHECKPOINT_PATH, map_location=device)["model_state_dict"])
        reports["M0"] = full_report("M0", m0_model, test_dataset, device, test_indices)
    else:
        print(f"\nM0 checkpoint not found at {M0_CHECKPOINT_PATH} -- skipping M0 comparison.")

    ordered = {k: reports[k] for k in ("M0", "M1") if k in reports}
    print_comparison(ordered)

    print("\nCOMPLETE")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Visualization: {VISUALIZATION_PATH}")
    print(f"Tuned visualization: {TUNED_VISUALIZATION_PATH}")