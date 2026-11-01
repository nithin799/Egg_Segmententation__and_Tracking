"""Classical egg segmentation (final algorithm = trial T9b) and evaluation helpers.

Pipeline
  1. Candidate generation: for channels gray, HSV-S, Lab-b, threshold at 9 percentile
     levels (10..90) in both polarities, open with 7x7, take connected components.
  2. Candidate filter: area 1.2%-60% of image, solidity >= 0.92, ellipse-fit IoU >= 0.88.
  3. Rank by ellipse score + 0.5 * area fraction, suppress overlaps, keep max 3 eggs.
  4. Fallback (only if nothing passed): 17 levels, area >= 2%, ellipse >= 0.80,
     solidity >= 0.85, keep the single best candidate.
  5. Refine each region with GrabCut, complete partial eggs with a fitted ellipse,
     fill holes smaller than 0.5% of the image.
"""
import json
import os

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------- parameters
P4 = dict(n_levels=9, open_k=7, min_frac=0.012, max_frac=0.6, min_ell=0.88,
          min_solidity=0.92, area_bonus=0.5, max_eggs=3)
FALLBACK = dict(n_levels=17, open_k=7, min_frac=0.02, max_frac=0.6, min_ell=0.80,
                min_solidity=0.85, area_bonus=0.5, max_eggs=1)


# ------------------------------------------------------------------ metrics
def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 1.0


def load_gt(name, gt_dir="outputs/task1/masks"):
    """Task 1 instance map -> list of boolean masks, one per egg."""
    inst = np.array(Image.open(os.path.join(gt_dir, f"{name}.png")))
    return [inst == i for i in np.unique(inst) if i != 0]


def score_image(pred_masks, gt_masks, match_thr=0.5):
    H, W = gt_masks[0].shape
    P = np.zeros((H, W), bool)
    G = np.zeros((H, W), bool)
    for m in pred_masks:
        P |= m
    for m in gt_masks:
        G |= m
    tp, fp, fn = (P & G).sum(), (P & ~G).sum(), (~P & G).sum()
    out = {"iou": iou(P, G), "dice": dice(P, G),
           "precision": tp / (tp + fp) if tp + fp else 0.0,
           "recall": tp / (tp + fn) if tp + fn else 0.0,
           "n_gt": len(gt_masks), "n_pred": len(pred_masks)}
    matched = []
    if pred_masks and gt_masks:
        M = np.array([[iou(p, g) for g in gt_masks] for p in pred_masks])
        r, c = linear_sum_assignment(-M)
        matched = [M[i, j] for i, j in zip(r, c) if M[i, j] >= match_thr]
    out["tp_inst"] = len(matched)
    out["fp_inst"] = len(pred_masks) - len(matched)
    out["fn_inst"] = len(gt_masks) - len(matched)
    out["mean_matched_iou"] = float(np.mean(matched)) if matched else 0.0
    return out


def summarize(df):
    return {
        "mean_iou": round(float(df.iou.mean()), 4),
        "median_iou": round(float(df.iou.median()), 4),
        "mean_dice": round(float(df.dice.mean()), 4),
        "precision": round(float(df.precision.mean()), 4),
        "recall": round(float(df.recall.mean()), 4),
        "inst_precision": round(float(df.tp_inst.sum() / max(df.n_pred.sum(), 1)), 4),
        "inst_recall": round(float(df.tp_inst.sum() / df.n_gt.sum()), 4),
        "count_mae": round(float((df.n_pred - df.n_gt).abs().mean()), 4),
    }


# ------------------------------------------------------------ segmentation
def seg_candidates(rgb, n_levels=9, open_k=7, min_frac=0.012, max_frac=0.6,
                   min_ell=0.88, min_solidity=0.92, area_bonus=0.5, max_eggs=3):
    """Multi-threshold candidates verified by solidity and ellipse fit (T4)."""
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    labc = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    chans = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), hsv[..., 1], labc[..., 2]]
    ok = np.ones((open_k, open_k), np.uint8)
    pad = 20
    cands = []
    for ch in chans:
        for t in np.percentile(ch, np.linspace(10, 90, n_levels)):
            for bw in (ch >= t, ch <= t):
                bw = cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_OPEN, ok)
                n, lab_, stats, _ = cv2.connectedComponentsWithStats(bw)
                for i in range(1, n):
                    x, y, w, h, a = stats[i]
                    if not (min_frac * H * W <= a <= max_frac * H * W):
                        continue
                    crop = np.pad((lab_[y:y + h, x:x + w] == i).astype(np.uint8), pad)
                    cs, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    c = max(cs, key=cv2.contourArea)
                    if len(c) < 5:
                        continue
                    filled = np.zeros_like(crop)
                    cv2.drawContours(filled, [c], -1, 1, -1)
                    area = int(filled.sum())
                    hull_area = cv2.contourArea(cv2.convexHull(c))
                    if hull_area == 0 or area / hull_area < min_solidity:
                        continue
                    ell = np.zeros_like(crop)
                    cv2.ellipse(ell, cv2.fitEllipse(c), 1, -1)
                    s = iou(filled.astype(bool), ell.astype(bool))
                    if s < min_ell:
                        continue
                    cands.append((s + area_bonus * area / (H * W), (y - pad, x - pad), filled))
    cands.sort(key=lambda c: -c[0])
    keep = []
    for _, (y0, x0), filled in cands:
        full = np.zeros((H, W), bool)
        ys, xs = np.nonzero(filled)
        ys, xs = ys + y0, xs + x0
        v = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        full[ys[v], xs[v]] = True
        if all((full & k).sum() / max(min(full.sum(), k.sum()), 1) < 0.5 for k in keep):
            keep.append(full)
        if len(keep) == max_eggs:
            break
    return keep


def refine_grabcut(rgb, mask, iters=3, erode=15, dilate=25):
    cv2.setRNGSeed(0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = mask.astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    sure_fg = cv2.erode(m, k(erode))
    maybe = cv2.dilate(m, k(dilate))
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    if sure_fg.sum() == 0:
        return mask
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8))
    if n <= 1:
        return mask
    return lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def complete_ellipse(mask, min_cover=0.6, min_inside=0.9):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return mask
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return mask
    ell = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(ell, cv2.fitEllipse(c), 1, -1)
    ell = ell.astype(bool)
    a_m, a_e = mask.sum(), ell.sum()
    if a_e == 0 or a_m == 0:
        return mask
    if a_m / a_e >= min_cover and (mask & ell).sum() / a_m >= min_inside:
        return mask | ell
    return mask


def fill_small_holes(mask, max_frac=0.005):
    m = mask.astype(np.uint8)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return mask
    out = m.copy()
    for c, h in zip(cs, hier[0]):
        if h[3] != -1 and cv2.contourArea(c) < max_frac * m.size:
            cv2.drawContours(out, [c], -1, 1, -1)
    return out.astype(bool)


def segment_eggs(rgb):
    """Final algorithm (T9b). rgb: HxWx3 uint8 -> list of boolean egg masks."""
    out = seg_candidates(rgb, **P4)
    if not out:
        out = seg_candidates(rgb, **FALLBACK)
    return [fill_small_holes(complete_ellipse(refine_grabcut(rgb, m))) for m in out]


# --------------------------------------------------------------------- I/O
def image_paths(data_dir):
    paths = {}
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                paths[f.split("_jpg")[0]] = os.path.join(root, f)
    return paths


def load_img(path):
    return np.array(Image.open(path).convert("RGB"))


def save_instance_map(masks, shape, path):
    inst = np.zeros(shape, np.uint8)
    for i, m in enumerate(masks, start=1):
        inst[m & (inst == 0)] = i
    Image.fromarray(inst).save(path)


def load_split(path="outputs/split.json"):
    s = json.load(open(path))
    return s["dev"], s["eval"]