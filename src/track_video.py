"""Task 4: Video Tracking of Individual Eggs using Classical Image Processing.

Pipeline:
  1. Classical Segmentation: Multi-threshold proposal generation (Gray, Lab-b) with
     morphological opening, solidity gating, and fitted-ellipse verification (adapted from V2).
  2. Dark shadow rejection: filters out table shadow artifacts based on shell luminance.
  3. Real-Time Multi-Object Tracker: Hungarian distance association with track continuity,
     occlusion coasting, and trajectory history.
  4. Video Rendering: Overlays colored semi-transparent instance masks, bounding boxes,
     unique track ID badges, motion trajectory tails, and an information HUD banner.

Usage:
  python src/track_video.py
"""
import argparse
from collections import deque
import os
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class EggTracker:
    """Maintains consistent identity and motion state for multiple eggs across frames."""

    def __init__(self, max_disappeared=35, max_distance=160):
        self.next_id = 1
        self.tracks = {}  # id -> dict(centroid, bbox, mask, history, disappeared, color)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.palette = [
            (0, 255, 128),  # Vibrant Emerald / Neon Green (Egg #1)
            (0, 165, 255),  # Vibrant Coral / Orange (Egg #2)
            (255, 215, 0),  # Gold / Amber
            (255, 105, 180), # Hot Pink
        ]

    def update(self, detections):
        """Update tracker with candidate detections: list of (score, bbox, mask)."""
        det_centroids = []
        for _, (x, y, w, h), _ in detections:
            det_centroids.append((x + w // 2, y + h // 2))

        # 1. No active tracks yet
        if len(self.tracks) == 0:
            for i, (score, bbox, mask) in enumerate(detections):
                self._add_track(det_centroids[i], bbox, mask)
            return self.tracks

        # 2. No detections in current frame
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["mask"] = None
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return self.tracks

        # 3. Associate existing tracks with detections via Hungarian algorithm
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]["centroid"] for tid in track_ids]

        cost_matrix = np.linalg.norm(
            np.array(track_centroids)[:, None] - np.array(det_centroids)[None, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        used_rows = set()
        used_cols = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > self.max_distance:
                continue
            tid = track_ids[r]
            score, bbox, mask = detections[c]
            cx, cy = det_centroids[c]
            self.tracks[tid]["centroid"] = (cx, cy)
            self.tracks[tid]["bbox"] = bbox
            self.tracks[tid]["mask"] = mask
            self.tracks[tid]["disappeared"] = 0
            self.tracks[tid]["history"].append((cx, cy))
            used_rows.add(r)
            used_cols.add(c)

        # Increment disappeared count for unmatched tracks
        for r, tid in enumerate(track_ids):
            if r not in used_rows:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["mask"] = None
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Register new track only if fewer than 2 active tracks exist
        for c in range(len(detections)):
            if c not in used_cols and len(self.tracks) < 2:
                self._add_track(det_centroids[c], detections[c][1], detections[c][2])

        return self.tracks

    def _add_track(self, centroid, bbox, mask):
        tid = self.next_id
        self.next_id += 1
        color = self.palette[(tid - 1) % len(self.palette)]
        history = deque(maxlen=45)
        history.append(centroid)
        self.tracks[tid] = {
            "centroid": centroid,
            "bbox": bbox,
            "mask": mask,
            "history": history,
            "disappeared": 0,
            "color": color,
        }


def detect_eggs_in_frame(img_bgr, min_lum=160):
    """Segment bright egg candidates in video frames using classical morphological techniques."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    chans = [gray, lab[..., 2]]
    opening_kernel = np.ones((7, 7), np.uint8)
    pad = 10
    candidates = []

    for ch in chans:
        for threshold_val in np.percentile(ch, np.linspace(30, 90, 7)):
            bw = ch >= threshold_val
            bw = cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_OPEN, opening_kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)

            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if not (0.008 * H * W <= area <= 0.25 * H * W):
                    continue
                if max(w, h) / max(min(w, h), 1) > 2.2:
                    continue

                crop = np.pad((labels[y : y + h, x : x + w] == i).astype(np.uint8), pad)
                contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                if len(c) < 5:
                    continue

                filled = np.zeros_like(crop)
                cv2.drawContours(filled, [c], -1, 1, -1)
                comp_area = int(filled.sum())
                hull_area = cv2.contourArea(cv2.convexHull(c))
                if hull_area == 0 or comp_area / hull_area < 0.88:
                    continue

                # Ellipse fit IoU
                ellipse_mask = np.zeros_like(crop)
                cv2.ellipse(ellipse_mask, cv2.fitEllipse(c), 1, -1)
                union_sum = (filled.astype(bool) | ellipse_mask.astype(bool)).sum()
                ell_iou = (filled.astype(bool) & ellipse_mask.astype(bool)).sum() / union_sum if union_sum else 0
                if ell_iou < 0.85:
                    continue

                full_mask = np.zeros((H, W), bool)
                ys, xs = np.nonzero(filled)
                ys, xs = ys + (y - pad), xs + (x - pad)
                valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                full_mask[ys[valid], xs[valid]] = True

                # Exclude dark table shadow artifacts
                if gray[full_mask].mean() < min_lum:
                    continue

                score = ell_iou + 0.5 * comp_area / (H * W)
                candidates.append((score, (x, y, w, h), full_mask))

    # Non-maximum suppression
    candidates.sort(key=lambda c: -c[0])
    keep = []
    for score, bbox, mask in candidates:
        if all((mask & k[2]).sum() / max(min(mask.sum(), k[2].sum()), 1) < 0.30 for k in keep):
            keep.append((score, bbox, mask))
        if len(keep) == 2:
            break

    return keep


def render_hud(frame, tracks, frame_idx, total_frames, fps, processing_fps):
    """Render professional overlay HUD banner."""
    H, W = frame.shape[:2]
    # Top banner bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 28), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    active_cnt = len([t for t in tracks.values() if t["disappeared"] == 0])
    text = (f"Classical Egg Tracker | Active Eggs: {active_cnt}/2 | "
            f"Frame: {frame_idx:4d}/{total_frames} ({frame_idx/fps:4.1f}s) | Speed: {processing_fps:.1f} FPS")
    cv2.putText(frame, text, (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    tracker = EggTracker(max_disappeared=35, max_distance=160)
    print(f"Tracking eggs in: {video_path}")
    print(f"Resolution: {w}x{h} | FPS: {fps} | Total frames: {total_frames}")

    t_start = time.time()
    preview_frames = {}

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        f_t0 = time.time()

        # Detection & Tracking
        detections = detect_eggs_in_frame(frame)
        tracks = tracker.update(detections)

        annotated = frame.copy()

        # 1. Semi-transparent segmentation mask
        for tid, tdata in tracks.items():
            mask = tdata["mask"]
            col = tdata["color"]
            if mask is not None:
                annotated[mask] = (0.50 * annotated[mask] + 0.50 * np.array(col)).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, col, 2)

        # 2. Motion Trajectory Trail
        for tid, tdata in tracks.items():
            history = list(tdata["history"])
            col = tdata["color"]
            for j in range(1, len(history)):
                thickness = int(np.clip(j / len(history) * 3, 1, 3))
                cv2.line(annotated, history[j - 1], history[j], col, thickness, cv2.LINE_AA)

        # 3. Bounding Boxes and Labels
        for tid, tdata in tracks.items():
            x, y, bw, bh = tdata["bbox"]
            col = tdata["color"]
            cx, cy = tdata["centroid"]

            if tdata["disappeared"] == 0:
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), col, 2)
                cv2.circle(annotated, (cx, cy), 4, col, -1)
                label = f"Egg #{tid}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x, max(0, y - lh - 8)), (x + lw + 6, y), col, -1)
                cv2.putText(annotated, label, (x + 3, max(lh + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                # Occluded or spinning coasting track
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), col, 1, cv2.LINE_AA)
                cv2.putText(annotated, f"Egg #{tid} [coasting]", (x, max(15, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

        # Processing speed computation
        f_time = max(time.time() - f_t0, 1e-4)
        processing_fps = 1.0 / f_time

        # 4. HUD
        annotated = render_hud(annotated, tracks, f_idx, total_frames, fps, processing_fps)
        writer.write(annotated)

        # Save snapshots at key intervals for the report
        if f_idx in [260, 480, 720, 940]:
            preview_frames[f_idx] = annotated.copy()

        if f_idx % 120 == 0 or f_idx == total_frames - 1:
            elapsed = time.time() - t_start
            print(f"[{f_idx:4d}/{total_frames}] {f_idx/fps:4.1f}s | "
                  f"Active tracks: {len([t for t in tracks.values() if t['disappeared']==0])} | "
                  f"Elapsed: {elapsed:.1f}s ({(f_idx+1)/elapsed:.1f} FPS)", flush=True)

    cap.release()
    writer.release()
    total_time = time.time() - t_start
    print(f"\nSuccessfully finished tracking video in {total_time:.1f}s ({total_frames / total_time:.1f} FPS)!")
    print(f"Output saved to: {output_path} ({os.path.getsize(output_path) / 1024**2:.2f} MB)")

    # Save 4-panel report summary figure
    if len(preview_frames) >= 4:
        os.makedirs("report/figs", exist_ok=True)
        keys = sorted(preview_frames.keys())[:4]
        top = np.hstack([preview_frames[keys[0]], preview_frames[keys[1]]])
        bottom = np.hstack([preview_frames[keys[2]], preview_frames[keys[3]]])
        grid = np.vstack([top, bottom])
        preview_img_path = "report/figs/task4_tracking_preview.png"
        cv2.imwrite(preview_img_path, grid)
        print(f"Saved 4-panel tracking preview to {preview_img_path}")


def main():
    parser = argparse.ArgumentParser(description="Task 4: Video Egg Tracking")
    parser.add_argument("--video", default="Raw and Boiled Egg Spinning - Cool Science Experiment _ Educational Videos by Mocomi.mp4",
                        help="Input video file path")
    parser.add_argument("--out", default="outputs/task4/tracked_egg_experiment.mp4",
                        help="Output tracked video path")
    args = parser.parse_args()
    process_video(args.video, args.out)


if __name__ == "__main__":
    main()

