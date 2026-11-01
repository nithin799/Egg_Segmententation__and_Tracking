# Automated Egg Segmentation and Real-Time Video Tracking

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Tracking Throughput](https://img.shields.io/badge/Video_Tracking-27.6_FPS-brightgreen.svg)](#task-4-real-time-video-tracking)
[![Evaluation Split](https://img.shields.io/badge/Eval_IoU-0.6537-blueviolet.svg)](#benchmark-performance-summary)

Comprehensive computer vision system for egg instance segmentation and real-time video tracking across diverse real-world scenes, combining deep foundation models (**Grounding DINO + SAM ViT-Base**) with an ultra-fast, classical computer vision pipeline (**V2 Multi-Channel Geometric GrabCut**).

Developed as part of the **Computer Vision Engineering Internship Assessment**.

---

## 📑 Core Deliverables & Deliverable Links

| Deliverable | Format | Access Link | Description |
| :--- | :---: | :--- | :--- |
| **Official PDF Report** | `PDF` | [`report/Egg_Segmentation_and_Tracking_Report.pdf`](report/Egg_Segmentation_and_Tracking_Report.pdf) | 5-page publication-quality technical report with full methodology, figures, and benchmark tables. |
| **Tracked Output Video** | `Cloud MP4` | [Google Drive Stream / Download](https://drive.google.com/file/d/12Lm6lshvtxvqnq_h5EeJbW2ggkfVuWfe/view?usp=sharing) | Full 60-second tracked experiment (1,440 frames @ 24 FPS) with dual-egg real-time trajectories and telemetry. |
| **Input Raw Video** | `Cloud MP4` | [Google Drive Link](https://drive.google.com/file/d/1VhZXbqQUeHSfHX41rsnuAWxlQTsljW1C/view?usp=sharing) | Original YouTube science experiment video (*Raw and Boiled Egg Spinning*). |
| **Local Output Video** | `MP4` | [`outputs/task4/tracked_egg_experiment.mp4`](outputs/task4/tracked_egg_experiment.mp4) | High-definition processed video generated on local hardware. |
| **Experiment Log** | `Markdown` | [`report/experiment_log.md`](report/experiment_log.md) | Granular hypothesis logs across all trials (T1 to T9b, parameter sweeps P1–P8, and V1–V2). |
| **Narrative Report** | `Markdown` | [`report/FINAL_REPORT.md`](report/FINAL_REPORT.md) | Exhaustive Markdown documentation of results, correlation studies, and failure taxonomy. |

---

## 📊 Benchmark Performance Summary

All algorithms evaluated against the **Task 1 SAM ViT-Base reference masks** across the 110-image dataset partitioned into a 4:1 stratified cluster split (88 Development / 22 Evaluation images).

| Method / Architecture | Split | Mean IoU | Median IoU | Mean Dice | Pixel Prec. | Pixel Recall | Inst. Prec. | Inst. Recall | Count MAE | Coverage | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Grounding DINO + SAM** | All (110) | Reference | Reference | Reference | Reference | Reference | 1.000 | 1.000 | 0.000 | 100% | ~1.2 s / img (GPU) |
| **Task 2: T1 Otsu Baseline** | Dev (88) | 0.338 | 0.238 | 0.453 | 0.372 | **0.868** | 0.372 | 0.391 | 1.398 | 100% | 4 ms / img |
| **Task 2: T9b Base Pipeline** | Dev (88) | 0.549 | 0.626 | 0.638 | 0.704 | 0.714 | 0.694 | 0.682 | **0.114** | 97.7% | 38 ms / img |
| **Task 2: T9b Base Pipeline** | Eval (22) | 0.588 | 0.649 | 0.695 | 0.760 | 0.768 | 0.690 | 0.714 | 0.136 | **100%** | 38 ms / img |
| **Task 2 & 3: V2 Final Algorithm** | Dev (88) | **0.599** | **0.698** | **0.686** | **0.791** | 0.646 | **0.802** | **0.736** | 0.148 | 92.0% | 22 ms / img |
| **Task 2 & 3: V2 Final Algorithm** | Eval (22) | **0.654** | **0.733** | **0.742** | **0.828** | 0.722 | **0.786** | **0.786** | **0.091** | **100%** | **22 ms / img** |
| **Task 4: Real-Time Tracker** | Video (1440)| — | — | — | — | — | **1.000** | **1.000** | **0.000** | **100%** | **27.6 FPS** |

> **Key Generalization Takeaway:** On the untouched 22-image evaluation split, V2 achieved **0.6537 Mean IoU** (Median **0.7331**) and **0.8278 Pixel Precision** with **zero empty detections** and a Count MAE of **0.0909**, confirming strong generalization without split memorization.

---

## 🔬 Task Summaries & Technical Approach

### Task 1: Foundation Model Pseudo-Ground Truth Pipeline
* **Zero-Shot Detection:** Open-vocabulary Grounding DINO prompted with `"an egg."` (box threshold 0.30) locates all egg candidates across complex real-world scenes.
* **Prompt-Guided Segmentation:** Segment Anything Model (SAM ViT-Base) converts predicted bounding boxes into sharp, pixel-accurate instance masks.
* **Dataset Scope:** 110 images comprising 138 egg instances (93 single-egg, 6 two-egg, 11 three-egg images).
* **Stability Verification:** Benchmarked against SAM ViT-Large across all 138 instances, yielding a median IoU of **0.995** (98% of masks virtually identical).
* **Code & Artifacts:** Implemented in [`notebooks/Task-1.ipynb`](notebooks/Task-1.ipynb); masks and overlays stored in [`outputs/task1/`](outputs/task1/).

### Task 2: Classical Computer Vision Algorithm Development
* **Design Philosophy:** Developed through systematic, hypothesis-driven iterations (T1–T9b, V1, and V2) using only classical operations without deep learning or learned parameters.
* **Architecture:**
  1. **Multi-Channel Candidate Generation:** Extracts candidates across Grayscale, CIELAB $b^*$, and HSV $S$ across 9 percentile levels in both normal and inverted polarities.
  2. **Geometric Invariant Filtering:** Discards non-egg regions using contour solidity ($\ge 0.92$), parametric ellipse fit score ($\ge 0.88$), aspect ratio clamp ($\le 2.5$), and area fraction bounds ($0.01 \le \text{area} \le 0.40$).
  3. **Parametric Ellipse Completion:** Reconstructs shadowed or partially clipped shell boundaries.
  4. **Targeted GrabCut Refinement:** Applies a tight bounding box with iterative Gaussian Mixture Models to snap boundaries to exact shell contours.
* **Code:** [`src/egg_seg.py`](src/egg_seg.py) and interactive exploration in [`code.ipynb`](code.ipynb).

### Task 3: Benchmarking, Statistical Analysis & Failure Taxonomy
* **Correlation Analysis:** Evaluated non-parametric Spearman rank correlation across all 110 images:
  * **Local Shell Contrast:** $\rho = +0.66$ ($p < 0.001$). Upper tercile achieves **0.76 Mean IoU** with 0 misses; lower tercile averages **0.29 Mean IoU**, explaining 12 of 14 severe failures.
  * **Egg Scale Invariance:** $\rho = -0.07$ ($p = 0.48$). Performance remains uniform across egg sizes ranging from $1.8\%$ to $35.6\%$ of frame area.
* **Five-Class Failure Taxonomy:**
  1. *White-on-White Blending* (e.g., `104`, `112`, `125`): $\Delta E < 5$ contrast in CIELAB space.
  2. *Atypical Shell Textures* (e.g., `27`, `78`): Roasted soot stains breaking contour solidity.
  3. *Structural Occlusions* (e.g., `11`, `12`, `117`): Foliage and skewers causing contour concavities.
  4. *Directional Cast Shadows* (e.g., `103`, `130`): Strong spotlights causing crescent under-segmentation.
  5. *Fluid Refraction in Boiling Water* (e.g., `124`, `145`): Water ripples and steam perturbing GrabCut color statistics.
* **Artifacts:** Summary tables in [`outputs/task3/`](outputs/task3/) and visual charts in [`report/figs/`](report/figs/).

### Task 4: Real-Time Multi-Object Video Tracking
* **Input Video:** 60-second YouTube science demonstration (*Raw and Boiled Egg Spinning*, 1,440 frames @ 24 FPS, $640 \times 360$).
* **Tracking Pipeline:**
  * Multi-channel detector initialized on candidate frames with spatial ROI constraints.
  * Constant-velocity state prediction with Euclidean distance bipartite Hungarian association.
  * Real-time smoothed centroid trajectory rendering and persistent ID assignment (`ID 1: Raw Egg [Cyan]`, `ID 2: Boiled Egg [Amber]`).
* **Performance:**
  * **Processing Time:** **52.2 seconds** for 1,440 frames on CPU.
  * **Throughput:** **27.6 FPS** (real-time, exceeding the video's 24 FPS native rate).
  * **Tracking Reliability:** Zero ID switches, zero tracklet drift during high-speed spinning.
* **Code & Script:** [`src/track_video.py`](src/track_video.py).

---

## 🚀 Setup & Execution Instructions

### 1. Environment Installation
Clone the repository and install required packages in a clean Python virtual environment:
```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/egg-segmentation-tracking.git
cd egg-segmentation-tracking

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Classical Segmentation Evaluation (Tasks 2 & 3)
Run the finalized V2 algorithm across the pre-defined development and evaluation splits:
```bash
# Run on Development Split (88 images)
python src/run_task.py --split dev --version v2

# Run on Frozen Evaluation Split (22 images)
python src/run_task.py --split eval --version v2 --final
```
Results, logs, and per-image metrics are saved to `outputs/task2/` and `outputs/task3/`.

### 3. Run Real-Time Video Tracker (Task 4)
Execute the multi-object tracker on the video demonstration:
```bash
python src/track_video.py
```
Outputs generated:
* Tracked video: `outputs/task4/tracked_egg_experiment.mp4`
* 4-panel trajectory montage: `report/figs/task4_tracking_preview.png`

### 4. Re-Generate the PDF Report
To re-compile the 5-page submission report directly from the source images and tables:
```bash
python src/generate_pdf_report.py
```
Output compiled at: `report/Egg_Segmentation_and_Tracking_Report.pdf`.

---

## 📁 Repository Structure

```
├── code.ipynb                        # Tasks 2 & 3 complete interactive notebook with trial logs
├── notebooks/
│   └── Task-1.ipynb                  # Task 1 Grounding DINO + SAM foundation model notebook
├── src/
│   ├── egg_seg.py                    # Classical segmentation algorithms (T9b, V1, V2)
│   ├── run_task.py                   # Benchmark CLI and evaluation harness
│   ├── track_video.py                # Real-time multi-object video tracker (Task 4)
│   └── generate_pdf_report.py        # Automated 5-page PDF report compiler
├── outputs/
│   ├── split.json                    # Perceptual hash leak-free 4:1 Dev/Eval split
│   ├── task1/                        # Pseudo-GT masks, visual overlays, caveats.json
│   ├── task2/                        # Per-image metric CSVs, summaries, trials.csv
│   ├── task3/                        # metrics_table.csv, per_image_analysis.csv
│   └── task4/                        # tracked_egg_experiment.mp4
├── report/
│   ├── Egg_Segmentation_and_Tracking_Report.pdf  # 5-page publication-quality technical report
│   ├── FINAL_REPORT.md               # Complete narrative technical report
│   ├── experiment_log.md             # Detailed engineering trial log (T1 to V2)
│   ├── figs/                         # Publication-grade figures (overview, failures, tracking)
│   └── pages_humanized/              # High-resolution rasterized report pages
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation and reproduction guide
```

---

## ⚖️ License & Attribution
* **Dataset:** Egg segmentation dataset (`egg.v1i.folder`).
* **Video Source:** *"Raw and Boiled Egg Spinning - Cool Science Experiment"* by Mocomi Kids ([YouTube](https://www.youtube.com/watch?v=3nrhsm_kM3M)).
* **Assessment Project:** Developed by Computer Vision Intern (2026).
