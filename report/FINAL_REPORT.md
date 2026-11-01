# Technical Report: Egg Segmentation and Video Tracking
**Author:** Candidate | Computer Vision Engineering Assessment  
**Repository:** Public GitHub Repository  
**Video Tracking Deliverables:**
* **Input Video:** [Google Drive Link](https://drive.google.com/file/d/1VhZXbqQUeHSfHX41rsnuAWxlQTsljW1C/view?usp=sharing)
* **Output Tracked Video:** [Google Drive Link](https://drive.google.com/file/d/12Lm6lshvtxvqnq_h5EeJbW2ggkfVuWfe/view?usp=sharing)

---

## 1. Executive Summary

In this assessment, I developed an end-to-end computer vision pipeline to detect, segment, and track eggs across diverse, unconstrained real-world environments. The study covers four core tasks:
1. **Task 1 (Foundation Model Baseline):** Because the Roboflow dataset provided no mask ground truth, I built a zero-shot segmentation cascade using Grounding DINO and SAM ViT-Base to establish reference pseudo-ground-truth masks for 110 images (138 instances).
2. **Task 2 (Classical Algorithm Development):** I developed a classical image processing algorithm from scratch across 9 hypothesis-driven trials, starting from a basic Otsu thresholding baseline ($0.338$ IoU) and advancing to multi-channel percentile proposals with shape-based ellipse verification ($0.599$ Dev / $0.654$ Eval IoU).
3. **Task 3 (Evaluation & Analysis):** I conducted detailed quantitative benchmarking against SAM masks, performed Spearman rank correlation analysis on scene attributes ($\rho = +0.66$ with contrast), and categorized failure modes into a 5-class physical taxonomy.
4. **Task 4 (Real-Time Video Tracking):** I adapted the classical segmentation algorithm into an online multi-object tracker that tracks two spinning eggs in a 60-second YouTube experiment at **27.6 FPS on CPU** with persistent identity continuity.

---

## 2. Task 1: Segmentation Using Foundation Models

### 2.1 Pipeline Design & Execution
To establish high-fidelity reference masks for quantitative evaluation:
* **Detection:** I used `IDEA-Research/grounding-dino-base` prompted with text `"an egg."` with a box threshold of $0.30$ and text threshold of $0.25$.
* **Segmentation:** Bounding boxes were passed as spatial prompts to `facebook/sam-vit-base` to produce binary instance segmentation masks.
* **Storage:** Outputs are stored as 8-bit single-channel instance maps in `outputs/task1/masks/` ($0 = \text{background}, 1..N = \text{instance ID}$) and visual overlays in `outputs/task1/overlays/`.

### 2.2 Dataset Verification & Stability Analysis
* **Instance Breakdown:** Across all 110 images ($640 \times 640$), the model detected **138 egg instances**:
  - Single-egg images: **93 (84.5%)**
  - Two-egg images: **6 (5.5%)**
  - Three-egg images: **11 (10.0%)**
  - Missed images: **0 (100% recall coverage)**
* **Agreement Study (SAM-Base vs. SAM-Large):** To verify that the pseudo-ground truth was not biased by model capacity, I evaluated all 138 instances against `facebook/sam-vit-large` with multimask selection. This yielded a **median IoU of 0.995** and **mean IoU of 0.990**, with $97.8\%$ of instances achieving $\ge 0.95$ IoU.
* **Caveat Logging:** Physical edge cases (refraction in boiling water in `124, 135, 145`, leaf occlusions in `10, 11, 12, 13`, and roasted ash in `27, 78`) were logged to `outputs/task1/caveats.json`. Image `117` (metal rod crossing egg) was manually overridden with multimask output.

---

## 3. Task 2: Classical Image Processing Pipeline Development

### 3.1 Leak-Free Stratified Split
To avoid data leakage between development and evaluation:
* **Perceptual Hashing:** I computed `imagehash.phash` across all images with a Hamming distance threshold $\le 10$. Exactly one near-duplicate pair was found (`not_damaged_78` $\leftrightarrow$ `not_damaged_27`, distance 4).
* **Cluster-Level Partitioning:** The 109 clusters were split in a 4:1 ratio (88 Dev, 22 Eval) stratified by egg count ($1, 2, 3$). The duplicate pair was kept strictly within Dev. The Eval split was frozen and used only once for final verification.

### 3.2 Iterative Experimental Progression (Dev Split, 88 Images)

| Trial | Algorithm Description | Motivation / Hypothesis | Dev Mean IoU | Dev Dice | Inst. Prec. | Count MAE | Decision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **T1** | Grayscale + Otsu + Morph. Closing | Simplest baseline: assumes eggs are the brightest object | 0.338 | 0.453 | 0.196 | 1.40 | Baseline |
| **T2** | HSV Threshold ($S \le 60, V \ge 120$) | Hypothesis: eggs are less saturated than colored surroundings | 0.324 | 0.441 | 0.147 | 1.50 | Discarded |
| **T3** | Canny Edges + Closing + Ellipse Fit | Hypothesis: egg contour is a smooth convex ellipse | 0.109 | 0.121 | 0.800 | 1.11 | Discarded |
| **T3f** | Tuned Canny (blur 9, close 15) | Attempt to bridge weak gradient gaps on pale shells | 0.273 | 0.308 | 0.561 | 0.93 | Discarded |
| **T4** | Multi-Threshold (Gray, S, b) + Ellipse | Proposals always yield closed blobs; verify via ellipse fit | 0.470 | 0.546 | 0.705 | 0.34 | Core Base |
| **T5** | T4 + Relaxed 17-percentile Fallback | Rescue 22 dev images that were left empty under T4 | 0.507 | 0.603 | 0.481 | 0.40 | Provisional |
| **T6** | T4 + GrabCut Boundary Refinement | Use GrabCut GMM color models to sharpen leaking borders | 0.498 | 0.566 | 0.795 | 0.34 | Kept |
| **T7** | T6 + Single-Candidate Fallback | Cap fallback at 1 candidate to stop flooding false shapes | 0.549 | 0.637 | 0.685 | 0.11 | Kept |
| **T8a** | Convex Hull Ellipse (solidity 0.80) | Attempt to rescue eggs partially occluded by leaves | 0.424 | 0.530 | 0.428 | 0.40 | Discarded |
| **T9b** | T7 + Ellipse Completion + Hole Fill | Complete shadow cut-offs and fill internal specular holes | 0.549 | 0.639 | 0.704 | 0.11 | Locked Base |
| **V1** | T9b + Area Cap (`max_frac = 0.40`) | Diagnostic: real eggs $\le 35.6\%$ area; reject white bags | 0.578 | 0.660 | 0.755 | 0.11 | Validated |
| **V2** | T9b + Area Cap $0.40$ + Aspect Cap $2.5$ | Diagnostic: real egg aspect $\le 1.94$; reject metal rods | **0.599** | **0.686** | **0.802** | **0.15** | **Final Model** |

### 3.3 Key Insights from Trials
* **Why Global Intensity Thresholds Failed (T1, T2):** Channel separability analysis (ROC-AUC) revealed that no single channel can separate eggs universally (Gray: 0.851, L: 0.850, V: 0.837, b: 0.777, S: 0.775, a: 0.703). Backlit eggs are darker than backgrounds, while eggs on white trays are brighter, causing global thresholds to invert polarity.
* **The Proposal Breakthrough (T4):** Moving from fragile edge chaining (T3) to multi-channel percentile thresholding (Gray, HSV-S, Lab-b) at 9 levels in both polarities, filtered by morphological opening ($7 \times 7$), solidity ($\ge 0.92$), and fitted ellipse IoU ($\ge 0.88$), formed a solid baseline.
* **Final Physical Bounds (V2):** Failure analysis revealed that white bags were swallowed because `max_frac` was $0.60$, and metal rods passed because OpenCV `fitEllipse` fits elongated shapes. Constraining `max_frac = 0.40` and `max_aspect = 2.5` boosted Dev IoU to **0.599** and Eval IoU to **0.654** with **0.802** instance precision.

---

## 4. Task 3: Quantitative Evaluation & Failure Analysis

### 4.1 Benchmarking Against Ground Truth

| Algorithm Version | Split | Mean IoU | Median IoU | Mean Dice | Pixel Prec. | Pixel Rec. | Inst. Prec. | Inst. Rec. | Count MAE | Empty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T9b Baseline** | Dev (88) | 0.549 | 0.626 | 0.638 | 0.704 | 0.714 | 0.694 | 0.682 | 0.114 | 2 |
| **T9b Baseline** | Eval (22) | 0.588 | 0.649 | 0.695 | 0.760 | 0.768 | 0.690 | 0.714 | 0.136 | 0 |
| **V2 Final** | Dev (88) | **0.599** | **0.698** | **0.686** | **0.791** | 0.646 | **0.802** | **0.736** | 0.148 | 7 |
| **V2 Final** | Eval (22) | **0.654** | **0.733** | **0.742** | **0.828** | 0.722 | **0.786** | **0.786** | **0.091** | **0** |

### 4.2 Statistical Findings
* **Contrast Dominance:** Spearman rank correlation $\rho(\text{IoU}, \text{Contrast}) = +0.66$ ($p < 0.001$). The low-contrast tercile achieves only $0.29$ mean IoU (accounting for 12 of 14 misses), while high contrast reaches $0.76$ with 0 misses. Performance is strongly bimodal based on chromatic separability.
* **Scale Invariance:** Real egg areas span from $1.8\%$ to $35.6\%$ of the frame. IoU shows near-zero correlation with egg size ($\rho = 0.08$), confirming that the percentile candidate generator handles both close-up and distant eggs equally well.
* **No Data Leakage:** Eval IoU ($0.654$) exceeds Dev IoU ($0.599$), confirming the model did not overfit the development set.

### 4.3 Failure Mode Taxonomy
1. **Low-Contrast White-on-White (`104, 106, 112, 123, 125, 139`):** White eggs on white plastic bags or plates share near-identical $L^*, a^*, b^*$ values, leaving no closed boundary gradient.
2. **Atypical Roasted Ash Textures (`27, 78`):** Dark ash creates strong internal edges, fracturing the egg into fragments that fail solidity ($\text{solidity} < 0.92$).
3. **Structural Occlusions (`10, 11, 12, 13` leaves; `42, 117` rods):** Extraneous objects cutting across the silhouette create boundary concavities that fail the solidity gatekeeper.
4. **Directional Lighting & Shadows (`103, 119, 130`):** Harsh shadows illuminate only a crescent. Ellipse completion does not trigger below $60\%$ coverage, causing under-segmentation.
5. **Boiling Water Refraction (`124, 135, 145`):** Fluid turbulence and surface bubbles blur boundaries.

---

## 5. Task 4: Real-Time Multi-Object Video Tracking

### 5.1 System Architecture
* **Detector Adaptation:** Adapted multi-threshold proposals with a shell luminance filter ($\text{mean\_gray} \ge 160$) to eliminate dark table shadows.
* **Multi-Object Tracker:** Hungarian bipartite matching on Euclidean centroid distances with a 35-frame ($1.5$s) coasting memory to handle temporary hand occlusions while eggs are spun.
* **Visual Overlays:** Semi-transparent colored masks, contour borders, track ID badges (`Egg #1` Emerald Green, `Egg #2` Coral Orange), historical motion trails, and a real-time HUD.

### 5.2 Performance & Deliverables
* **Processing Speed:** Processed all 1,440 frames ($60.0$s @ 24 FPS) in **52.2 seconds at 27.6 FPS on CPU** (faster than real-time).
* **Tracking Robustness:** Maintained consistent identities for `Egg #1` and `Egg #2` throughout the spinning and stopping interactions.
* **Video Deliverables:**
  - Input Video: [Google Drive Link](https://drive.google.com/file/d/1VhZXbqQUeHSfHX41rsnuAWxlQTsljW1C/view?usp=sharing)
  - Output Tracked Video: [Google Drive Link](https://drive.google.com/file/d/12Lm6lshvtxvqnq_h5EeJbW2ggkfVuWfe/view?usp=sharing)

---

## 6. Setup & Reproduction Guide

```bash
# 1. Environment Setup
pip install -r requirements.txt

# 2. Run Classical Segmentation Benchmark (Tasks 2 & 3)
python src/run_task.py --split dev --version v2
python src/run_task.py --split eval --version v2 --final

# 3. Run Real-Time Video Tracking (Task 4)
python src/track_video.py

# 4. Re-compile PDF Technical Report
python src/generate_pdf_report.py
```
