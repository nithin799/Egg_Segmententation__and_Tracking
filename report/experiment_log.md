# Experiment Log: Task 2 (Classical Egg Segmentation)

## 1. Setup

**Ground truth.** The dataset has no masks, so the Task 1 pseudo-ground-truth is used.
- Pipeline: Grounding DINO (prompt "an egg.", box threshold 0.3) + SAM ViT-base, one mask per box.
- Coverage: 110 images, 138 egg instances (93 single-egg, 6 two-egg, 11 three-egg images).
- Convention: mask = visible egg pixels only (occluders such as leaves and rods are excluded).
- Stability: SAM-base vs. SAM-large median IoU 0.995; 3 of 138 eggs below 0.95 (117, 145, 135).
- All 110 overlays were reviewed manually. One manual override: image 117 (SAM-base, multimask output).
- Known caveats are in `outputs/task1/caveats.json` (boiling-water eggs, leaves, ash, 42, 44).

**Split.** Cluster-level, so near-duplicates never straddle dev and eval. It is stratified by egg count, seed 42, and saved in `outputs/split.json`.
- Near-duplicate detection: pHash, Hamming distance <= 10 (one true pair found: 78 and 27, kept together).
- Dev: 88 images {1 egg: 75, 2 eggs: 4, 3 eggs: 9}
- Eval: 22 images {1 egg: 18, 2 eggs: 2, 3 eggs: 2}. It is small, so multi-egg numbers on eval are indicative only.
- The eval split is frozen and is used once, for the final algorithm. `log_trial` asserts that no eval image appears in a dev trial.

**Metrics** (computed by `score_image`, all against the Task 1 masks):
- Pixel level (all eggs merged): IoU, Dice, precision, recall.
- Instance level: Hungarian matching on IoU, match threshold 0.5. Reported as instance precision, instance recall and count MAE.
- The scorer was sanity-checked: perfect = 1.0, empty = 0.0, 20 px shift = 0.775 IoU (3 of 3 matched), merged blob = 0 of 3 matched.

## 2. Trials (dev split, 88 images)

| Trial | What changed | Why | Parameters | Dev mean IoU / Dice | Inst. recall | Kept? and why |
|---|---|---|---|---|---|---|
| T1 | Gray + Otsu + CC | baseline | min_area=2000 | 0.338 / 0.453 | 0.391 | Baseline. Recall 0.87 but precision 0.37. Egg merges with bright backgrounds (white bag, tile, lid) into one blob (125, 108, 127, 123, 42); on backlit hands the egg is darker than the background so it is missed (20, 44). Brightness alone cannot separate egg from background. |

### T1 details
- Full dev metrics: median IoU 0.238, pixel precision 0.372, pixel recall 0.868, instance precision 0.196, instance recall 0.391, count MAE 1.40.
- Pattern: high recall, very low precision. A single global brightness threshold assumes the egg is the only bright object.
- Worst dev images: 20 (0.01), 44 (0.01), 125 (0.04), 108 (0.04), 42 (0.05), 117 (0.06), 127 (0.06), 123 (0.06).
- Failure types seen: egg merged with a bright background (white-on-white), and a darker egg on a bright background (backlit hands).
- Next idea: colour saturation (HSV or Lab) instead of brightness.

## 3. Planned trials (not yet run)

- T2: HSV saturation / value threshold
- Further candidates: Lab channel, blur or illumination correction, morphological cleanup, watershed for touching eggs, contour filtering by area and circularity or ellipse fit, GrabCut. Failed attempts will be logged too.

## 4. Final run on eval

Not yet done. It is run once, after the algorithm is locked.