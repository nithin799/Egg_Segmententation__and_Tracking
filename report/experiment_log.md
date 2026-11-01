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
| T2 | HSV: S <= 60 and V >= 120, then CC | T1 failed on brightness; hypothesis: eggs are less saturated than skin, plastic, fabric | s_max=60, v_min=120, min_area=2000 | 0.324 / 0.441 | 0.309 | Discarded. Worse than T1 (inst. recall 0.39 to 0.31, count MAE 1.40 to 1.50); better on 47 images, worse on 41. White walls, bags and trays share the egg's low saturation and high value; backlit eggs (20, 44) fail V >= 120. Channel analysis showed S is a weaker cue (separability 0.775) than gray (0.851), so the hypothesis was wrong. |
### Channel separability (dev, per-image AUC, polarity chosen per image, so optimistic)
gray 0.851 | L 0.850 | V 0.837 | b 0.777 | S 0.775 | a 0.703. Images with AUC > 0.9: gray 34, L 34, V 31, b 19, S 19, a 5 (of 88).
Conclusion: no single channel separates egg from background on most images; the egg is brighter than its background in some images and darker in others.
| T3 | Canny + closing (5x5) + ellipse-fit contour filter | No channel separates egg from background (best AUC 0.85); eggs are the one consistent smooth-ellipse shape | sigma=0.33, close_k=5, min_frac=0.01, max_frac=0.5, min_ell=0.85 | 0.109 / 0.121 | 0.146 | Discarded as a standalone method: median IoU 0.0, most images get no prediction (pixel recall 0.11). Instance precision rose to 0.80, so the ellipse test is a good verifier. Diagnosed with edge maps: on a pale egg on marble (100) Canny finds only two broken arcs; on an egg in a hand (1) the outline is almost closed but merges with finger and palm edges into one non-elliptical contour. Idea kept: generate candidates from a method that always returns closed regions, and use the ellipse test to filter them. |
| T3b | T3 with lower Canny thresholds | Pale eggs have weak gradients (image 100) | blur 5, Canny 0.33/0.8 x median, close 5, min_ell 0.85 | 0.206 / 0.224 | 0.255 | Better than T3, but 65 of 88 images get no prediction. Inst. precision 0.93. |
| T3c | T3b + blur 9, close 9 | Reduce texture edges, bridge gaps | blur 9, Canny 0.33/0.8, close 9, min_ell 0.85 | 0.170 / 0.188 | 0.200 | Discarded. Worse than T3b: more smoothing weakened the egg edges too. |
| T3d | T3c with min_ell 0.75 | Accept less perfect ellipses | blur 9, Canny 0.33/0.8, close 9, min_ell 0.75 | 0.175 / 0.201 | 0.209 | Discarded. Inst. precision fell 0.96 to 0.58 for almost no gain in recall. |
| T3e | blur 5, close 9, min_ell 0.75 | Keep sharp edges, bridge gaps | blur 5, Canny 0.33/0.8, close 9, min_ell 0.75 | 0.238 / 0.268 | 0.300 | Better; 50 images with no prediction. |
| T3f | T3e with blur 9, Canny 0.2/0.6, close 15 | Weakest edges plus the largest gap bridging | blur 9, Canny 0.2/0.6, close 15, min_ell 0.75 | 0.273 / 0.308 | 0.336 | Best of the edge family, but still below T1 (0.338) and empty on 48 of 88 images. Edge-based candidates are the bottleneck, so change approach. |
| T4 | Multi-threshold candidates (gray, S, b; both polarities; 9 percentile levels) + opening 7x7 + solidity and ellipse-fit filter, best-ranked non-overlapping, max 3 eggs | T3 edges gave no closed regions; thresholds always give closed regions, and the ellipse test verified detections well (T3 inst. precision 0.80) | n_levels=9, open_k=7, min_ell=0.88, min_solidity=0.92, area_bonus=0.5, max_eggs=3 | 0.470 / 0.546 | 0.564 | Kept as the base. Best at that point (T1 0.338). Median IoU 0.539, count MAE 0.34, 20 s for 88 images. 22 of 88 images get no prediction: white egg on white bag/wall (104, 106, 109, 111, 112, 123) and eggs under leaves (11, 12). Ranking bonus and max_eggs=3 are heuristics. |
| T5 | T4, then relaxed fallback only if T4 returns nothing | 22 of 88 dev images were empty under T4 | fallback: n_levels=17, min_ell=0.80, min_solidity=0.85, min_frac=0.01 | 0.507 / 0.603 | 0.582 | Kept (best mean IoU), with a caveat. Empty images 22 to 2, pixel recall 0.54 to 0.68. But inst. precision fell 0.70 to 0.48 and count MAE rose 0.34 to 0.40, so the relaxed filters also accept wrong shapes. Fallback outputs not yet inspected. |Kept provisionally. Empty images 22 to 2, but of the 22 rescued images only 1 reaches IoU > 0.5 and 9 are below 0.1. Inspection: the relaxed filters accept background shapes (bag folds, the hole in the bag, a red rim, wall edges) in 112, 123, 125, 139, where the real egg is missed. Mean IoU gain comes from partial overlaps. Inst. precision fell 0.70 to 0.48. Candidate for removal.|Kept provisionally. Empty images 22 to 2, but of the 22 rescued images only 1 reaches IoU > 0.5 and 9 are below 0.1. Inspection: the relaxed filters accept background shapes (bag folds, the hole in the bag, a red rim, wall edges) in 112, 123, 125, 139, where the real egg is missed. Mean IoU gain comes from partial overlaps. Inst. precision fell 0.70 to 0.48. Candidate for removal.|
| T6 | T4 + GrabCut refinement of each accepted region | Found eggs have partial or leaking boundaries (52, 41, 12) | iters=3, erode=15, dilate=25, base = T4 params | 0.498 / 0.566 | 0.636 | Kept. Median IoU 0.570, inst. precision 0.70 to 0.80, inst. recall 0.56 to 0.64, count MAE unchanged (0.34). Does not fix the 22 images where no egg is found. |
| T7 | T6 + strict fallback: if T4 is empty, retry with relaxed filters, keep only the single best candidate (min_frac 0.02), then GrabCut | T5's fallback flooded false eggs; cap it at one candidate per empty image | fallback: n_levels=17, min_frac=0.02, min_ell=0.80, min_solidity=0.85, max_eggs=1 | 0.549 / 0.637 | 0.673 | Kept (best so far). Mean IoU 0.498 to 0.549, median 0.570 to 0.624, count MAE 0.34 to 0.11, empty images 22 to 2. Inst. precision fell 0.80 to 0.69. Quality of the rescued images still to be inspected. |Kept (best so far). Mean IoU 0.498 to 0.549, median 0.570 to 0.624, count MAE 0.34 to 0.11, empty images 22 to 2. Inst. precision fell 0.80 to 0.69. Rescue check on the 22 images T6 left empty: 4 reach IoU > 0.5, 10 are below 0.1, 8 are partial. Failures: white egg on white bag/wall (112, 123, 125, 139) gets a background blob or nothing; partial egg coverage (41, 12, 39) where a threshold isolates only part of the egg. Remaining misses are the white-on-white family.|



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