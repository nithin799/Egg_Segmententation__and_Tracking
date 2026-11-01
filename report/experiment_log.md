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
| T3 | Canny + closing (5x5) + ellipse-fit contour filter | No channel separates egg from background (best AUC 0.85); eggs are the one consistent smooth-ellipse shape | sigma=0.33, close_k=5, min_frac=0.01, max_frac=0.5, min_ell=0.85 | 0.109 / 0.121 | 0.146 | Discarded as a standalone method: median IoU 0.0, most images get no prediction (pixel recall 0.11). Instance precision rose to 0.80, so the ellipse test is a good verifier. Likely cause: Canny edges around pale eggs are not closed. Idea kept: use the ellipse test to filter candidates from another method. |

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