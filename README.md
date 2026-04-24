# Where Do Lane Detectors Fail?
### A Scenario-Based Failure Analysis of CLRNet on CULane

---

## Overview

Lane detection models are typically evaluated using a single aggregate F1 score, which obscures how performance varies across real-world driving conditions. This project conducts a scenario-stratified failure analysis of CLRNet — a state-of-the-art lane detection model — on the CULane benchmark, with the goal of understanding not just *where* the model fails, but *why*.

The analysis proceeds in three stages: baseline evaluation across all CULane scenarios, Grad-CAM interpretability analysis to identify what spatial regions drive failures, and a targeted test-time augmentation intervention to probe model robustness under adverse conditions.

---

## Baseline Results (F1 @ IoU 0.5)

| Scenario | F1 | Notes |
|----------|-----|-------|
| Normal | 0.933 | Strong baseline; well-lit, unoccluded lanes |
| Arrow | 0.899 | Road markings provide strong cues |
| Shadow | 0.792 | Partial degradation from contrast variation |
| Crowd | 0.786 | Dense traffic occludes lane boundaries |
| Night | 0.749 | Low ambient light reduces feature quality |
| Highlight | 0.740 | Glare causes feature saturation |
| Curve | 0.697 | Geometric extrapolation required beyond anchor coverage |
| No Line | 0.522 | Near-absent lane markings; model lacks signal |
| Crossroad | 0.000 | Complete failure; intersections have no lane structure |
| **Overall** | **0.796** | |

The performance gap between Normal (0.933) and Crossroad (0.000) spans nearly the full range of possible F1 scores, making this an analytically rich benchmark for failure characterization.

---

## Methods

### Model

CLRNet with a ResNet-18 backbone, evaluated using official pretrained weights. The CLRNet codebase was patched for PyTorch 2.x compatibility: deprecated `x.type().is_cuda()` calls in the NMS CUDA extension were replaced with `x.is_cuda()`, and deprecated `np.bool` usage was replaced with the built-in `bool` type. No changes were made to model weights, architecture, or evaluation logic.

### Dataset

CULane is a large-scale benchmark comprising 133,235 images stratified across 9 driving scenarios. Each scenario is designed to isolate a distinct source of difficulty — lighting variation (night, highlight), occlusion (crowd), road geometry (curve, crossroad), and marking degradation (no line). This stratification makes CULane well suited for scenario-level failure analysis, as it directly exposes conditions under which performance degrades rather than averaging failures across the full test set.

### Evaluation Protocol

The official CULane evaluation metric was used throughout. Lane predictions are matched to ground truth using pixel-level IoU computed on 30-pixel-wide lane masks. A predicted lane is a true positive if its IoU with a ground truth lane exceeds 0.5. Precision, recall, and F1 are computed per scenario from TP/FP/FN counts.

### Failure Case Identification

Failure cases were identified at the image level from prediction output files generated during the official evaluation run. An image was classified as a failure if its prediction file was empty — indicating zero detected lanes. A parallel set of success cases (non-empty prediction files) was collected for comparison. Both sets were stratified by scenario to construct the failure and success atlases used in the Grad-CAM analysis.

### Gradient-Based Visualization

Grad-CAM was applied to analyze model attention, computing a weighted sum of feature map activations using gradients flowing back from the target output with respect to the final convolutional layer (`layer4[-1]` of the ResNet-18 backbone). A custom target function extracted the maximum lane confidence score across the 192 prior anchors from the classification logits, ensuring gradients correspond to lane detection outputs rather than auxiliary heads.

### Test-Time Augmentation

To probe model robustness under the hardest scenarios, a test-time augmentation pipeline was applied during re-evaluation: CLAHE (Contrast Limited Adaptive Histogram Equalization) for low-light and shadow conditions, gamma correction to simulate illumination variation, horizontal flip for geometric robustness, and unsharp masking to recover edge structure in degraded images. Augmented outputs were compared against baseline F1 to measure recovery.

---

## Key Findings

**Attention shifts to irrelevant regions in failure cases.** In successful predictions, Grad-CAM attention concentrates along road surface regions where lane markings are present. In failure cases, attention systematically shifts to irrelevant structures: tree canopies in shadow scenarios, intersection geometry in crossroads, and low-contrast sky regions in no-line conditions. This indicates that failures arise from misdirected attention rather than from insufficient model capacity.

**Crossroad failure is structural, not perceptual.** CLRNet achieves F1 = 0.000 on crossroads — a complete failure — despite the images being visually clear. The failure is not caused by poor lighting or occlusion but by the absence of lane structure at intersections. CLRNet's anchor-based prior assumes continuous lane lines; at intersections this assumption breaks entirely and the model produces zero detections. Grad-CAM confirms that attention disperses across intersection geometry rather than converging on any road feature.

**No Line performance reveals the limits of appearance-based detection.** F1 = 0.522 on the no-line scenario indicates that the model retains partial ability to infer lane position from road context (lane boundaries, adjacent vehicles, road edges) even without explicit marking cues. However, attention maps show high variance across failure cases in this scenario, suggesting inconsistent reliance on contextual versus marking-based features.

**Shadow boundary is the primary failure mechanism in shadow scenarios.** In shadow scenes, attention concentrates on the lit-to-shadow transition boundary rather than on the lane marking itself. This is plausible — the boundary is a high-gradient edge — but leads to detection errors when lanes cross through shadow regions without coinciding with lighting transitions.

**CLAHE partially recovers performance on low-contrast scenarios.** Applying CLAHE at test time improved detection rates on no-line and night cases by enhancing local contrast in regions where lane markings are faint. The improvement was inconsistent across images, suggesting the augmentation helps when the failure mode is contrast-related but not when it is structural (e.g., crossroad).

---

## Implementation

All experiments were run on Google Colab with a single NVIDIA A100 GPU. The CULane dataset was loaded via KaggleHub. Grad-CAM was implemented using the `pytorch-grad-cam` library with a custom target class for the CLRNet classification head.

---

## Repository Structure

```
CLRNet/                  — patched CLRNet source (PyTorch 2.x compatible)
configs/clrnet/          — model config with dataset paths pre-set
checkpoints/             — model weights (downloaded at runtime)
outputs/                 — failure atlas and success atlas visualizations
nn_project.ipynb         — main analysis notebook
```
