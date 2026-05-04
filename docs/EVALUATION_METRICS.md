# Evaluation metrics

Metrics are computed **inside the training notebooks** using a combination of **Keras built-ins**, **custom Keras metrics**, and **scikit-learn** utilities for offline summaries.

## Where they are implemented

| Kind | Typical location in notebooks |
|------|-------------------------------|
| Keras `compile` metrics | Same cell as `model.compile(...)` |
| Custom `dice_coefficient`, `f1_score`, etc. | Code cells above `compile` |
| Post-training / fold summaries | Cells exporting CSV under `Project_Outputs/` |

Search for: `dice_coefficient`, `MeanIoU`, `IoU`, `precision`, `recall`, `f1_score`, `model.evaluate`.

**TODO:** confirm that Keras `IoU` / `MeanIoU` class indexing (`target_class_ids`, `num_classes`) matches the **foreground definition** used in your masks when comparing numbers to external benchmarks.

---

## Accuracy

- **What it measures:** Fraction of pixels (or samples) predicted correctly vs ground truth at a fixed threshold (typically 0.5 on sigmoid logits after thresholding for binary masks).
- **Why it matters in segmentation:** Easy to interpret globally, but can be **misleading** when the background dominates rare foreground pixels.
- **Formula (pixel-wise):** \(\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}\)

## Precision

- **What it measures:** Of pixels predicted as foreground, how many are truly foreground.
- **Why it matters:** High precision means fewer **false alarms** on the object class—relevant when false detections are costly.
- **Formula:** \(\text{Precision} = \frac{TP}{TP + FP}\)

## Recall

- **What it measures:** Of all true foreground pixels, how many were retrieved by the model.
- **Why it matters:** High recall means fewer **missed** camouflaged regions.
- **Formula:** \(\text{Recall} = \frac{TP}{TP + FN}\)

## F1-score

- **What it measures:** Harmonic mean of precision and recall, balancing both.
- **Why it matters:** Useful single scalar when **precision–recall trade-off** matters.
- **Formula:** \(F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}\)

## Intersection over Union (IoU)

- **What it measures:** Overlap between predicted and ground-truth foreground regions: intersection divided by union.
- **Why it matters for camouflage segmentation:** IoU penalizes **both missed pixels and bloated predictions**, so it tracks **boundary quality** more strictly than raw accuracy—**critical** for segmentation-focused assessment.
- **Formula (class-wise or binary foreground):** \(\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}\)

## Dice coefficient

- **What it measures:** Dice similarity for binary (or soft) overlap; closely related to F1 for binary masks.
- **Formula:** \(\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2 TP}{2 TP + FP + FN}\)

Implemented as `dice_coefficient` in the notebooks (see `dice_coefficient` definition cell).
