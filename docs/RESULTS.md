# Experimental results (from the paper)

The table below reproduces the **key headline numbers** reported in *Deep Learning Models Integrating Attention Mechanisms For Military Camouflaged Object Detection* (El-Cezeri Journal of Science and Engineering, 2026). **These are paper-reported metrics**, not values recomputed inside this README.

| Model | Preprocessing | Optimizer | Learning Rate | Dropout | Accuracy | IoU |
|-------|-----------------|-----------|---------------|---------|----------|-----|
| Attention U-Net / ResNet-50 | CLAHE | Adamax | 1e-5 | 0.2 | 96.88% | 92.01% |
| Attention U-Net++ | Similar experimental setup | Adam | - | - | 98.32% | 82.09% |

## Interpretation

- **Attention U-Net + CLAHE + Adamax** achieves a **higher IoU**, indicating **stronger overlap** between predicted and true object regions. For **segmentation-centric** deployment (accurate object extent and boundaries), this configuration is particularly compelling.
- **Attention U-Net++ with Adam** attains **higher accuracy**, which can reflect **overall pixel correctness** under the dataset’s class balance and evaluation protocol. It is attractive when **global correctness** is the dominant operational criterion.
- **Model choice should be application-driven:** prioritize **IoU** when mask quality and boundary fidelity matter; consider **accuracy** alongside domain constraints when decisions depend on aggregate pixel agreement.

## Relation to this repository

Numerical results are produced by running the corresponding **notebook experiments** on the **ACD1K** split used in the study. Notebook filenames encode optimizer, dropout, CLAHE presence, and architecture variant—see `docs/TRAINING.md` for the **paper vs notebook** alignment notes.
