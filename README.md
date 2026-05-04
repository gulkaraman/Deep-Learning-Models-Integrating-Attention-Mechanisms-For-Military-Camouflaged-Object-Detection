# Deep Learning Models Integrating Attention Mechanisms For Military Camouflaged Object Detection

Official companion repository for **military camouflaged object detection and segmentation** using **attention-enhanced deep networks** on the **ACD1K** dataset.

**GitHub:** [https://github.com/gulkaraman/Deep-Learning-Models-Integrating-Attention-Mechanisms-For-Military-Camouflaged-Object-Detection](https://github.com/gulkaraman/Deep-Learning-Models-Integrating-Attention-Mechanisms-For-Military-Camouflaged-Object-Detection)

## Article information

| Field | Detail |
|--------|--------|
| **Title** | Deep Learning Models Integrating Attention Mechanisms For Military Camouflaged Object Detection |
| **Journal** | El-Cezeri Journal of Science and Engineering |
| **Year / Volume / Issue** | 2026, Volume 13, Issue 2 |
| **Pages** | 146–160 |
| **DOI** | [10.31202/ecjse.1747013](https://doi.org/10.31202/ecjse.1747013) |
| **Article link** | [DergiPark article 1747013](https://dergipark.org.tr/en/pub/ecjse/article/1747013) |

**Authors:** Nilgün Şengöz, Gül Karaman, Mert Samet Çeliker, Nazmi Yücel Çan

## Overview

This repository provides **Jupyter notebook experiments** (archival) and a **modular Python pipeline** under `src/` for training and evaluation with TensorFlow / Keras, ResNet-50 backbones, optional CLAHE, and attention-based U-Net / U-Net++ decoders.

## Published paper results

Reported headline metrics (see `docs/RESULTS.md` for the full table and interpretation):

- **Attention U-Net + CLAHE + Adamax + lr 1e-5 + dropout 0.2:** Accuracy **96.88%**, IoU **92.01%**
- **Attention U-Net++ + Adam:** Accuracy **98.32%**, IoU **82.09%**

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── config/
│   ├── example_config.yaml
│   ├── paper_attention_unet_clahe_adamax.yaml
│   └── paper_attention_unetpp_adam.yaml
├── docs/
├── src/
├── mimariler/
├── checkpoints/          # local weights (only .gitkeep tracked; see .gitignore)
├── outputs/              # local metrics / predictions / figures (see .gitignore)
├── att unet *.ipynb      # Attention U-Net experiment notebooks
└── att-unet++*.ipynb     # Attention U-Net++ experiment notebooks
```

## Script-based Usage

Run from the **repository root** (this folder).

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Train

```bash
python -m src.training.train --config config/paper_attention_unet_clahe_adamax.yaml
```

### Evaluate

```bash
python -m src.evaluation.evaluate --config config/paper_attention_unet_clahe_adamax.yaml --checkpoint checkpoints/attention_unet_paper.keras --split test
```

### Predict

```bash
python -m src.evaluation.predict --config config/paper_attention_unet_clahe_adamax.yaml --checkpoint checkpoints/attention_unet_paper.keras --input path/to/images --output outputs/predictions
```

## Notebook-based usage

Open the matching `*.ipynb` files in Jupyter, set dataset `PATH` variables, and run cells. See `docs/REPRODUCIBILITY.md`.

## Citation

```bibtex
@article{Sengoz2026CamouflagedObjectDetection,
  title={Deep Learning Models Integrating Attention Mechanisms For Military Camouflaged Object Detection},
  author={Şengöz, Nilgün and Karaman, Gül and Çeliker, Mert Samet and Çan, Nazmi Yücel},
  journal={El-Cezeri Journal of Science and Engineering},
  volume={13},
  number={2},
  pages={146--160},
  year={2026},
  doi={10.31202/ecjse.1747013}
}
```

Machine-readable metadata: `CITATION.cff`.

## Acknowledgement

We would like to thank all contributors, supervisors, and colleagues who supported this research.

## License / usage note

See `LICENSE`. Do not commit **ACD1K** data or large model weights to Git; use `dataset/` and `checkpoints/` locally only.

---

## Türkçe kısa özet

Bu depo, **kamufle askerî nesnelerin** görüntüde **tespit ve segmentasyonu** için **dikkat mekanizmalarıyla güçlendirilmiş derin öğrenme** modellerini ve deneysel not defterlerini içerir. Ayrıntılar için `docs/` klasörüne bakınız.
