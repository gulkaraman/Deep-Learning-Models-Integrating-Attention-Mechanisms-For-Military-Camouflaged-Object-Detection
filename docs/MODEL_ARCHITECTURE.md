# Model architectures

All learnable models in this repository are implemented **inside the Jupyter notebooks** (TensorFlow / Keras). There is no separate `models.py`; the sections below map **concepts → notebook filenames → function names**.

## Attention U-Net with ResNet-50 encoder

### Encoder (ResNet-50)

**ResNet-50** (`tensorflow.keras.applications.ResNet50`) is used as a **strong convolutional backbone** for multi-scale feature extraction. Encoder stages provide skip connections at multiple resolutions, which is essential for recovering fine object boundaries after downsampling.

### Decoder and skip connections

The decoder path upsamples feature maps and fuses them with encoder skips in a **U-shaped** layout, producing a dense per-pixel prediction map.

### Attention Gate on skip connections

The **`attention_gate(x, g, inter_channel)`** function (same name across Attention U-Net notebooks) implements a **gating signal** `g` from the decoder that modulates the encoder skip `x`. Intuitively, the gate **suppresses irrelevant background responses** and **emphasizes spatial locations** that are informative for the current decoding stage—useful when camouflaged objects are visually similar to surroundings.

### Where to find the code

| Component | Location |
|-----------|----------|
| Attention U-Net (ResNet-50 + gates + decoder) | Notebooks matching `att unet *.ipynb` (space in filename) |
| `attention_gate`, `build_unet` (or equivalent builder) | Code cells in those notebooks (search within the notebook) |

Example notebooks (non-exhaustive): `att unet adam dropout=0,2.ipynb`, `att unet adamax dropout=0,2-clahesiz.ipynb`, `att unet adamax dropout=0,5.ipynb`.

## U-Net++ (nested / dense skip connections)

### Nested skip structure

**U-Net++** replaces plain skip connections with a **nested series of convolutional blocks** on skip pathways, enabling **dense multi-scale feature aggregation** before fusion into the decoder. This tends to enrich gradient flow and representation mixing across scales compared with the original U-Net.

### Attention U-Net++

Here, the nested U-Net++ topology is combined with **attention gating** on skip-style connections, aiming for **stronger selective fusion** of multi-scale features—helpful when foreground cues are subtle.

### Where to find the code

| Component | Location |
|-----------|----------|
| Attention U-Net++ with ResNet-50 | Notebooks matching `att-unet++*.ipynb` |
| Builder | `build_resnet50_attention_unet_plus_plus(...)` (code cell; name may vary slightly—search in notebook) |

Example notebooks: `att-unet++-adam-lr=1e-5-droput=0,2-clahesiz.ipynb`, `att-unet++-adamax-lr=1e-5-droput=0,2-claheli.ipynb`.

## Diagrams

Static figures illustrating the architectures and flow are under `mimariler/` (for example `attention-unet_model_diagram.png`, `attention-unetplusplus-model-diagam.png`, `kod_akis_semasi.png`).
