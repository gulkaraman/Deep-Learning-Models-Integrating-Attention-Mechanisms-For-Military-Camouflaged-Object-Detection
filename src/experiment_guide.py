"""
Experiment code map for the camouflaged object segmentation study.

Primary archive
---------------
Jupyter notebooks at the repository root (``att unet *.ipynb``,
``att-unet++*.ipynb``) remain the **historical experiment record**.

Script-based pipeline
---------------------
Modular TensorFlow code now lives under ``src/`` (see ``README.md`` section
*Script-based Usage* and ``docs/REPRODUCIBILITY.md``). Start from:

- ``python -m src.training.train --config config/example_config.yaml``
- ``python -m src.evaluation.evaluate --config ... --checkpoint ...``
- ``python -m src.evaluation.predict --config ... --checkpoint ... --input ...``

Notebook families
-----------------
- ``att unet *.ipynb``: ResNet-50 **Attention U-Net** ablations.
- ``att-unet++*.ipynb``: ResNet-50 **Attention U-Net++** ablations.

Core symbols (notebooks)
------------------------
- **Dataset loader:** ``load_data(img_path, mask_path)``
- **CLAHE branch:** LAB + ``cv2.createCLAHE`` inside ``load_data`` when enabled.
- **Attention gate:** ``attention_gate(...)`` (two variants across notebook families).
- **Builders:** ``build_unet``, ``build_resnet50_attention_unet_plus_plus``.
- **Metrics / compile / fit:** see notebook cells (mirrored in ``src/training/``).

For prose documentation, see ``docs/`` and ``README.md``.
"""

NOTEBOOK_FAMILY_ATTENTION_UNET = "att unet *.ipynb"
NOTEBOOK_FAMILY_ATTENTION_UNET_PLUS_PLUS = "att-unet++*.ipynb"
