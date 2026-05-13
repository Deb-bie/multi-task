# SynthRAD2025 Multi-task CycleGAN

A multi-task learning extension of Paired CycleGAN for joint MRI↔CT image synthesis and organ segmentation on the [SynthRAD2025](https://synthrad2025.grand-challenge.org/) dataset.  A shared encoder jointly learns to synthesise images in both directions and to segment anatomical structures, improving synthesis fidelity through anatomical consistency constraints.

---

## Project structure

```
synthrad2025-multitask/
├── configs/
│   ├── base_config.json            # Global hyperparameters
│   └── anatomy/
│       ├── head_neck.json          # Head & neck organ classes
│       ├── thorax.json             # Thorax organ classes
│       └── abdomen.json            # Abdomen organ classes
├── src/
│   ├── metrics.py                  # Centralised metric functions
│   └── models/
│       ├── shared_encoder.py       # SharedEncoder (ResNet backbone)
│       ├── synthesis_decoder.py    # SynthesisDecoder (G and F)
│       ├── seg_decoder.py          # SegDecoder (4-block U-Net head)
│       ├── multitask_cyclegan.py   # Full MultitaskCycleGAN
│       ├── discriminators.py       # PatchGAN + MultiScaleDiscriminator
│       └── utils.py                # init_weights, ImageBuffer
│   └── losses/
│       ├── gan_loss.py             # LSGAN (MSE)
│       ├── perceptual_loss.py      # VGG16 perceptual loss
│       ├── seg_loss.py             # DiceLoss + SegLoss
│       └── anatomy_loss.py         # AnatomyConsistencyLoss
├── train/
│   ├── train_multitask.py          # Training loop
│   └── validate.py                 # Validation loop
├── evaluate/
│   ├── ablation_runner.py          # 6-configuration ablation study
│   └── test_multitask.py           # Test-set evaluation + NIfTI export
├── notebooks/
│   └── results_analysis.ipynb     # Training curves, bar charts, heatmaps
├── scripts/
│   ├── run_totalsegmentator.sh     # Batch TotalSegmentator segmentation
│   └── prepare_splits.py           # Train/val/test split generation
└── figures/                        # Generated at runtime by the notebook
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install \
    torchmetrics \
    SimpleITK \
    scipy \
    pandas \
    seaborn \
    matplotlib \
    tqdm \
    jupyter \
    --break-system-packages    # omit if using a virtual environment
```

### 2. TotalSegmentator

TotalSegmentator is required to generate organ segmentation labels from the raw CT volumes.

```bash
pip install TotalSegmentator
# GPU is strongly recommended; --fast mode uses a lighter model
TotalSegmentator --help   # verify installation
```

---

## Data preparation

### Step 1 — Run TotalSegmentator on all CT volumes

```bash
export DATA_ROOT=/data/synthrad2025
export ANATOMY=head_neck          # repeat for thorax, abdomen
export OUTPUT_DIR=/data/synthrad2025_segs

bash scripts/run_totalsegmentator.sh
```

Segmentation labels are written to:
`OUTPUT_DIR/{ANATOMY}/{patient_id}/seg/{organ_name}.nii.gz`

### Step 2 — Copy segmentation labels into the dataset tree

After `run_totalsegmentator.sh` completes, place (or symlink) each patient's
`seg.nii.gz` (merged multi-class label volume) into the expected location:

`DATA_ROOT/{ANATOMY}/{patient_id}/seg.nii.gz`

A helper utility for merging per-organ binary masks into a single multi-label
volume can be called with SimpleITK:

```python
# Pseudocode — adapt paths as needed
import SimpleITK as sitk, numpy as np
merged = np.zeros(ref_shape, dtype=np.uint8)
for cls_idx, organ in enumerate(['brainstem','parotid_L',...], start=1):
    mask = sitk.GetArrayFromImage(sitk.ReadImage(f'seg/{organ}.nii.gz'))
    merged[mask > 0] = cls_idx
sitk.WriteImage(sitk.GetImageFromArray(merged), 'seg.nii.gz')
```

### Step 3 — Generate train/val/test splits

```bash
python scripts/prepare_splits.py \
    --data_root /data/synthrad2025 \
    --anatomy   head_neck \
    --split_dir splits/ \
    --seed      42
```

Repeat for `thorax` and `abdomen`.  Output: `splits/{anatomy}_split.json`.

---

## Training

### Single anatomy

```bash
python train/train_multitask.py \
    --anatomy      head_neck \
    --data_root    /data/synthrad2025 \
    --split_dir    splits/ \
    --config       configs/base_config.json \
    --output_dir   . \
    --ablation_name plus_anatomy_consistency
```

### All three anatomies (sequential)

```bash
for ANAT in head_neck thorax abdomen; do
    python train/train_multitask.py \
        --anatomy      "${ANAT}" \
        --data_root    /data/synthrad2025 \
        --split_dir    splits/ \
        --config       configs/base_config.json \
        --output_dir   . \
        --ablation_name plus_anatomy_consistency
done
```

Checkpoints are saved every 10 epochs to `checkpoints/{ablation_name}/{anatomy}/`.
The best model (by mean SSIM with the MR→CT SSIM floor guard) is saved as
`checkpoints/{ablation_name}/{anatomy}/best_model.pth`.

---

## Ablation study

Run all 6 ablation configurations for one anatomy:

```bash
python evaluate/ablation_runner.py \
    --anatomy    head_neck \
    --data_root  /data/synthrad2025 \
    --split_dir  splits/ \
    --base_config configs/base_config.json \
    --output_dir  .
```

Add `--dry_run` to print commands without executing training.

After all runs complete, a Markdown results table is printed to stdout and
saved to `results/ablation_summary_{anatomy}.csv`.

---

## Evaluation

Run test-set evaluation for a specific ablation:

```bash
python evaluate/test_multitask.py \
    --anatomy       head_neck \
    --ablation_name plus_anatomy_consistency \
    --data_root     /data/synthrad2025 \
    --split_dir     splits/ \
    --checkpoint    checkpoints/plus_anatomy_consistency/head_neck/best_model.pth \
    --config        configs/base_config.json \
    --output_dir    .
```

Per-patient NIfTI volumes are written to `outputs/{anatomy}/{patient_id}/`:
- `synthetic_ct.nii.gz`
- `synthetic_mr.nii.gz`
- `seg_pred.nii.gz`

Per-patient metrics (including HU-equivalent MAE) are saved to:
`results/{ablation_name}_{anatomy}_test_results.csv`

---

## Results analysis

Open the notebook:

```bash
cd notebooks
jupyter lab results_analysis.ipynb
```

Set `ANATOMY` in Cell 1, then run all cells top-to-bottom.  Figures are saved
to `figures/{anatomy}_{plot_name}.png` at 300 DPI.

---

## Results

*The following table is a placeholder.  Replace with values from your
`results/ablation_summary_{anatomy}.csv` after completing experiments.*

| Ablation                  | MRI→CT SSIM | MRI→CT MAE (HU) | CT→MR SSIM | Mean Dice | Best Epoch |
|---------------------------|:-----------:|:---------------:|:----------:|:---------:|:----------:|
| baseline_cyclegan         | —           | —               | —          | —         | —          |
| paired_cyclegan           | —           | —               | —          | —         | —          |
| plus_seg_loss             | —           | —               | —          | —         | —          |
| plus_anatomy_consistency  | —           | —               | —          | —         | —          |
| no_perceptual             | —           | —               | —          | —         | —          |
| no_warmup                 | —           | —               | —          | —         | —          |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2026synthrad,
  title   = {Multi-task Paired CycleGAN for Joint MRI-CT Synthesis and Organ Segmentation},
  author  = {Your Name and Co-Authors},
  journal = {arXiv preprint},
  year    = {2026},
}
```

SynthRAD2025 dataset:

```bibtex
@inproceedings{synthrad2025,
  title     = {SynthRAD2025 Grand Challenge},
  booktitle = {Medical Image Analysis},
  year      = {2025},
}
```
