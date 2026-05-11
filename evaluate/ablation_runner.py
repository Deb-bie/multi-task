"""
Ablation study runner for the multi-task CycleGAN (HN-focused).

Runs 18 ablation configurations sequentially on a single anatomical region
by launching ``train/train_multitask.py`` as a subprocess for each one.
Each ablation receives:

* Its own checkpoint directory: ``checkpoints/{ablation_name}/{anatomy}/``
* Its own CSV log: ``logs/{ablation_name}_{anatomy}.csv``
* A temporary config JSON that merges base_config + anatomy overrides +
  ablation-specific overrides.

After all runs finish (or are found to have existing checkpoints), a summary
table of best validation metrics is printed to stdout and saved as a CSV and
Markdown table.

Ablation groups (18 configurations total)
-----------------------------------------
GROUP 0 — Build-up (3 configs)
  g0_baseline_cyclegan      Pure bidirectional CycleGAN, no seg/anatomy
  g0_plus_seg               + segmentation decoder, no anatomy consistency
  g0_full_model             Full model — reference (no overrides)

GROUP 1 — Architecture (1 config)
  g1_separate_encoders      Domain-specific encoders E_MR + E_CT

GROUP 2 — Anatomy consistency design (5 configs)
  g2_anatomy_mr2ct_only     MR→CT consistency only (CT→MR disabled)
  g2_anatomy_soft_targets   Soft KD targets (T=2) vs hard argmax
  g2_lambda_anat_0p5        λ_anat = 0.5
  g2_lambda_anat_1          λ_anat = 1.0
  g2_lambda_anat_5          λ_anat = 5.0

GROUP 3 — Deep supervision (2 configs)
  g3_no_deep_sup            No auxiliary heads
  g3_deep_sup_coarse_only   Coarsest scale (1/4 res) only

GROUP 4 — Per-structure HN (4 configs)
  g4_no_brainstem           Class weight → 0 for brainstem
  g4_no_parotids            Class weight → 0 for parotid L+R
  g4_no_mandible            Class weight → 0 for mandible
  g4_cord_bg_only           Supervise only background + spinal cord

GROUP 5 — Warmup sensitivity (3 configs)
  g5_warmup_0               No warmup (seg active from epoch 1)
  g5_warmup_10              10-epoch warmup
  g5_warmup_20              20-epoch warmup

Usage::

    python evaluate/ablation_runner.py \\
        --anatomy head_neck \\
        --data_root /data/synthrad2025 \\
        --split_dir splits/ \\
        --base_config configs/base_config.json \\
        --output_dir . \\
        --epochs 100 \\
        [--dry_run]      # print commands without executing
        [--device 1]     # pin to GPU 1 (default: inherit CUDA_VISIBLE_DEVICES)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ablation configuration table
# ---------------------------------------------------------------------------

#: Each entry is a dict of config overrides applied ON TOP of
#: (base_config + anatomy_config).  Empty dict = full model (reference).
#:
#: Organised into five groups for the HN-focused ablation study:
#:
#: GROUP 0 — Build-up (incremental component contribution table)
#: GROUP 1 — Architecture design (shared vs separate encoder)
#: GROUP 2 — Anatomy consistency design (target mode, direction, λ sensitivity)
#: GROUP 3 — Deep supervision (scale selection)
#: GROUP 4 — Per-structure HN (which organs drive the benefit)
#: GROUP 5 — Warmup sensitivity (seg loss schedule)
#:
#: HN class index mapping (6 classes including background):
#:   0 = background  1 = brainstem  2 = parotid_L  3 = parotid_R
#:   4 = mandible    5 = spinal_cord

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── GROUP 0: Build-up ───────────────────────────────────────────────────
    # Answers: which components are necessary?

    "g0_baseline_cyclegan": {
        # Pure bidirectional CycleGAN — no seg, no anatomy, no paired, no perc
        "LAMBDA_SEG":          0,
        "LAMBDA_ANATOMY":      0,
        "LAMBDA_ANATOMY_CT2MR": 0,
        "LAMBDA_PAIRED_MR2CT": 0,
        "LAMBDA_PAIRED_CT2MR": 0,
        "LAMBDA_PERCEPTUAL":   0,
        "LAMBDA_IDENTITY":     0,
    },
    "g0_plus_seg": {
        # Add segmentation decoder — no anatomy consistency yet
        "LAMBDA_ANATOMY":      0,
        "LAMBDA_ANATOMY_CT2MR": 0,
    },
    "g0_full_model": {
        # Reference model — no overrides (all components active)
    },

    # ── GROUP 1: Architecture ───────────────────────────────────────────────
    # Answers: does a shared encoder actually help over domain-specific encoders?

    "g1_separate_encoders": {
        # Two independent encoders E_MR and E_CT instead of one shared E.
        # All cross-modal alignment must emerge from the cycle loss alone.
        "SHARED_ENCODER": False,
    },

    # ── GROUP 2: Anatomy consistency design ────────────────────────────────
    # Answers: how should the anatomy consistency loss be configured?

    "g2_anatomy_mr2ct_only": {
        # Disable CT→MRI anatomy consistency; keep MR→CT direction only.
        # Tests whether bidirectional consistency is necessary or if one
        # direction carries most of the benefit.
        "LAMBDA_ANATOMY_CT2MR": 0,
    },
    "g2_anatomy_soft_targets": {
        # Replace hard argmax pseudo-labels with temperature-scaled softmax
        # distributions, preserving the teacher's uncertainty signal.
        "ANATOMY_SOFT_TARGETS": True,
        "ANATOMY_TEMPERATURE":  2.0,
    },
    "g2_lambda_anat_0p5": {
        # λ_anat = 0.5 (weak anatomy signal)
        "LAMBDA_ANATOMY":      0.5,
        "LAMBDA_ANATOMY_CT2MR": 0.5,
    },
    "g2_lambda_anat_1": {
        # λ_anat = 1.0
        "LAMBDA_ANATOMY":      1.0,
        "LAMBDA_ANATOMY_CT2MR": 1.0,
    },
    "g2_lambda_anat_5": {
        # λ_anat = 5.0 (strong anatomy signal — may dominate synthesis)
        "LAMBDA_ANATOMY":      5.0,
        "LAMBDA_ANATOMY_CT2MR": 5.0,
    },

    # ── GROUP 3: Deep supervision ───────────────────────────────────────────
    # Answers: how many auxiliary scales are needed?

    "g3_no_deep_sup": {
        # No auxiliary heads — only the main segmentation head is supervised.
        "DEEP_SUPERVISION_WEIGHTS": [],
    },
    "g3_deep_sup_coarse_only": {
        # Only the 1/4-resolution (coarsest) auxiliary head.
        # Tests whether fine-scale auxiliary supervision is necessary.
        "DEEP_SUPERVISION_WEIGHTS": [0.4],
    },

    # ── GROUP 4: Per-structure (HN-specific) ───────────────────────────────
    # Answers: which structures drive the synthesis improvement?
    # Zero-weight classes are excluded from CE and Dice loss computation
    # but the decoder still predicts them (they become unsupervised).
    # Class index: 0=bg 1=brainstem 2=parotid_L 3=parotid_R 4=mandible 5=cord

    "g4_no_brainstem": {
        # Remove brainstem (hardest structure, small volume, poor CT contrast).
        "CLASS_WEIGHTS": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    },
    "g4_no_parotids": {
        # Remove both parotid glands (most clinically relevant for HN RT).
        "CLASS_WEIGHTS": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    },
    "g4_no_mandible": {
        # Remove mandible (large bony structure, most CT-distinct).
        "CLASS_WEIGHTS": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    },
    "g4_cord_bg_only": {
        # Supervise only background + spinal cord (easiest classes).
        # Shows lower bound of segmentation complexity for synthesis benefit.
        "CLASS_WEIGHTS": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },

    # ── GROUP 5: Warmup sensitivity ─────────────────────────────────────────
    # Answers: how sensitive is training to the seg loss warmup schedule?

    "g5_warmup_0": {
        # Segmentation loss active from epoch 1 — no warmup.
        "SEG_WARMUP_EPOCHS": 0,
    },
    "g5_warmup_10": {
        # 10-epoch warmup (2× the default of 5).
        "SEG_WARMUP_EPOCHS": 10,
    },
    "g5_warmup_20": {
        # 20-epoch warmup — GAN/cycle losses well-established before seg engages.
        "SEG_WARMUP_EPOCHS": 20,
    },
}

# Canonical display order — groups are printed together in the summary table
_ABLATION_ORDER: List[str] = list(ABLATION_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helper: read best metrics from a CSV log
# ---------------------------------------------------------------------------

def _best_row_from_csv(csv_path: Path) -> Optional[Dict[str, str]]:
    """Return the CSV row with the highest ``mean_ssim`` (avg of both dirs).

    If the file does not exist or is empty, returns ``None``.

    Args:
        csv_path: Path to the per-ablation CSV log written by the training
                  script.

    Returns:
        Dict of column→value strings for the best row, or ``None``.
    """
    if not csv_path.exists():
        return None

    best_row: Optional[Dict[str, str]] = None
    best_ssim = -1.0

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                mr2ct = float(row.get("mr2ct_ssim", -1))
                ct2mr = float(row.get("ct2mr_ssim", -1))
                mean  = 0.5 * (mr2ct + ct2mr)
            except (ValueError, TypeError):
                continue
            if mean > best_ssim:
                best_ssim = mean
                best_row  = dict(row)

    return best_row


# ---------------------------------------------------------------------------
# Helper: print summary table
# ---------------------------------------------------------------------------

def _print_summary(
    anatomy:     str,
    output_dir:  Path,
    metrics:     List[Tuple[str, Optional[Dict[str, str]]]],
) -> None:
    """Print a formatted summary table of best validation metrics.

    Args:
        anatomy:    Anatomical region name (for header only).
        output_dir: Output root (printed in header for reference).
        metrics:    List of ``(ablation_name, best_row_or_None)`` tuples.
    """
    cols = [
        ("mr2ct_ssim",  "MR→CT SSIM"),
        ("ct2mr_ssim",  "CT→MR SSIM"),
        ("mr2ct_mae",   "MR→CT MAE"),
        ("ct2mr_mae",   "CT→MR MAE"),
        ("mr2ct_psnr",  "MR→CT PSNR"),
        ("mean_dice",   "Mean Dice"),
    ]
    name_w  = max(len(name) for name, _ in metrics) + 2
    col_w   = 11

    # Header
    print(f"\n{'═' * (name_w + col_w * len(cols) + 2)}")
    print(f"  Ablation summary — {anatomy.upper()}  |  output: {output_dir}")
    print(f"{'═' * (name_w + col_w * len(cols) + 2)}")
    header = f"{'Ablation':<{name_w}}" + "".join(f"{label:>{col_w}}" for _, label in cols)
    print(header)
    print(f"{'─' * len(header)}")

    for ablation_name, row in metrics:
        if row is None:
            vals = "  (no data)" + " " * (col_w * len(cols) - 10)
        else:
            vals = ""
            for key, _ in cols:
                try:
                    vals += f"{float(row[key]):>{col_w}.4f}"
                except (KeyError, ValueError):
                    vals += f"{'N/A':>{col_w}}"
        print(f"{ablation_name:<{name_w}}{vals}")

    print(f"{'═' * (name_w + col_w * len(cols) + 2)}\n")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_ablations(
    anatomy:     str,
    data_root:   str,
    split_dir:   str,
    base_config: str,
    output_dir:  str,
    dry_run:     bool = False,
    train_script: Optional[str] = None,
    epochs:      Optional[int] = None,
    device:      Optional[str] = None,
) -> None:
    """Run all 6 ablation configurations sequentially.

    For each ablation:
    1. Merge base_config → anatomy overrides → ablation overrides.
    2. Write the merged config to a temporary JSON file.
    3. Launch ``train/train_multitask.py`` as a subprocess, passing
       ``--config /tmp/cfg.json --ablation_name {name}``.
    4. Collect wall-clock time.

    When all ablations are done, print a summary table.

    Args:
        anatomy:     Anatomical region (``"head_neck"``, ``"thorax"``,
                     ``"abdomen"``).
        data_root:   Root of the SynthRAD2025 dataset.
        split_dir:   Directory with train/val/test split files.
        base_config: Path to ``configs/base_config.json``.
        output_dir:  Root output directory.
        dry_run:     If ``True``, print commands and exit without running.
        train_script: Optional path override to ``train_multitask.py``.
        epochs:      Override ``EPOCHS`` in the merged config.  If ``None``,
                     the value from base_config / anatomy config is used.
                     Pass ``100`` to match a full-model run of 100 epochs.
        device:      CUDA device string passed to each subprocess via the
                     ``CUDA_VISIBLE_DEVICES`` environment variable
                     (e.g. ``"0"``, ``"1"``).  ``None`` = inherit from caller.
    """
    root         = Path(__file__).resolve().parents[1]
    output_dir_p = Path(output_dir)
    logs_dir     = output_dir_p / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Locate training script
    if train_script is None:
        train_script_path = root / "train" / "train_multitask.py"
    else:
        train_script_path = Path(train_script)

    # Load base config
    with open(base_config) as fh:
        base_cfg: Dict[str, Any] = json.load(fh)

    # Load anatomy-specific overrides
    anatomy_cfg_path = root / "configs" / "anatomy" / f"{anatomy}.json"
    anatomy_cfg: Dict[str, Any] = {}
    if anatomy_cfg_path.exists():
        with open(anatomy_cfg_path) as fh:
            raw = json.load(fh)
        anatomy_cfg = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Collect results for summary
    summary_rows: List[Tuple[str, Optional[Dict[str, str]]]] = []

    for ablation_name in _ABLATION_ORDER:
        overrides = ABLATION_CONFIGS[ablation_name]

        # ── Build merged config ──────────────────────────────────────────
        merged: Dict[str, Any] = {}
        merged.update(base_cfg)
        merged.update(anatomy_cfg)
        merged.update(overrides)
        # Remove comment/metadata keys that JSON5 would allow but stdlib won't
        merged = {k: v for k, v in merged.items() if not k.startswith("_")}

        # Apply CLI epoch override LAST so it always wins
        if epochs is not None:
            merged["EPOCHS"] = epochs
            # Also fix DECAY_EPOCH so LR decay starts at the right point
            # (default: decay from 50% of EPOCHS onwards)
            if "DECAY_EPOCH" not in overrides:
                merged["DECAY_EPOCH"] = epochs // 2

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"ablation_{ablation_name}_",
            delete=False,
        ) as tmp:
            json.dump(merged, tmp, indent=2)
            tmp_cfg_path = tmp.name

        # ── Build subprocess command ─────────────────────────────────────
        cmd = [
            sys.executable,
            str(train_script_path),
            "--anatomy",      anatomy,
            "--data_root",    data_root,
            "--split_dir",    split_dir,
            "--config",       tmp_cfg_path,
            "--output_dir",   output_dir,
            "--ablation_name", ablation_name,
        ]

        # Checkpoint dir this ablation writes to
        ckpt_dir = output_dir_p / "checkpoints" / ablation_name / anatomy
        csv_path = logs_dir / f"{ablation_name}_{anatomy}.csv"

        print(f"\n{'─' * 70}")
        print(f"Ablation : {ablation_name}")
        print(f"Overrides: {overrides if overrides else '(none — full model)'}")
        print(f"Ckpt dir : {ckpt_dir}")
        print(f"CSV log  : {csv_path}")
        print(f"Command  : {' '.join(cmd)}")

        if dry_run:
            if device is not None:
                print(f"Env      : CUDA_VISIBLE_DEVICES={device}")
            print("[dry-run] Skipping execution.")
            summary_rows.append((ablation_name, _best_row_from_csv(csv_path)))
            continue

        # ── Execute ──────────────────────────────────────────────────────
        env = os.environ.copy()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device

        t0 = time.monotonic()
        result = subprocess.run(cmd, check=False, env=env)
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            print(f"[WARNING] Ablation '{ablation_name}' exited with code "
                  f"{result.returncode}  ({elapsed/60:.1f} min)")
        else:
            print(f"  Done in {elapsed/60:.1f} min.")

        # Read best metrics from this ablation's CSV
        summary_rows.append((ablation_name, _best_row_from_csv(csv_path)))

        # Clean up temp config
        try:
            os.unlink(tmp_cfg_path)
        except OSError:
            pass

    # ── Summary table ─────────────────────────────────────────────────────
    _print_summary(anatomy, output_dir_p, summary_rows)

    # ── Markdown results table + CSV ───────────────────────────────────────
    _print_markdown_table(anatomy, output_dir_p, summary_rows, logs_dir)


# ---------------------------------------------------------------------------
# Markdown results table generator
# ---------------------------------------------------------------------------

def _print_markdown_table(
    anatomy:     str,
    output_dir:  Path,
    metrics:     List[Tuple[str, Optional[Dict[str, str]]]],
    logs_dir:    Path,
) -> None:
    """Print a Markdown results table and save it as a CSV summary.

    The table contains one row per ablation configuration with the following
    columns:

    * **MRI→CT SSIM** — best val SSIM for MR→CT synthesis.
    * **MRI→CT MAE (HU)** — MAE in Hounsfield Units.  The training CSV stores
      MAE in normalised [-1,1] units; this function converts to HU using:
      ``HU_MAE = normalised_MAE * 4000 / 2`` (half the full 4000 HU range,
      since [-1,1] spans 2 units).  If the CSV already stores HU values the
      same conversion is applied, but training logs typically store normalised.
    * **CT→MR SSIM** — best val SSIM for CT→MR synthesis.
    * **Mean Dice** — mean foreground Dice across classes.
    * **Best epoch** — epoch at which the best mean SSIM was achieved.

    The table is also saved as:
        ``results/ablation_summary_{anatomy}.csv``

    Args:
        anatomy:    Anatomical region name.
        output_dir: Root output directory.
        metrics:    List of ``(ablation_name, best_row_or_None)`` tuples.
        logs_dir:   Directory containing per-ablation CSV logs (for best-epoch
                    lookup, which is already embedded in *metrics*).
    """
    # ── Markdown columns (key, header, format) ─────────────────────────────
    MD_COLS = [
        ("mr2ct_ssim", "MRI→CT SSIM",    ".4f"),
        ("mr2ct_mae",  "MRI→CT MAE (HU)", ".1f"),
        ("ct2mr_ssim", "CT→MR SSIM",     ".4f"),
        ("mean_dice",  "Mean Dice",       ".4f"),
        ("epoch",      "Best Epoch",      "d"),
    ]

    _HU_NORM_SCALE = 2000.0  # normalised MAE → HU: multiply by (4000/2)

    def _fmt_cell(row: Optional[Dict[str, str]], key: str, fmt: str) -> str:
        if row is None:
            return "—"
        raw = row.get(key, None)
        if raw is None:
            return "—"
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return str(raw)
        # Convert normalised MAE to HU-equivalent
        if key == "mr2ct_mae":
            val = val * _HU_NORM_SCALE
        if fmt == "d":
            return str(int(round(val)))
        return format(val, fmt)

    # ── Build Markdown string ───────────────────────────────────────────────
    headers  = ["Ablation"] + [h for _, h, _ in MD_COLS]
    col_w    = [max(len(h), 18) for h in headers]

    def _row_str(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " |"

    sep_row  = "|" + "|".join("-" * (w + 2) for w in col_w) + "|"

    lines = [
        "",
        f"## Ablation results — {anatomy.upper()}",
        "",
        _row_str(headers),
        sep_row,
    ]
    for ablation_name, row in metrics:
        cells = [ablation_name] + [
            _fmt_cell(row, key, fmt) for key, _, fmt in MD_COLS
        ]
        lines.append(_row_str(cells))
    lines.append("")

    md_table = "\n".join(lines)
    print(md_table)

    # ── Save CSV summary ───────────────────────────────────────────────────
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = results_dir / f"ablation_summary_{anatomy}.csv"

    csv_headers = ["ablation_name"] + [h for _, h, _ in MD_COLS]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_headers)
        writer.writeheader()
        for ablation_name, row in metrics:
            csv_row: Dict[str, Any] = {"ablation_name": ablation_name}
            for key, header, fmt in MD_COLS:
                csv_row[header] = _fmt_cell(row, key, fmt)
            writer.writerow(csv_row)

    print(f"Ablation summary CSV saved to: {csv_path}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation runner.

    Returns:
        Parsed argument namespace.
    """
    p = argparse.ArgumentParser(
        description="Run all 6 multi-task CycleGAN ablation configurations."
    )
    p.add_argument(
        "--anatomy",
        required=True,
        choices=["head_neck", "thorax", "abdomen"],
        help="Anatomical region to ablate.",
    )
    p.add_argument(
        "--data_root",
        required=True,
        help="Root directory of the SynthRAD2025 dataset.",
    )
    p.add_argument(
        "--split_dir",
        required=True,
        help="Directory containing train/val/test split files.",
    )
    p.add_argument(
        "--base_config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "base_config.json"),
        help="Path to base_config.json (ablation overrides are applied on top).",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root output directory for checkpoints and logs.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing training.",
    )
    p.add_argument(
        "--train_script",
        default=None,
        help="Override path to train_multitask.py (default: auto-detected).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override EPOCHS for all ablation runs.  "
            "If omitted, the value in base_config.json is used.  "
            "Example: --epochs 100 to match a 100-epoch full-model run.  "
            "DECAY_EPOCH is automatically set to epochs // 2 unless already "
            "present in the ablation override."
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help=(
            "CUDA device index passed as CUDA_VISIBLE_DEVICES to every "
            "training subprocess (e.g. --device 1 to use GPU 1).  "
            "Omit to inherit the caller's CUDA_VISIBLE_DEVICES."
        ),
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the ablation runner."""
    args = parse_args()
    run_ablations(
        anatomy=args.anatomy,
        data_root=args.data_root,
        split_dir=args.split_dir,
        base_config=args.base_config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        train_script=args.train_script,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
