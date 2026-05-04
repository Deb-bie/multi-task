"""
Ablation study runner for the multi-task CycleGAN.

Runs all 6 ablation configurations sequentially on a single anatomical
region by launching ``train/train_multitask.py`` as a subprocess for each
configuration.  Each ablation receives:

* Its own checkpoint directory: ``checkpoints/{ablation_name}/{anatomy}/``
* Its own CSV log: ``logs/{ablation_name}_{anatomy}.csv``
* A temporary config JSON that merges base_config + anatomy overrides +
  ablation-specific overrides.

After all runs finish (or are found to have existing checkpoints),
a summary table of best validation metrics is printed to stdout.

Ablation configurations
-----------------------
+---------------------------+-------------------------------------------+
| Name                      | Overrides from base config                |
+===========================+===========================================+
| baseline_cyclegan         | Removes seg, anatomy, paired, perc, idt  |
| paired_cyclegan           | Removes seg, anatomy (keeps paired+perc)  |
| plus_seg_loss             | Removes anatomy only                      |
| plus_anatomy_consistency  | Full model (no overrides)                 |
| no_perceptual             | Sets LAMBDA_PERCEPTUAL=0                  |
| no_warmup                 | Sets SEG_WARMUP_EPOCHS=0                  |
+---------------------------+-------------------------------------------+

Usage::

    python evaluate/ablation_runner.py \\
        --anatomy head_neck \\
        --data_root /data/synthrad2025 \\
        --split_dir splits/ \\
        --base_config configs/base_config.json \\
        --output_dir . \\
        [--dry_run]          # print commands without executing
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
#: (base_config + anatomy_config).  Empty dict = full model.
ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_cyclegan": {
        # Pure unpaired CycleGAN: remove all paired/seg/perc supervision
        "LAMBDA_SEG":          0,
        "LAMBDA_ANATOMY":      0,
        "LAMBDA_PAIRED_MR2CT": 0,
        "LAMBDA_PAIRED_CT2MR": 0,
        "LAMBDA_PERCEPTUAL":   0,
        "LAMBDA_IDENTITY":     0,
    },
    "paired_cyclegan": {
        # Paired CycleGAN without segmentation tasks
        "LAMBDA_SEG":     0,
        "LAMBDA_ANATOMY": 0,
    },
    "plus_seg_loss": {
        # Paired CycleGAN + seg supervision, no anatomy consistency
        "LAMBDA_ANATOMY": 0,
    },
    "plus_anatomy_consistency": {
        # Full model — no overrides
    },
    "no_perceptual": {
        # Ablate the perceptual loss only
        "LAMBDA_PERCEPTUAL": 0,
    },
    "no_warmup": {
        # Train segmentation at full weight from epoch 1
        "SEG_WARMUP_EPOCHS": 0,
    },
}

# Canonical display order for the summary table
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
            print("[dry-run] Skipping execution.")
            summary_rows.append((ablation_name, _best_row_from_csv(csv_path)))
            continue

        # ── Execute ──────────────────────────────────────────────────────
        t0 = time.monotonic()
        result = subprocess.run(cmd, check=False)
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
    )


if __name__ == "__main__":
    main()
