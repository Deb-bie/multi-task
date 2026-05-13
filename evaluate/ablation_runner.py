"""
Ablation study runner for the multi-task CycleGAN (HN-focused).

Runs 3 ablation configurations sequentially on a single anatomical region
by launching ``train/train_multitask.py`` as a subprocess for each one.
Each ablation receives:

* Its own checkpoint directory: ``checkpoints/{ablation_name}/{anatomy}/``
* Its own CSV log: ``logs/{ablation_name}_{anatomy}.csv``
* A temporary config JSON that merges base_config + anatomy overrides +
  ablation-specific overrides.

After all runs finish, a summary table of best validation metrics is
printed to stdout and saved as a CSV and Markdown table.

Ablation configurations (3 total)
----------------------------------
  g0_baseline_cyclegan   Pure bidirectional CycleGAN — no seg, no anatomy loss
  g0_plus_seg            + segmentation decoder, anatomy consistency disabled
  g1_separate_encoders   Separate per-domain encoders E_MR + E_CT (no shared E)

Each is compared against the full model trained by train_multitask.py.

Usage::

    python evaluate/ablation_runner.py \\
        --anatomy head_neck \\
        --data_root /data/synthrad2025 \\
        --split_dir splits/ \\
        --base_config configs/base_config.json \\
        --output_dir . \\
        --epochs 50 \\
        [--dry_run]      # print commands without executing
        [--device 0]     # pin to a specific GPU (default: inherit CUDA_VISIBLE_DEVICES)
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a timestamped log line to stdout (unbuffered)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _banner(msg: str, char: str = "═", width: int = 70) -> None:
    """Print a banner line."""
    print(f"\n{char * width}", flush=True)
    print(f"  {msg}", flush=True)
    print(f"{char * width}", flush=True)


# ---------------------------------------------------------------------------
# Ablation configuration table
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── Baseline: pure bidirectional CycleGAN ──────────────────────────────
    # No segmentation head, no anatomy consistency, no paired/perceptual loss.
    # Shows synthesis quality when the model has no anatomical guidance.
    "g0_baseline_cyclegan": {
        "LAMBDA_SEG":           0,
        "LAMBDA_ANATOMY":       0,
        "LAMBDA_ANATOMY_CT2MR": 0,
        "LAMBDA_PAIRED_MR2CT":  0,
        "LAMBDA_PAIRED_CT2MR":  0,
        "LAMBDA_PERCEPTUAL":    0,
        "LAMBDA_IDENTITY":      0,
    },

    # ── + Segmentation decoder, anatomy consistency disabled ───────────────
    # Adds the segmentation head but zeros out the anatomy consistency loss.
    # Isolates the contribution of the anatomy loss vs the seg head alone.
    "g0_plus_seg": {
        "LAMBDA_ANATOMY":       0,
        "LAMBDA_ANATOMY_CT2MR": 0,
    },

    # ── Separate per-domain encoders ───────────────────────────────────────
    # Replaces the single shared encoder with domain-specific E_MR + E_CT.
    # Tests whether the shared encoder's cross-modal alignment is necessary.
    "g1_separate_encoders": {
        "SHARED_ENCODER": False,
    },
}

_ABLATION_ORDER: List[str] = list(ABLATION_CONFIGS.keys())

_ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "g0_baseline_cyclegan":  "Pure CycleGAN — no seg, no anatomy loss",
    "g0_plus_seg":           "+ seg decoder, anatomy consistency OFF",
    "g1_separate_encoders":  "Separate encoders E_MR + E_CT (no shared E)",
}


# ---------------------------------------------------------------------------
# Helper: read best metrics from a CSV log
# ---------------------------------------------------------------------------

def _best_row_from_csv(csv_path: Path) -> Optional[Dict[str, str]]:
    """Return the CSV row with the highest mean SSIM (avg of both directions).

    Args:
        csv_path: Path to the per-ablation CSV log written by the training script.

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
    anatomy:    str,
    output_dir: Path,
    metrics:    List[Tuple[str, Optional[Dict[str, str]]]],
) -> None:
    cols = [
        ("mr2ct_ssim", "MR→CT SSIM"),
        ("ct2mr_ssim", "CT→MR SSIM"),
        ("mr2ct_mae",  "MR→CT MAE"),
        ("ct2mr_mae",  "CT→MR MAE"),
        ("mr2ct_psnr", "MR→CT PSNR"),
        ("mean_dice",  "Mean Dice"),
    ]
    name_w = max(len(name) for name, _ in metrics) + 2
    col_w  = 11

    print(f"\n{'═' * (name_w + col_w * len(cols) + 2)}", flush=True)
    print(f"  Ablation summary — {anatomy.upper()}  |  output: {output_dir}", flush=True)
    print(f"{'═' * (name_w + col_w * len(cols) + 2)}", flush=True)
    header = f"{'Ablation':<{name_w}}" + "".join(f"{label:>{col_w}}" for _, label in cols)
    print(header, flush=True)
    print(f"{'─' * len(header)}", flush=True)

    for ablation_name, row in metrics:
        if row is None:
            vals = "  (no data)"
        else:
            vals = ""
            for key, _ in cols:
                try:
                    vals += f"{float(row[key]):>{col_w}.4f}"
                except (KeyError, ValueError):
                    vals += f"{'N/A':>{col_w}}"
        print(f"{ablation_name:<{name_w}}{vals}", flush=True)

    print(f"{'═' * (name_w + col_w * len(cols) + 2)}\n", flush=True)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_ablations(
    anatomy:      str,
    data_root:    str,
    split_dir:    str,
    base_config:  str,
    output_dir:   str,
    dry_run:      bool = False,
    train_script: Optional[str] = None,
    epochs:       Optional[int] = None,
    device:       Optional[str] = None,
) -> None:
    """Run all 3 ablation configurations sequentially.

    Args:
        anatomy:      Anatomical region (``"head_neck"``, ``"thorax"``, ``"abdomen"``).
        data_root:    Root of the SynthRAD2025 dataset.
        split_dir:    Directory with train/val/test split JSON files.
        base_config:  Path to ``configs/base_config.json``.
        output_dir:   Root output directory for checkpoints and logs.
        dry_run:      If ``True``, print commands and exit without running.
        train_script: Optional path override to ``train_multitask.py``.
        epochs:       Override ``EPOCHS`` in the merged config.
        device:       CUDA device string for ``CUDA_VISIBLE_DEVICES``.
    """
    n_total      = len(_ABLATION_ORDER)
    root         = Path(__file__).resolve().parents[1]
    output_dir_p = Path(output_dir)
    logs_dir     = output_dir_p / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if train_script is None:
        train_script_path = root / "train" / "train_multitask.py"
    else:
        train_script_path = Path(train_script)

    _banner(f"MT-CycleGAN Ablation Study  —  {anatomy.upper()}")
    _log(f"Anatomy      : {anatomy}")
    _log(f"Data root    : {data_root}")
    _log(f"Split dir    : {split_dir}")
    _log(f"Output dir   : {output_dir}")
    _log(f"Epochs       : {epochs if epochs is not None else '(from base_config)'}")
    _log(f"GPU device   : {device if device is not None else '(inherited)'}")
    _log(f"Train script : {train_script_path}")
    _log(f"Configs      : {n_total}  →  {', '.join(_ABLATION_ORDER)}")
    if dry_run:
        _log("Mode         : DRY RUN — commands will be printed but not executed")

    # Load base config
    _log("Loading base config...")
    with open(base_config) as fh:
        base_cfg: Dict[str, Any] = json.load(fh)
    _log(f"  Base config loaded from: {base_config}")

    # Load anatomy-specific overrides
    anatomy_cfg_path = root / "configs" / "anatomy" / f"{anatomy}.json"
    anatomy_cfg: Dict[str, Any] = {}
    if anatomy_cfg_path.exists():
        with open(anatomy_cfg_path) as fh:
            raw = json.load(fh)
        anatomy_cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
        _log(f"  Anatomy config loaded from: {anatomy_cfg_path}")
    else:
        _log(f"  No anatomy config found at {anatomy_cfg_path} — using base only")

    summary_rows: List[Tuple[str, Optional[Dict[str, str]]]] = []
    job_start = time.monotonic()

    for idx, ablation_name in enumerate(_ABLATION_ORDER, start=1):
        overrides = ABLATION_CONFIGS[ablation_name]
        desc      = _ABLATION_DESCRIPTIONS.get(ablation_name, "")

        _banner(
            f"[{idx}/{n_total}]  {ablation_name}",
            char="─", width=70,
        )
        _log(f"Description  : {desc}")
        _log(f"Overrides    : {overrides if overrides else '(none)'}")

        # ── Build merged config ────────────────────────────────────────────
        merged: Dict[str, Any] = {}
        merged.update(base_cfg)
        merged.update(anatomy_cfg)
        merged.update(overrides)
        merged = {k: v for k, v in merged.items() if not k.startswith("_")}

        if epochs is not None:
            merged["EPOCHS"] = epochs
            if "DECAY_EPOCH" not in overrides:
                merged["DECAY_EPOCH"] = epochs // 2
            _log(f"Epoch override: EPOCHS={epochs}, DECAY_EPOCH={merged['DECAY_EPOCH']}")

        # Write temp config
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"ablation_{ablation_name}_",
            delete=False,
        ) as tmp:
            json.dump(merged, tmp, indent=2)
            tmp_cfg_path = tmp.name
        _log(f"Temp config  : {tmp_cfg_path}")

        # ── Build command ──────────────────────────────────────────────────
        cmd = [
            sys.executable,
            str(train_script_path),
            "--anatomy",       anatomy,
            "--data_root",     data_root,
            "--split_dir",     split_dir,
            "--config",        tmp_cfg_path,
            "--output_dir",    output_dir,
            "--ablation_name", ablation_name,
        ]

        ckpt_dir = output_dir_p / "checkpoints" / ablation_name / anatomy
        csv_path = logs_dir / f"{ablation_name}_{anatomy}.csv"

        _log(f"Ckpt dir     : {ckpt_dir}")
        _log(f"CSV log      : {csv_path}")
        _log(f"Command      : {' '.join(cmd)}")

        if dry_run:
            if device is not None:
                _log(f"Env          : CUDA_VISIBLE_DEVICES={device}")
            _log("[dry-run] Skipping execution.")
            summary_rows.append((ablation_name, _best_row_from_csv(csv_path)))
            continue

        # ── Execute ────────────────────────────────────────────────────────
        env = os.environ.copy()
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device

        _log(f"Starting training for '{ablation_name}'...")
        t0     = time.monotonic()
        result = subprocess.run(cmd, check=False, env=env)
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            _log(f"[WARNING] '{ablation_name}' exited with code {result.returncode} "
                 f"({elapsed/60:.1f} min)")
        else:
            _log(f"Finished '{ablation_name}' in {elapsed/60:.1f} min")

        # Read best metrics
        best = _best_row_from_csv(csv_path)
        if best:
            _log(f"Best val  —  "
                 f"MR→CT SSIM: {best.get('mr2ct_ssim', 'N/A')}  "
                 f"CT→MR SSIM: {best.get('ct2mr_ssim', 'N/A')}  "
                 f"Mean Dice: {best.get('mean_dice', 'N/A')}  "
                 f"(epoch {best.get('epoch', 'N/A')})")
        else:
            _log(f"No CSV results found at {csv_path}")

        summary_rows.append((ablation_name, best))

        # Remaining time estimate
        configs_done      = idx
        configs_remaining = n_total - configs_done
        if configs_done > 0 and configs_remaining > 0:
            avg_min = (time.monotonic() - job_start) / 60 / configs_done
            eta_min = avg_min * configs_remaining
            _log(f"Progress: {configs_done}/{n_total} done — "
                 f"~{eta_min:.0f} min remaining "
                 f"(avg {avg_min:.1f} min/config)")

        # Clean up temp config
        try:
            os.unlink(tmp_cfg_path)
        except OSError:
            pass

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - job_start
    _banner("Ablation study complete")
    _log(f"Total wall time: {total_elapsed/60:.1f} min")

    _print_summary(anatomy, output_dir_p, summary_rows)
    _print_markdown_table(anatomy, output_dir_p, summary_rows, logs_dir)


# ---------------------------------------------------------------------------
# Markdown results table generator
# ---------------------------------------------------------------------------

def _print_markdown_table(
    anatomy:    str,
    output_dir: Path,
    metrics:    List[Tuple[str, Optional[Dict[str, str]]]],
    logs_dir:   Path,
) -> None:
    """Print a Markdown results table and save it as a CSV summary."""
    MD_COLS = [
        ("mr2ct_ssim", "MRI→CT SSIM",     ".4f"),
        ("mr2ct_mae",  "MRI→CT MAE (HU)", ".1f"),
        ("ct2mr_ssim", "CT→MR SSIM",      ".4f"),
        ("mean_dice",  "Mean Dice",        ".4f"),
        ("epoch",      "Best Epoch",       "d"),
    ]
    _HU_NORM_SCALE = 2000.0

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
        if key == "mr2ct_mae":
            val = val * _HU_NORM_SCALE
        if fmt == "d":
            return str(int(round(val)))
        return format(val, fmt)

    headers = ["Ablation"] + [h for _, h, _ in MD_COLS]
    col_w   = [max(len(h), 18) for h in headers]

    def _row_str(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " |"

    sep_row = "|" + "|".join("-" * (w + 2) for w in col_w) + "|"

    lines = [
        "",
        f"## Ablation results — {anatomy.upper()}",
        "",
        _row_str(headers),
        sep_row,
    ]
    for ablation_name, row in metrics:
        cells = [ablation_name] + [_fmt_cell(row, key, fmt) for key, _, fmt in MD_COLS]
        lines.append(_row_str(cells))
    lines.append("")

    md_table = "\n".join(lines)
    print(md_table, flush=True)

    # Save CSV summary
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

    _log(f"Ablation summary saved to: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 3 MT-CycleGAN ablation configurations sequentially."
    )
    p.add_argument("--anatomy",     required=True,
                   choices=["head_neck", "thorax", "abdomen"])
    p.add_argument("--data_root",   required=True)
    p.add_argument("--split_dir",   required=True)
    p.add_argument("--base_config",
                   default=str(Path(__file__).resolve().parents[1] / "configs" / "base_config.json"))
    p.add_argument("--output_dir",
                   default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--train_script", default=None)
    p.add_argument("--epochs",      type=int, default=None,
                   help="Override EPOCHS for all ablation runs (e.g. --epochs 50).")
    p.add_argument("--device",      default=None,
                   help="CUDA_VISIBLE_DEVICES for subprocesses (e.g. --device 0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _log("=== MT-CycleGAN Ablation Runner starting ===")
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
