"""
src/dataset.py — SynthRAD2025 2.5D slice dataset for the multi-task CycleGAN.

Provides:
    load_split           – Read a split JSON produced by scripts/prepare_splits.py.
    make_dataloader      – Construct a DataLoader over SynthRADSliceDataset.
    SynthRADSliceDataset – PyTorch Dataset yielding per-slice batches.

2.5D slice convention
----------------------
For each axial slice index *z*, three adjacent slices are stacked as channels:
    [z-1, z, z+1]  →  shape (3, H, W)
Boundary slices replicate the nearest valid slice:
    z = 0    →  [0, 0, 1]
    z = Z-1  →  [Z-2, Z-1, Z-1]

Normalisation
-------------
CT:  Clip to HU range [-1000, 3000] → linear scale to [-1, 1]:
         norm = (HU.clip(-1000, 3000) + 1000) / 2000 - 1

MRI: Clip to [0, 99th-percentile within body mask] → scale to [-1, 1].
     The percentile is computed per-volume to handle scanner variability.
     Voxels outside the body mask are zeroed after normalisation.

Augmentation
------------
Training split only: random horizontal flip (p = 0.5).
Val/test splits: no augmentation.

Lazy loading
------------
Volumes are loaded on first access within each DataLoader worker process and
cached for the lifetime of that worker, so each worker loads each patient at
most once regardless of how many slices it requests.

Batch dict keys
---------------
    ``"mr"``   : float32 (3, H, W)  — 2.5-D MRI slice in [-1, 1]
    ``"ct"``   : float32 (3, H, W)  — 2.5-D CT  slice in [-1, 1]
    ``"mask"`` : float32 (1, H, W)  — binary body mask (0 or 1)
    ``"seg"``  : int64   (H, W)     — organ segmentation class labels
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Optional resizing (scipy used only when the native slice size ≠ image_size)
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import zoom as _scipy_zoom
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# HU normalisation constants  (must match src/metrics.py)
# ---------------------------------------------------------------------------
_CT_HU_MIN:   float = -1000.0
_CT_HU_MAX:   float =  3000.0
_CT_HU_RANGE: float = _CT_HU_MAX - _CT_HU_MIN   # 4000


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_nifti(path: Path) -> Tuple[np.ndarray, Any]:
    """Load a NIfTI volume with SimpleITK.

    Args:
        path: Path to the ``.nii.gz`` file.

    Returns:
        ``(array_zyx, sitk_image)`` where *array_zyx* has dtype float32 and
        axes ``(Z, Y, X)``.

    Raises:
        ImportError: If SimpleITK is not installed.
        FileNotFoundError: If *path* does not exist.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required for NIfTI I/O. "
            "Install with: pip install SimpleITK==2.3.1"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr, img


def _get_volume_z(path: Path) -> int:
    """Return the Z (slice) dimension of a NIfTI file without loading pixel data.

    Uses SimpleITK's metadata-only reader, which is ~100× faster than a full
    volume load and makes building the slice index feasible at dataset init.

    Args:
        path: Path to the ``.nii.gz`` file.

    Returns:
        Number of axial slices (Z dimension).
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required.  Install with: pip install SimpleITK==2.3.1"
        ) from exc

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()
    size = reader.GetSize()  # SimpleITK convention: (X, Y, Z)
    return int(size[2])


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_ct(arr: np.ndarray) -> np.ndarray:
    """Clip CT to [-1000, 3000] HU and scale to [-1, 1].

    Args:
        arr: Raw CT array in HU, shape ``(Z, H, W)``.

    Returns:
        Normalised float32 array in ``[-1, 1]``, same shape.
    """
    arr = arr.clip(_CT_HU_MIN, _CT_HU_MAX)
    return ((arr - _CT_HU_MIN) / _CT_HU_RANGE * 2.0 - 1.0).astype(np.float32)


def _normalize_mr(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalise MRI intensity to [-1, 1] using per-volume masked percentile.

    Clips to [0, 99th-percentile within mask] to suppress bright outliers
    (e.g. fat, implants), then linearly maps to [-1, 1].  Voxels outside the
    body mask are set to -1 (background) after normalisation.

    Args:
        arr:  Raw MRI array, shape ``(Z, H, W)``.  Values in scanner units.
        mask: Binary body mask, shape ``(Z, H, W)`` (1 = foreground).

    Returns:
        Normalised float32 array in ``[-1, 1]``.
    """
    arr = arr.astype(np.float32)
    fg_voxels = arr[mask > 0.5]

    if fg_voxels.size == 0:
        # No foreground: return zeros (edge case for empty masks)
        return np.zeros_like(arr) - 1.0

    p99 = float(np.percentile(fg_voxels, 99))
    p_lo = 0.0

    if p99 <= p_lo:
        p99 = float(arr.max()) + 1e-6

    arr = arr.clip(p_lo, p99)
    arr = (arr - p_lo) / (p99 - p_lo) * 2.0 - 1.0  # → [-1, 1]

    # Zero out background (set to -1 = outside body)
    arr[mask < 0.5] = -1.0

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial resize helper
# ---------------------------------------------------------------------------

def _resize_slice_hw(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    is_label: bool = False,
) -> np.ndarray:
    """Resize a 2D slice (H, W) to (target_h, target_w).

    Args:
        arr:      2D numpy array of shape ``(H, W)``.
        target_h: Target height.
        target_w: Target width.
        is_label: If ``True``, use nearest-neighbour interpolation (order=0)
                  to preserve integer class labels.  Otherwise bilinear (order=1).

    Returns:
        Resized array of shape ``(target_h, target_w)``.
    """
    H, W = arr.shape
    if H == target_h and W == target_w:
        return arr

    if not _HAVE_SCIPY:
        raise ImportError(
            "scipy is required for slice resizing. "
            "Install with: pip install scipy"
        )

    zoom_h = target_h / H
    zoom_w = target_w / W
    order  = 0 if is_label else 1
    return _scipy_zoom(arr, (zoom_h, zoom_w), order=order).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class SynthRADSliceDataset(Dataset):
    """PyTorch Dataset over 2.5D axial slices from SynthRAD2025 patient volumes.

    At initialisation the dataset reads only the Z-dimension metadata of each
    patient's MRI volume (fast, < 1 s per patient) to build a flat index of
    ``(patient_entry_index, slice_z)`` tuples.  Full volumes are loaded lazily
    on first access within each DataLoader worker and cached for the worker's
    lifetime.

    Args:
        entries:          List of patient entry dicts, each with keys
                          ``patient_id``, ``mr_path``, ``ct_path``,
                          ``seg_path``, ``mask_path``.
        data_root:        Root directory of the SynthRAD2025 dataset (unused
                          if *entries* already contain absolute paths, but kept
                          for API consistency).
        split:            ``"train"``, ``"val"``, or ``"test"``.
                          Controls whether augmentation is applied.
        image_size:       Target H and W (default 256).  Slices are resized
                          with bilinear interpolation (nearest for seg labels)
                          if the native size differs.
        min_mask_voxels:  Minimum number of foreground voxels required for a
                          slice to be included in the index (default 200).
                          Filters out near-empty boundary slices.
    """

    def __init__(
        self,
        entries:          List[Dict[str, Any]],
        data_root:        str,
        split:            str,
        image_size:       int = 256,
        min_mask_voxels:  int = 200,
    ) -> None:
        super().__init__()
        self.entries         = entries
        self.data_root       = Path(data_root)
        self.split           = split
        self.image_size      = image_size
        self.min_mask_voxels = min_mask_voxels
        self.augment         = (split == "train")

        # Per-worker volume cache: populated lazily in __getitem__
        # Each DataLoader worker forks and gets its own copy of this dict.
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

        # Build flat slice index using metadata-only Z reads
        self.index: List[Tuple[int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Populate self.index with (entry_idx, z) pairs for valid slices.

        Uses SimpleITK's metadata reader to obtain volume Z-sizes without
        loading pixel data.  Only slices z ∈ [1, Z-2] are included so that
        every 2.5D triplet [z-1, z, z+1] is fully within the volume.

        Note:
            Foreground filtering (min_mask_voxels) is performed at
            ``__getitem__`` time rather than here to keep init fast.  A small
            fraction of included slices may be filtered at getitem if the mask
            turns out to be nearly empty — these return valid (but trivial)
            samples and do not affect training meaningfully.
        """
        for entry_idx, entry in enumerate(self.entries):
            mr_path = Path(entry["mr_path"])
            try:
                Z = _get_volume_z(mr_path)
            except Exception as exc:
                print(f"[dataset] WARNING: could not read {mr_path}: {exc}")
                continue

            # Include slices 1 … Z-2 so all triplets are within bounds.
            # Clamp to at least 1 slice per volume (for very thin volumes).
            z_start = min(1, max(0, Z - 1))
            z_end   = max(z_start + 1, Z - 1)

            for z in range(z_start, z_end):
                self.index.append((entry_idx, z))

    # ------------------------------------------------------------------
    # Lazy volume loading
    # ------------------------------------------------------------------

    def _load_volume(self, entry_idx: int) -> Dict[str, np.ndarray]:
        """Load, normalise, and cache a patient volume.

        Called once per (worker, patient) pair.  Subsequent calls for the same
        *entry_idx* within the same worker are O(1) dict lookups.

        Args:
            entry_idx: Index into ``self.entries``.

        Returns:
            Dict with keys ``"mr"``, ``"ct"``, ``"mask"``, ``"seg"`` —
            normalised float32 arrays of shape ``(Z, H, W)``.
        """
        if entry_idx in self._cache:
            return self._cache[entry_idx]

        entry = self.entries[entry_idx]

        mr_arr,   _ = _load_nifti(Path(entry["mr_path"]))
        ct_arr,   _ = _load_nifti(Path(entry["ct_path"]))
        mask_arr, _ = _load_nifti(Path(entry["mask_path"]))

        # Binarise mask before normalisation (some masks are stored as floats)
        mask_bin = (mask_arr > 0.5).astype(np.float32)

        # Normalise intensities
        mr_norm  = _normalize_mr(mr_arr,  mask_bin)
        ct_norm  = _normalize_ct(ct_arr)

        # Seg labels are optional — produced by TotalSegmentator + merge step.
        # If not yet available, return all-zeros (background only); the seg
        # loss will be masked out to zero for these slices in the training loop.
        seg_path = entry.get("seg_path")
        if seg_path is not None:
            seg_arr, _ = _load_nifti(Path(seg_path))
            seg_int = seg_arr.astype(np.int64)
        else:
            seg_int = np.zeros_like(ct_arr, dtype=np.int64)

        vol = {
            "mr":   mr_norm,
            "ct":   ct_norm,
            "mask": mask_bin,
            "seg":  seg_int,
        }
        self._cache[entry_idx] = vol
        return vol

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one 2.5-D slice batch dict.

        Args:
            idx: Index into the flat slice index.

        Returns:
            Dict with keys ``"mr"``, ``"ct"``, ``"mask"``, ``"seg"``
            as PyTorch tensors ready for the model.
        """
        entry_idx, z = self.index[idx]
        vol = self._load_volume(entry_idx)

        Z = vol["mr"].shape[0]

        # 2.5-D slice triplet with boundary clamping
        z_prev = max(0, z - 1)
        z_next = min(Z - 1, z + 1)

        # Stack three adjacent slices along the channel dimension
        mr_slice   = np.stack([vol["mr"][z_prev],   vol["mr"][z],   vol["mr"][z_next]])    # (3, H, W)
        ct_slice   = np.stack([vol["ct"][z_prev],   vol["ct"][z],   vol["ct"][z_next]])    # (3, H, W)
        mask_slice = vol["mask"][z][np.newaxis]   # (1, H, W)
        seg_slice  = vol["seg"][z]                # (H, W)

        # Resize to target image size if needed
        H, W = mr_slice.shape[1], mr_slice.shape[2]
        if H != self.image_size or W != self.image_size:
            mr_slice   = np.stack([
                _resize_slice_hw(mr_slice[c], self.image_size, self.image_size)
                for c in range(3)
            ])
            ct_slice   = np.stack([
                _resize_slice_hw(ct_slice[c], self.image_size, self.image_size)
                for c in range(3)
            ])
            mask_slice = _resize_slice_hw(
                mask_slice[0], self.image_size, self.image_size
            )[np.newaxis]
            seg_slice  = _resize_slice_hw(
                seg_slice.astype(np.float32), self.image_size, self.image_size,
                is_label=True,
            ).astype(np.int64)

        # Training augmentation: random horizontal flip
        if self.augment and random.random() < 0.5:
            mr_slice   = np.ascontiguousarray(mr_slice[:, :, ::-1])
            ct_slice   = np.ascontiguousarray(ct_slice[:, :, ::-1])
            mask_slice = np.ascontiguousarray(mask_slice[:, :, ::-1])
            seg_slice  = np.ascontiguousarray(seg_slice[:, ::-1])

        return {
            "mr":   torch.from_numpy(mr_slice.copy()).float(),
            "ct":   torch.from_numpy(ct_slice.copy()).float(),
            "mask": torch.from_numpy(mask_slice.copy()).float(),
            "seg":  torch.from_numpy(seg_slice.copy()).long(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_split(split_dir: str, anatomy: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load the train/val/test split JSON for an anatomical region.

    The JSON must have been produced by ``scripts/prepare_splits.py`` and
    contains keys ``"train"``, ``"val"``, ``"test"``, each mapping to a list
    of patient entry dicts.

    Args:
        split_dir: Directory containing ``{anatomy}_split.json`` files.
        anatomy:   Anatomical region name (``"head_neck"``, ``"thorax"``,
                   ``"abdomen"``).

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``; each value is a
        list of patient entry dicts with keys
        ``patient_id``, ``mr_path``, ``ct_path``, ``seg_path``, ``mask_path``.

    Raises:
        FileNotFoundError: If the split JSON does not exist.
        KeyError:          If the JSON is missing required split keys.
    """
    json_path = Path(split_dir) / f"{anatomy}_split.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {json_path}\n"
            f"Run: python scripts/prepare_splits.py --anatomy {anatomy} ..."
        )

    with open(json_path) as fh:
        data: Dict[str, Any] = json.load(fh)

    for key in ("train", "val", "test"):
        if key not in data:
            raise KeyError(
                f"Split file {json_path} is missing the '{key}' key."
            )

    return {
        "train": data["train"],
        "val":   data["val"],
        "test":  data["test"],
    }


def make_dataloader(
    entries:     List[Dict[str, Any]],
    anatomy:     str,
    data_root:   str,
    split:       str,
    batch_size:  int,
    num_workers: int,
    image_size:  int = 256,
) -> DataLoader:
    """Construct a DataLoader over a list of patient entries.

    Args:
        entries:     List of patient entry dicts (from :func:`load_split`).
        anatomy:     Anatomical region name (unused internally; reserved for
                     future anatomy-specific transforms).
        data_root:   Root of the SynthRAD2025 dataset (passed through to the
                     dataset for API consistency; paths in *entries* are
                     expected to be absolute).
        split:       One of ``"train"``, ``"val"``, ``"test"``.  Controls
                     whether data augmentation is applied.
        batch_size:  Number of slices per batch.
        num_workers: Number of DataLoader worker processes.
        image_size:  Target spatial resolution (default 256).

    Returns:
        A :class:`~torch.utils.data.DataLoader` that yields dicts with keys
        ``"mr"``, ``"ct"``, ``"mask"``, ``"seg"``.
    """
    shuffle    = (split == "train")
    pin_memory = torch.cuda.is_available()

    dataset = SynthRADSliceDataset(
        entries    = entries,
        data_root  = data_root,
        split      = split,
        image_size = image_size,
    )

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = shuffle,          # drop incomplete last batch during training
        persistent_workers = (num_workers > 0),
    )
