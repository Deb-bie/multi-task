#!/usr/bin/env bash
# ==============================================================================
# scripts/run_totalsegmentator.sh
#
# Batch-run TotalSegmentator on all CT volumes for one anatomical region and
# copy the relevant organ labels into a unified 6-class naming convention.
#
# Environment variables (required unless defaults are acceptable)
# ──────────────────────────────────────────────────────────────
#   DATA_ROOT    Root of the SynthRAD2025 dataset tree.
#                Expected layout: DATA_ROOT/{ANATOMY}/{patient_id}/ct.nii.gz
#   ANATOMY      One of: head_neck | thorax | abdomen
#   OUTPUT_DIR   Destination root for segmentation outputs.
#                Labels are saved to: OUTPUT_DIR/{ANATOMY}/{patient_id}/seg/
#
# Optional
# ────────
#   TOTALSEG_BIN Path to the TotalSegmentator binary (default: TotalSegmentator
#                on PATH — installed via `pip install TotalSegmentator`).
#   DRY_RUN      If set to "1", print commands without executing them.
#   N_JOBS       Parallelism (default 1 — run serially to stay within VRAM).
#
# Usage example
# ─────────────
#   export DATA_ROOT=/data/synthrad2025
#   export ANATOMY=abdomen
#   export OUTPUT_DIR=/data/synthrad2025_segs
#   bash scripts/run_totalsegmentator.sh
#
# Label mapping
# ─────────────
# TotalSegmentator uses task-specific label names.  This script maps them to
# the project's 6-class convention (1-indexed, background=0 implicit):
#
#   head_neck:
#     1 brainstem      ← brainstem.nii.gz
#     2 parotid_L      ← parotid_gland_left.nii.gz
#     3 parotid_R      ← parotid_gland_right.nii.gz
#     4 mandible       ← mandible.nii.gz
#     5 spinal_cord    ← spinal_cord.nii.gz
#
#   thorax:
#     1 lung_L         ← lung_left.nii.gz
#     2 lung_R         ← lung_right.nii.gz
#     3 heart          ← heart.nii.gz
#     4 spinal_cord    ← spinal_cord.nii.gz
#     5 esophagus      ← esophagus.nii.gz
#
#   abdomen:
#     1 liver          ← liver.nii.gz
#     2 kidney_L       ← kidney_left.nii.gz
#     3 kidney_R       ← kidney_right.nii.gz
#     4 spleen         ← spleen.nii.gz
#     5 pancreas       ← pancreas.nii.gz
# ==============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ANATOMY="${ANATOMY:?'ANATOMY must be set to head_neck, thorax, or abdomen'}"
OUTPUT_DIR="${OUTPUT_DIR:?'OUTPUT_DIR must be set to the segmentation output root'}"
TOTALSEG_BIN="${TOTALSEG_BIN:-TotalSegmentator}"
DRY_RUN="${DRY_RUN:-0}"
N_JOBS="${N_JOBS:-1}"

# DATA_DIR can be set directly (e.g. /data/Task1/Task1/HN) to avoid needing
# the directory to be named after the anatomy label.
# If not set, falls back to DATA_ROOT/ANATOMY (original behaviour).
DATA_DIR="${DATA_DIR:-}"
DATA_ROOT="${DATA_ROOT:-}"

LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/totalsegmentator_${ANATOMY}.log"

mkdir -p "${LOG_DIR}"

# ── Validate anatomy ──────────────────────────────────────────────────────────
case "${ANATOMY}" in
    head_neck|thorax|abdomen) ;;
    *)
        echo "[ERROR] Unknown anatomy '${ANATOMY}'. Must be one of: head_neck thorax abdomen" >&2
        exit 1
        ;;
esac

# ── Label mapping per anatomy ─────────────────────────────────────────────────
# Associative array: project_name → TotalSegmentator filename stem
declare -A LABEL_MAP

case "${ANATOMY}" in
    head_neck)
        LABEL_MAP=(
            [brainstem]=brainstem
            [parotid_L]=parotid_gland_left
            [parotid_R]=parotid_gland_right
            [mandible]=mandible
            [spinal_cord]=spinal_cord
        )
        TS_TASK="total"   # uses the 'total' task which includes HN structures
        ;;
    thorax)
        LABEL_MAP=(
            [lung_L]=lung_left
            [lung_R]=lung_right
            [heart]=heart
            [spinal_cord]=spinal_cord
            [esophagus]=esophagus
        )
        TS_TASK="total"
        ;;
    abdomen)
        LABEL_MAP=(
            [liver]=liver
            [kidney_L]=kidney_left
            [kidney_R]=kidney_right
            [spleen]=spleen
            [pancreas]=pancreas
        )
        TS_TASK="total"
        ;;
esac

# ── Logging helper ────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
# Resolve anatomy data directory: DATA_DIR takes precedence over DATA_ROOT/ANATOMY
if [[ -n "${DATA_DIR}" ]]; then
    ANAT_DATA_DIR="${DATA_DIR}"
elif [[ -n "${DATA_ROOT}" ]]; then
    ANAT_DATA_DIR="${DATA_ROOT}/${ANATOMY}"
else
    echo "[ERROR] Set either DATA_DIR (direct path to anatomy folder) or DATA_ROOT" >&2
    exit 1
fi

if [[ ! -d "${ANAT_DATA_DIR}" ]]; then
    log "[ERROR] Anatomy directory not found: ${ANAT_DATA_DIR}"
    exit 1
fi

# Collect all patient CT paths — SynthRAD2025 uses .mha format
mapfile -t CT_PATHS < <(find "${ANAT_DATA_DIR}" -name "ct.mha" | sort)

if [[ ${#CT_PATHS[@]} -eq 0 ]]; then
    log "[ERROR] No ct.mha files found under ${ANAT_DATA_DIR}"
    exit 1
fi

log "Found ${#CT_PATHS[@]} CT volumes for anatomy=${ANATOMY}"
log "Output root : ${OUTPUT_DIR}"
log "Log file    : ${LOG_FILE}"
log "DRY_RUN     : ${DRY_RUN}"

n_ok=0
n_fail=0

for CT_PATH in "${CT_PATHS[@]}"; do
    # Derive patient ID from directory name
    PATIENT_DIR="$(dirname "${CT_PATH}")"
    PATIENT_ID="$(basename "${PATIENT_DIR}")"

    SEG_OUT_DIR="${OUTPUT_DIR}/${ANATOMY}/${PATIENT_ID}/seg_raw"
    FINAL_SEG_DIR="${OUTPUT_DIR}/${ANATOMY}/${PATIENT_ID}/seg"

    log "────────────────────────────────────────"
    log "Patient : ${PATIENT_ID}"
    log "CT path : ${CT_PATH}"
    log "Seg out : ${SEG_OUT_DIR}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "[DRY-RUN] ${TOTALSEG_BIN} -i ${CT_PATH} -o ${SEG_OUT_DIR} --fast --task ${TS_TASK}"
        continue
    fi

    mkdir -p "${SEG_OUT_DIR}" "${FINAL_SEG_DIR}"

    # ── Run TotalSegmentator ───────────────────────────────────────────────
    if "${TOTALSEG_BIN}" \
            -i  "${CT_PATH}" \
            -o  "${SEG_OUT_DIR}" \
            --fast \
            --task "${TS_TASK}" \
            2>> "${LOG_FILE}"; then

        log "[OK] TotalSegmentator completed for ${PATIENT_ID}"

        # ── Copy and rename relevant labels ───────────────────────────────
        copy_ok=1
        for PROJECT_NAME in "${!LABEL_MAP[@]}"; do
            TS_STEM="${LABEL_MAP[${PROJECT_NAME}]}"
            SRC_FILE="${SEG_OUT_DIR}/${TS_STEM}.nii.gz"
            DST_FILE="${FINAL_SEG_DIR}/${PROJECT_NAME}.nii.gz"

            if [[ -f "${SRC_FILE}" ]]; then
                cp "${SRC_FILE}" "${DST_FILE}"
                log "  Copied: ${TS_STEM}.nii.gz → ${PROJECT_NAME}.nii.gz"
            else
                log "  [WARN] Label not found: ${SRC_FILE}"
                copy_ok=0
            fi
        done

        if [[ "${copy_ok}" -eq 1 ]]; then
            log "[OK] All labels copied for ${PATIENT_ID}"
            (( n_ok++ )) || true
        else
            log "[WARN] Some labels missing for ${PATIENT_ID}"
            (( n_fail++ )) || true
        fi

    else
        log "[FAIL] TotalSegmentator failed for ${PATIENT_ID} (exit code $?)"
        (( n_fail++ )) || true
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
log "════════════════════════════════════════"
log "Summary: ${n_ok} OK | ${n_fail} FAILED | total ${#CT_PATHS[@]}"
log "Full log: ${LOG_FILE}"

if [[ "${n_fail}" -gt 0 ]]; then
    exit 1
fi
