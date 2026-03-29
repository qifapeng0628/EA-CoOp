#!/usr/bin/env bash
set -u
set -o pipefail

# Usage:
#   bash scripts/gridsearch_gaze_lambdas.sh <DATA_ROOT> <DATASET> <MODEL>
#
# Example:
#   bash scripts/gridsearch_gaze_lambdas.sh data btmri BiomedCoOp_BiomedCLIP
#
# Notes:
# - This script will iterate:
#     GAZE_ROI_CE_LAMBDA in {0.00..1.00 step 0.05}
#     GAZE_CONS_LAMBDA   in {0.00..1.00 step 0.05}
# - After each run, it parses logs and then deletes output dirs to allow the next run.

DATA_ROOT="${1:?Need DATA_ROOT}"
DATASET="${2:?Need DATASET}"
MODEL="${3:?Need MODEL}"

# -------- You can override these by env vars --------
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/biomedcoop/base2new_gaze_consistency_v4.sh}"

# These 3 define the output directories you told me must be deleted:
SHOTS="${SHOTS:-16}"
TRAINER_TAG="${TRAINER_TAG:-BiomedCoOp_BiomedCLIP}"
RUN_NAME="${RUN_NAME:-nctx4_cscFalse_ctpend}"

# Result logging
RESULTS_DIR="${RESULTS_DIR:-./gridsearch_results}"
mkdir -p "${RESULTS_DIR}"
CSV="${RESULTS_DIR}/grid_${DATASET}_shots${SHOTS}_${TRAINER_TAG}_${RUN_NAME}.csv"
BEST_TXT="${RESULTS_DIR}/best_${DATASET}_shots${SHOTS}_${TRAINER_TAG}_${RUN_NAME}.txt"

# Output dirs to parse/delete (as you specified pattern)
TRAIN_DIR="output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER_TAG}/${RUN_NAME}"
NEW_DIR="output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER_TAG}/${RUN_NAME}"

# Init CSV
if [[ ! -f "${CSV}" ]]; then
  echo "roi_ce_lambda,cons_lambda,base_metric,new_metric,hm,status" > "${CSV}"
fi

# Generate grid values: 0.00, 0.05, ..., 1.00
GRID_VALS=()
while IFS= read -r v; do GRID_VALS+=("$v"); done < <(python - <<'PY'
vals=[i/20 for i in range(0,21)]  # 0..1 step 0.05
for v in vals:
    print(f"{v:.2f}")
PY
)

already_done () {
  local roi="$1"
  local cons="$2"
  # exact match on first 2 columns
  grep -q "^${roi},${cons}," "${CSV}"
}

cleanup_outputs () {
  rm -rf "${TRAIN_DIR}" "${NEW_DIR}"
}

parse_metric () {
  # Print ONE float metric extracted from parse_test_res.py output
  # We try to capture patterns like "accuracy: 83.21", "top-1: 76.5", etc.
  local d="$1"
  local out
  out="$(python parse_test_res.py --directory "${d}" --test-log 2>&1 || true)"
  python - <<PY
import re, sys
text = """${out}"""

patterns = [
    r'(?:top[- ]?1|top1|accuracy|acc|mean_acc|macc|auc)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
    r'([0-9]+(?:\.[0-9]+)?)\s*%',
]
for pat in patterns:
    m = re.findall(pat, text, flags=re.I)
    if m:
        print(m[-1])
        sys.exit(0)

# fallback: last float in output (best-effort)
nums = re.findall(r'([0-9]+(?:\.[0-9]+)?)', text)
print(nums[-1] if nums else "nan")
PY
}

compute_hm () {
  local a="$1"
  local b="$2"
  python - <<PY
import math
a=float("${a}") if "${a}"!="nan" else float("nan")
b=float("${b}") if "${b}"!="nan" else float("nan")
if not (math.isfinite(a) and math.isfinite(b)) or (a+b)<=0:
    print("nan")
else:
    print(2*a*b/(a+b))
PY
}

update_best () {
  python - <<PY
import csv, math
csv_path="${CSV}"
best=None
with open(csv_path, newline="") as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            hm=float(row["hm"])
        except Exception:
            continue
        if not math.isfinite(hm):
            continue
        if best is None or hm>best[0]:
            best=(hm,row)
if best is None:
    print("No valid runs yet.")
else:
    hm,row=best
    print(f'BEST: HM={hm:.6f} roi_ce_lambda={row["roi_ce_lambda"]} cons_lambda={row["cons_lambda"]} '
          f'base={row["base_metric"]} new={row["new_metric"]} status={row["status"]}')
PY
}

echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] TRAIN_DIR=${TRAIN_DIR}"
echo "[INFO] NEW_DIR=${NEW_DIR}"
echo "[INFO] CSV=${CSV}"
echo

# Main loop
for roi in "${GRID_VALS[@]}"; do
  for cons in "${GRID_VALS[@]}"; do

    if already_done "${roi}" "${cons}"; then
      echo "[SKIP] roi=${roi} cons=${cons} (already in CSV)"
      continue
    fi

    echo "============================================================"
    echo "[RUN] roi=${roi} cons=${cons}"
    echo "============================================================"

    # Ensure clean start (your requirement)
    cleanup_outputs

    # Export lambdas for the training script.
    # Your base2new_gaze_consistency_v4.sh reads these env vars (or defaults). :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
    export GAZE_ROI_CE_LAMBDA="${roi}"
    export GAZE_CONS_LAMBDA="${cons}"

    # Per-run log
    RUN_TAG="roi${roi}_cons${cons}"
    TRAIN_LOG="${RESULTS_DIR}/train_${DATASET}_${RUN_TAG}.log"
    PARSE_BASE_LOG="${RESULTS_DIR}/parse_base_${DATASET}_${RUN_TAG}.log"
    PARSE_NEW_LOG="${RESULTS_DIR}/parse_new_${DATASET}_${RUN_TAG}.log"

    # Train
    set +e
    bash "${TRAIN_SCRIPT}" "${DATA_ROOT}" "${DATASET}" "${MODEL}" 2>&1 | tee "${TRAIN_LOG}"
    rc_train=${PIPESTATUS[0]}
    set -e

    if [[ ${rc_train} -ne 0 ]]; then
      echo "[FAIL] training rc=${rc_train}"
      echo "${roi},${cons},nan,nan,nan,TRAIN_FAIL" >> "${CSV}"
      cleanup_outputs
      continue
    fi

    # Check output dirs exist
    if [[ ! -d "${TRAIN_DIR}" || ! -d "${NEW_DIR}" ]]; then
      echo "[FAIL] expected output dirs not found"
      echo "  TRAIN_DIR=${TRAIN_DIR}"
      echo "  NEW_DIR=${NEW_DIR}"
      echo "${roi},${cons},nan,nan,nan,NO_OUTPUT_DIR" >> "${CSV}"
      cleanup_outputs
      continue
    fi

    # Parse metrics (save raw parse logs too)
    set +e
    python parse_test_res.py --directory "${TRAIN_DIR}" --test-log 2>&1 | tee "${PARSE_BASE_LOG}"
    rc_base=${PIPESTATUS[0]}
    python parse_test_res.py --directory "${NEW_DIR}" --test-log 2>&1 | tee "${PARSE_NEW_LOG}"
    rc_new=${PIPESTATUS[0]}
    set -e

    if [[ ${rc_base} -ne 0 || ${rc_new} -ne 0 ]]; then
      echo "[FAIL] parse failed base_rc=${rc_base} new_rc=${rc_new}"
      echo "${roi},${cons},nan,nan,nan,PARSE_FAIL" >> "${CSV}"
      cleanup_outputs
      continue
    fi

    base_metric="$(parse_metric "${TRAIN_DIR}")"
    new_metric="$(parse_metric "${NEW_DIR}")"
    hm="$(compute_hm "${base_metric}" "${new_metric}")"

    echo "[METRIC] base=${base_metric} new=${new_metric} HM=${hm}"
    echo "${roi},${cons},${base_metric},${new_metric},${hm},OK" >> "${CSV}"

    # Clean for next run (your requirement)
    cleanup_outputs

    # Print best so far
    update_best | tee "${BEST_TXT}"

  done
done

echo
echo "[DONE] Grid search finished."
echo "[DONE] CSV saved to: ${CSV}"
echo "[DONE] Best summary saved to: ${BEST_TXT}"
echo "[DONE] Final best:"
update_best
