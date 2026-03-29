#!/usr/bin/env bash
set -u
set -o pipefail

# Usage:
#   bash scripts/gridsearch_fewshot_gaze_lambdas.sh <DATA_ROOT> <DATASET> <MODEL>
# Example:
#   bash scripts/gridsearch_fewshot_gaze_lambdas.sh data btmri BiomedCLIP
#
# What it does:
# - Grid search over:
#     GAZE_ROI_CE_LAMBDA in [0,1] step 0.05
#     GAZE_CONS_LAMBDA   in [0,1] step 0.05
# - For each run:
#     1) delete output dir (required by you)
#     2) train
#     3) parse_test_res.py
#     4) write CSV
#     5) delete output dir again

DATA_ROOT="${1:?Need DATA_ROOT, e.g. data}"
DATASET="${2:?Need DATASET, e.g. btmri}"
MODEL="${3:?Need MODEL, e.g. BiomedCLIP}"

# ---- You can override these by exporting env vars before running ----
SHOTS="${SHOTS:-16}"

# Your training wrapper (the one that exports gaze envs and then calls scripts/biomedcoop/few_shot.sh)
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/biomedcoop/few_shot_gaze_consistency_v4.sh}"

# This must match YOUR few-shot output naming convention
# If your few_shot.sh uses NCTX=4, CSC=False, CTP=end, then:
RUN_NAME="${RUN_NAME:-nctx4_cscFalse_ctpend}"

# Trainer folder name used in output path
# In your codebase, trainer name usually is BiomedCoOp_${MODEL} => BiomedCoOp_BiomedCLIP
TRAINER_TAG="${TRAINER_TAG:-BiomedCoOp_${MODEL}}"

# Output dir that parse_test_res.py will read, and that you require deleting
OUT_DIR="output/${DATASET}/shots_${SHOTS}/${TRAINER_TAG}/${RUN_NAME}"

# Result logging
RESULTS_DIR="${RESULTS_DIR:-./gridsearch_results}"
mkdir -p "${RESULTS_DIR}"
CSV="${RESULTS_DIR}/grid_fewshot_${DATASET}_shots${SHOTS}_${TRAINER_TAG}_${RUN_NAME}.csv"
BEST_TXT="${RESULTS_DIR}/best_fewshot_${DATASET}_shots${SHOTS}_${TRAINER_TAG}_${RUN_NAME}.txt"

# Init CSV
if [[ ! -f "${CSV}" ]]; then
  echo "roi_ce_lambda,cons_lambda,metric,status" > "${CSV}"
fi

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
  grep -q "^${roi},${cons}," "${CSV}"
}

cleanup_outputs () {
  rm -rf "${OUT_DIR}"
}

parse_metric_from_log () {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re, sys, pathlib, math
p = pathlib.Path(sys.argv[1])
txt = p.read_text(errors="ignore") if p.exists() else ""

# Try common keys (keep broad to tolerate different formats)
patterns = [
    r'(?:top[- ]?1|top1|accuracy|acc|mean_acc|macc|auc)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
    r'([0-9]+(?:\.[0-9]+)?)\s*%',  # e.g. "83.21%"
]
for pat in patterns:
    m = re.findall(pat, txt, flags=re.I)
    if m:
        print(m[-1])
        sys.exit(0)

# fallback: last float
nums = re.findall(r'([0-9]+(?:\.[0-9]+)?)', txt)
print(nums[-1] if nums else "nan")
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
            v=float(row["metric"])
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        if best is None or v>best[0]:
            best=(v,row)
if best is None:
    print("No valid runs yet.")
else:
    v,row=best
    print(f'BEST: metric={v:.6f} roi_ce_lambda={row["roi_ce_lambda"]} cons_lambda={row["cons_lambda"]} status={row["status"]}')
PY
}

echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] CSV=${CSV}"
echo

for roi in "${GRID_VALS[@]}"; do
  for cons in "${GRID_VALS[@]}"; do

    if already_done "${roi}" "${cons}"; then
      echo "[SKIP] roi=${roi} cons=${cons} (already in CSV)"
      continue
    fi

    echo "============================================================"
    echo "[RUN] roi=${roi} cons=${cons}"
    echo "============================================================"

    # Your requirement: must delete before next train
    cleanup_outputs

    export GAZE_ROI_CE_LAMBDA="${roi}"
    export GAZE_CONS_LAMBDA="${cons}"

    RUN_TAG="roi${roi}_cons${cons}"
    TRAIN_LOG="${RESULTS_DIR}/train_fewshot_${DATASET}_${RUN_TAG}.log"
    PARSE_LOG="${RESULTS_DIR}/parse_fewshot_${DATASET}_${RUN_TAG}.log"

    # Train
    set +e
    bash "${TRAIN_SCRIPT}" "${DATA_ROOT}" "${DATASET}" "${SHOTS}" "${MODEL}" 2>&1 | tee "${TRAIN_LOG}"
    rc_train=${PIPESTATUS[0]}
    set -e

    if [[ ${rc_train} -ne 0 ]]; then
      echo "[FAIL] training rc=${rc_train}"
      echo "${roi},${cons},nan,TRAIN_FAIL" >> "${CSV}"
      cleanup_outputs
      continue
    fi

    # Output dir must exist
    if [[ ! -d "${OUT_DIR}" ]]; then
      echo "[FAIL] expected OUT_DIR not found: ${OUT_DIR}"
      echo "${roi},${cons},nan,NO_OUTPUT_DIR" >> "${CSV}"
      cleanup_outputs
      continue
    fi

    # Parse
    set +e
    python parse_test_res.py --directory "${OUT_DIR}" --test-log 2>&1 | tee "${PARSE_LOG}"
    rc_parse=${PIPESTATUS[0]}
    set -e

    if [[ ${rc_parse} -ne 0 ]]; then
      echo "[FAIL] parse failed rc=${rc_parse}"
      echo "${roi},${cons},nan,PARSE_FAIL" >> "${CSV}"
      cleanup_outputs
      continue
    fi

    metric="$(parse_metric_from_log "${PARSE_LOG}")"
    echo "[METRIC] ${metric}"
    echo "${roi},${cons},${metric},OK" >> "${CSV}"

    # Your requirement: must delete after parse
    cleanup_outputs

    update_best | tee "${BEST_TXT}"

  done
done

echo
echo "[DONE] Grid search finished."
echo "[DONE] CSV: ${CSV}"
echo "[DONE] Best: ${BEST_TXT}"
echo "[DONE] Final best:"
update_best
