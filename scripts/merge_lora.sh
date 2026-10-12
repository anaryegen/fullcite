set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER=${ADAPTER:?set ADAPTER to a run directory, e.g. runs/sft-qwen3-8b}
OUTPUT_DIR=${OUTPUT_DIR:-${ADAPTER%/}-merged}
CHECKPOINT=${CHECKPOINT:-final}

python -m sft.merge_lora \
    --adapter "$ADAPTER" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    ${BASE:+--base "$BASE"} \
    ${OVERWRITE:+--overwrite}
echo "Merged model: $OUTPUT_DIR"
