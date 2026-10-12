set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_NO_USAGE_STATS=1

MODEL=${MODEL:?set MODEL to a merged checkpoint or a Hub id}
DATA=${DATA:?set DATA to an evaluation file}
OUTPUT=${OUTPUT:?set OUTPUT to the predictions file to write}
mkdir -p "$(dirname "$OUTPUT")"

python inference.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --citation-format "${CITATION_FORMAT:-full}" \
    --constrained "${CONSTRAINED:-false}" \
    --tensor-parallel-size "${TP:-0}" \
    ${EXTRA_ARGS:-}
