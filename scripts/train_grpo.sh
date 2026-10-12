set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
# Rollout batch shapes vary step to step; expandable segments limit fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=${CONFIG:-grpo/configs/grpo.yaml}
ROLLOUT=${ROLLOUT:-hf}   # hf | vllm | vllm-fsa
EXTRA=()
[ -n "${MODEL:-}" ] && EXTRA+=(--model "$MODEL")
[ -n "${OUTPUT_DIR:-}" ] && EXTRA+=(--output-dir "$OUTPUT_DIR")
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")
[ -n "${REPORT_TO:-}" ] && EXTRA+=(--report-to "$REPORT_TO")
if [ -n "${RESUME:-}" ]; then
    case "$RESUME" in
        1|true|auto) EXTRA+=(--resume) ;;
        *)           EXTRA+=(--resume "$RESUME") ;;
    esac
fi

NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")}
echo "config=$CONFIG rollout=$ROLLOUT gpus=$NUM_GPUS"

python -m accelerate.commands.launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m grpo.train_grpo \
        --config "$CONFIG" \
        --rollout "$ROLLOUT" \
        "${EXTRA[@]+"${EXTRA[@]}"}"
