set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-Qwen/Qwen3-8B}
DATA=${DATA:-data/snippet_citation_train.json}
CITATION_FORMAT=${CITATION_FORMAT:-full}
CITATION_WEIGHT=${CITATION_WEIGHT:-2}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-$((LORA_R * 2))}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}
MICRO_BATCH=${MICRO_BATCH:-2}
TARGET_BATCH=${TARGET_BATCH:-32}
OUTPUT_DIR=${OUTPUT_DIR:-runs/sft-qwen3-8b}

NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")}
# Hold the effective batch at TARGET_BATCH sequences regardless of GPU count.
ACCUM=$(( TARGET_BATCH / (NUM_GPUS * MICRO_BATCH) ))
[ "$ACCUM" -lt 1 ] && ACCUM=1
echo "model=$MODEL gpus=$NUM_GPUS effective_batch=$((MICRO_BATCH * ACCUM * NUM_GPUS)) output=$OUTPUT_DIR"

python -m accelerate.commands.launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m sft.train \
        --model "$MODEL" \
        --data "$DATA" \
        --output-dir "$OUTPUT_DIR" \
        --citation-format "$CITATION_FORMAT" \
        --citation-loss-weight "$CITATION_WEIGHT" \
        --lora-r "$LORA_R" \
        --lora-alpha "$LORA_ALPHA" \
        --epochs "$EPOCHS" \
        --learning-rate "$LR" \
        --per-device-batch-size "$MICRO_BATCH" \
        --gradient-accumulation-steps "$ACCUM" \
        --max-seq-len 2048 \
        --attn-implementation sdpa \
        ${EXTRA_ARGS:-}
