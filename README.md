# FullCITE: Training LLMs to Cite Verbatim Evidence Spans

Code for training and running the citation models from the paper. A model
answers a question from a set of retrieved documents and grounds each claim in a
verbatim span of one document:

```
<claim text> [doc:<document_id> snippet:"<exact text copied from that document>"]
```

Training has two stages:

1. **SFT**: LoRA fine-tuning 
2. **GRPO**: reinforcement learning from the merged SFT model, with a layered
   reward: a format gate, answer quality (`R_answer`), the correct document (`R_doc`),
   span exactness (`R_span`), NLI support of the claim by the span (`R_support`),
   and an omission penalty (`P_omission`) (`grpo/`).

Inference uses vLLM, with optional citation-constrained decoding that checks and
repairs every citation against the source documents (`inference.py`,
`constrained_decoding/`).

## Repository layout

```
get_train_data.py        build training data: align reference passages to exact document spans
sft/
  train.py               LoRA SFT with weighted citation loss
  data.py, prompts.py    example construction; the prompt shared by training and inference
  merge_lora.py          fold a LoRA adapter into its base model (needed by vLLM)
grpo/
  train_grpo.py          GRPO training (TRL)
  rewards.py, nli.py     reward components and the NLI support scorer
  citations.py           completion parsing
  rollouts.py            optional vLLM / FSA-constrained rollouts
  configs/               main run and paper ablations
constrained_decoding/    citation formats, document index, verifier, constrained decoder
config/, models/         vLLM engine wrapper
inference.py             batched generation (free or constrained)
scripts/                 launchers for each step
```

## Installation

```bash
conda create -n incite python=3.11 && conda activate fullcite
pip install -r requirements.txt
```

Training was run on a single node with 2× A100 80GB GPUs.

## Data

Evaluation data: ASQA, ExpertQA and BioASQ are located in `data/`

```

Each output record has this form:

```json
{"question": "...",
 "parsed_documents": [{"id": 1, "content": "..."}],
 "snippets": [{"document": 1, "text": "...", "start": 0, "end": 42}],
 "answer": "<claim> [doc: 1 snippet: ...] <claim> ..."}
```

**Evaluation data** (ASQA, ExpertQA, BioASQ) is a JSON list with one entry per question:

```json
{"body": "question text",
 "type": "question type",
 "parsed_documents": [{"id": "93714402", "content": "document text"}]}
```

The first five documents with an id and non-empty content are given to the model.

## Training

```bash
# 1. SFT: Qwen3-8B, LoRA r=64/α=128, 3 epochs, lr 1e-5, effective batch 32,
#    citation-token loss weight 2
bash scripts/train_sft.sh                                   # -> runs/sft-qwen3-8b

# 2. Merge the adapter (GRPO and vLLM need full weights)
ADAPTER=runs/sft-qwen3-8b bash scripts/merge_lora.sh        # -> runs/sft-qwen3-8b-merged

# 3. GRPO: 1000 steps, 8 generations per prompt, lr 3e-5, β=0.04
bash scripts/train_grpo.sh                                  # -> runs/grpo-qwen3-8b

# 4. Merge the GRPO adapter for inference
ADAPTER=runs/grpo-qwen3-8b bash scripts/merge_lora.sh       # -> runs/grpo-qwen3-8b-merged
```

Every hyperparameter can be overridden through environment variables (see the
header of each script). The full configuration of each run is written to
`sft_config.json` / `grpo_config.json` in its output directory.

### Ablations

**Reward components.** These drop one term from the main reward. The remaining
weights are rescaled so the total reward weight, and so the effective KL constraint,
stays the same.

```bash
CONFIG=grpo/configs/ablation_no_doc.yaml     bash scripts/train_grpo.sh
CONFIG=grpo/configs/ablation_no_span.yaml    bash scripts/train_grpo.sh
CONFIG=grpo/configs/ablation_no_support.yaml bash scripts/train_grpo.sh
```


## Inference

```bash
MODEL=runs/grpo-qwen3-8b-merged DATA=data/asqa.json OUTPUT=outputs/asqa.json \
    bash scripts/inference.sh

# citation-constrained decoding
CONSTRAINED=true MODEL=... DATA=... OUTPUT=... bash scripts/inference.sh
```

