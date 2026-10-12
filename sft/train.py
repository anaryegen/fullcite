import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constrained_decoding.citation_format import (  # noqa: E402
    DEFAULT_FORMAT,
    REPRESENTATIONS,
)
from constrained_decoding.constrained_decoder import ConstrainedDecodingConfig  # noqa: E402
from sft.data import (  # noqa: E402
    CitationSFTDataset,
    SFTCollator,
    build_examples,
)

logger = logging.getLogger("sft.train")

DEFAULT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

EXCLUDED_NAME_PARTS = (
    "vision_tower", "vision_model", "multi_modal_projector", "audio_tower",
    "vision_embed_tokens",
)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


def resolve_target_modules(model, suffixes) -> list:
    
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        if parts[-1] not in suffixes:
            continue
        if any(part in EXCLUDED_NAME_PARTS for part in parts):
            continue
        names.append(name)
    return names


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:

    weights = weights.masked_fill(labels.eq(-100), 0.0)
    total_weight = weights.sum()

    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, logits.shape[1], chunk_size):
        end = start + chunk_size
        chunk = logits[:, start:end].float()
        losses = F.cross_entropy(
            chunk.reshape(-1, chunk.shape[-1]),
            labels[:, start:end].reshape(-1),
            reduction="none",
            ignore_index=-100,
        )
        total = total + (losses * weights[:, start:end].reshape(-1)).sum()

    return total / total_weight.clamp(min=1.0)


class WeightedLossTrainer(Trainer):


    def __init__(self, *args, loss_chunk_size: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_chunk_size = loss_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        loss = weighted_cross_entropy(
            outputs.logits[:, :-1, :],
            labels[:, 1:],
            weights[:, 1:],
            chunk_size=self.loss_chunk_size,
        )
        return (loss, outputs) if return_outputs else loss


def load_model(args):
    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    kwargs = dict(
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    if quantization_config is not None:
        # Quantized weights cannot be moved after loading, so each rank places
        # its own copy on the device the launcher assigned to it.
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    try:
        return AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except TypeError:
        # transformers < 4.56 spells the dtype argument differently.
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(args.model, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument("--data", default="data/snippet_citation_train.json")
    data.add_argument("--doc-id-style", choices=["synthetic", "index"],
                      default="synthetic",
                      help="synthetic gives training documents PubMed-shaped ids")
    data.add_argument("--max-documents", type=int, default=0,
                      help="cap documents per example (0 = keep all)")
    data.add_argument("--max-seq-len", type=int, default=2048)
    data.add_argument("--eval-fraction", type=float, default=0.02)
    data.add_argument("--max-examples", type=int, default=0,
                      help="truncate the dataset, for smoke tests")
    data.add_argument("--keep-uncited", action="store_true",
                      help="keep answers whose citations were all dropped")
    data.add_argument("--min-snippet-chars", type=int, default=20)
    data.add_argument("--max-snippet-chars", type=int, default=512)
    data.add_argument("--citation-format", default=DEFAULT_FORMAT,
                      choices=sorted(REPRESENTATIONS),
                      help="citation representation to train (default: full, the "
                           "project's [doc:<id> snippet:\"<text>\"])")
    data.add_argument("--length-filter-reference", default=None,
                      choices=sorted(REPRESENTATIONS),
                      help="decide which examples exceed --max-seq-len using this "
                           "representation's rendering instead of the trained "
                           "one. Pass 'full' across a representation ablation so "
                           "every arm trains on the same examples; the terser "
                           "forms would otherwise keep examples full drops.")

    model = parser.add_argument_group("model")
    model.add_argument("--model", default="Qwen/Qwen3-8B",
                       help="any causal LM: Qwen3, Gemma-3, Llama, ...")
    model.add_argument("--lora-r", type=int, default=32)
    model.add_argument("--lora-alpha", type=int, default=64)
    model.add_argument("--lora-dropout", type=float, default=0.05)
    model.add_argument("--target-suffixes", nargs="+", default=list(DEFAULT_TARGET_SUFFIXES))
    model.add_argument("--load-in-4bit", action="store_true")
    model.add_argument("--attn-implementation", default="sdpa",
                       choices=["sdpa", "eager", "flash_attention_2"])

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--citation-loss-weight", type=float, default=1.0,
                       help="loss weight on tokens inside a citation (ablation: 1 vs 2)")
    optim.add_argument("--epochs", type=float, default=2.0)
    optim.add_argument("--learning-rate", type=float, default=1e-4)
    optim.add_argument("--warmup-ratio", type=float, default=0.03)
    optim.add_argument("--weight-decay", type=float, default=0.0)
    optim.add_argument("--per-device-batch-size", type=int, default=1)
    optim.add_argument("--gradient-accumulation-steps", type=int, default=8)
    optim.add_argument("--no-gradient-checkpointing", action="store_true")
    optim.add_argument("--loss-chunk-size", type=int, default=512)
    optim.add_argument("--seed", type=int, default=42)

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", required=True)
    out.add_argument("--logging-steps", type=int, default=20)
    out.add_argument("--save-steps", type=int, default=0,
                     help="0 saves once per epoch")
    out.add_argument("--report-to", default="none")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    set_seed(args.seed)

    decoding_config = ConstrainedDecodingConfig(
        min_snippet_length=args.min_snippet_chars,
        max_snippet_length=args.max_snippet_chars,
        citation_format=args.citation_format,
    )

    logger.info("Loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not tokenizer.is_fast:
        raise SystemExit(
            "A fast tokenizer is required: citation loss weights come from the "
            "offset mapping, which the slow tokenizers do not provide."
        )

    logger.info("Loading data: %s", args.data)
    with open(args.data, encoding="utf-8") as handle:
        records = json.load(handle)

    data_stats = Counter()
    examples = build_examples(
        records,
        config=decoding_config,
        doc_id_style=args.doc_id_style,
        max_documents=args.max_documents,
        keep_uncited=args.keep_uncited,
        stats=data_stats,
        length_filter_reference=args.length_filter_reference,
    )
    random.Random(args.seed).shuffle(examples)
    if args.max_examples:
        examples = examples[:args.max_examples]

    n_eval = int(len(examples) * args.eval_fraction)
    eval_examples, train_examples = examples[:n_eval], examples[n_eval:]

    encode_stats = Counter()
    train_dataset = CitationSFTDataset(
        tokenizer, train_examples, args.max_seq_len, args.citation_loss_weight,
        stats=encode_stats,
    )
    eval_dataset = (
        CitationSFTDataset(
            tokenizer, eval_examples, args.max_seq_len, args.citation_loss_weight,
            stats=Counter(),
        )
        if eval_examples else None
    )

    if is_main_process():
        logger.info("Data: %s", dict(data_stats))
        logger.info("Encoding: %s", dict(encode_stats))
        logger.info("Train examples: %d | eval examples: %d",
                    len(train_dataset), len(eval_dataset) if eval_dataset else 0)
        if encode_stats["answer_tokens"]:
            logger.info(
                "Citation tokens: %.1f%% of supervised tokens",
                100.0 * encode_stats["citation_tokens"] / encode_stats["answer_tokens"],
            )

    if len(train_dataset) == 0:
        raise SystemExit("No training examples survived preprocessing.")

    logger.info("Loading model: %s", args.model)
    base_model = load_model(args)
    base_model.config.use_cache = False

    from peft import LoraConfig, get_peft_model
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
        )

    targets = resolve_target_modules(base_model, set(args.target_suffixes))
    if not targets:
        raise SystemExit(
            "No LoRA target modules matched %s for %s"
            % (args.target_suffixes, type(base_model).__name__)
        )
    if is_main_process():
        logger.info("LoRA targets: %d modules (e.g. %s)", len(targets), targets[0])

    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    if not args.no_gradient_checkpointing:
        peft_model.enable_input_require_grads()
    if is_main_process():
        peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        #warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        eval_strategy="epoch" if eval_dataset else "no",
        prediction_loss_only=True,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )

    trainer = WeightedLossTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTCollator(pad_token_id=tokenizer.pad_token_id),
        loss_chunk_size=args.loss_chunk_size,
    )

    trainer.train()

    if is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        # The adapter alone does not say how the prompt was built; record the
        # settings the evaluation has to reproduce.
        with open(output_dir / "sft_config.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "args": vars(args),
                    "data_stats": dict(data_stats),
                    "encoding_stats": dict(encode_stats),
                    "num_train_examples": len(train_dataset),
                    "lora_target_modules": targets,
                },
                handle,
                indent=2,
            )
        logger.info("Saved adapter and tokenizer to %s", output_dir)


if __name__ == "__main__":
    main()
