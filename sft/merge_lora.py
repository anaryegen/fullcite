import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("sft.merge_lora")


def resolve_adapter_dir(adapter: Path, checkpoint: str) -> Path:
    
    if checkpoint == "final":
        if not (adapter / "adapter_config.json").exists() and (
            adapter / "final" / "adapter_config.json"
        ).exists():
            return adapter / "final"
        return adapter
    if checkpoint == "last":
        candidates = sorted(
            (p for p in adapter.glob("checkpoint-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not candidates:
            raise SystemExit("No checkpoint-* directories under %s" % adapter)
        return candidates[-1]
    path = adapter / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
    if not path.is_dir():
        raise SystemExit("No such checkpoint: %s" % path)
    return path


def load_base(model_name: str, dtype: torch.dtype):
    kwargs = dict(trust_remote_code=True, device_map="cpu", low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        # transformers < 4.56 spells the dtype argument differently.
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def save_processor(base_model_name: str, output_dir: Path) -> bool:

    from transformers import AutoProcessor
    from transformers.processing_utils import ProcessorMixin

    try:
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    except Exception as exc:  # text-only bases have no processor to resolve
        logger.info("No processor for %s (%s); tokenizer only", base_model_name, exc)
        return False
    if not isinstance(processor, ProcessorMixin):
        return False
    logger.info("Saving %s from the base model", type(processor).__name__)
    processor.save_pretrained(str(output_dir))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True,
                        help="adapter directory produced by sft/train.py")
    parser.add_argument("--output-dir", default=None,
                        help="default: <adapter>-merged")
    parser.add_argument("--base", default=None,
                        help="base model; default: the adapter's base_model_name_or_path")
    parser.add_argument("--checkpoint", default="final",
                        help="final (the adapter dir itself), last, or checkpoint-N")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="merge dtype; keep bfloat16 to match training and vLLM")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    adapter = Path(args.adapter)
    weights_dir = resolve_adapter_dir(adapter, args.checkpoint)
    if not (weights_dir / "adapter_config.json").exists():
        raise SystemExit(
            "Not a LoRA adapter directory (no adapter_config.json): %s" % weights_dir
        )
    with open(weights_dir / "adapter_config.json", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    base_model_name = args.base or adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise SystemExit(
            "The adapter records no base_model_name_or_path; pass --base explicitly."
        )

    output_dir = Path(args.output_dir or (str(adapter).rstrip("/") + "-merged"))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit("%s exists and is not empty (pass --overwrite)" % output_dir)

    logger.info("Adapter: %s (r=%s, alpha=%s, %d target modules)",
                weights_dir, adapter_config.get("r"), adapter_config.get("lora_alpha"),
                len(adapter_config.get("target_modules") or []))
    logger.info("Base model: %s", base_model_name)

    tokenizer_source = base_model_name
    for candidate in (weights_dir, adapter):
        if (candidate / "tokenizer_config.json").exists():
            tokenizer_source = candidate
            break
    logger.info("Loading tokenizer: %s", tokenizer_source)
    tokenizer_kwargs = {}
    tokenizer_config_path = Path(tokenizer_source) / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        extra = json.loads(tokenizer_config_path.read_text()).get("extra_special_tokens")
        if isinstance(extra, list):
            tokenizer_kwargs = {"extra_special_tokens": {},
                                "additional_special_tokens": extra}
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source), trust_remote_code=True, **tokenizer_kwargs)

    logger.info("Loading base weights (CPU, %s)...", args.dtype)
    model = load_base(base_model_name, getattr(torch, args.dtype))

    embed_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embed_rows:
        logger.warning("Resizing embeddings %d -> %d to match the tokenizer",
                       embed_rows, len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    from peft import PeftModel

    logger.info("Applying adapter and folding it into the base weights...")
    model = PeftModel.from_pretrained(model, str(weights_dir), torch_dtype=model.dtype)
    model = model.merge_and_unload()
    model.config.use_cache = True

    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config

    logger.info("Saving merged model to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    save_processor(base_model_name, output_dir)
    
    tokenizer.save_pretrained(str(output_dir))

    
    for name in ("sft_config.json", "grpo_config.json"):
        for source in (adapter / name, weights_dir.parent / name):
            if source.exists():
                shutil.copy(source, output_dir / name)
                break
    with open(output_dir / "merge_info.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model": base_model_name,
                "adapter": str(adapter),
                "adapter_weights": str(weights_dir),
                "lora_r": adapter_config.get("r"),
                "lora_alpha": adapter_config.get("lora_alpha"),
                "num_target_modules": len(adapter_config.get("target_modules") or []),
                "dtype": args.dtype,
            },
            handle,
            indent=2,
        )

    if not (output_dir / "config.json").exists():
        raise SystemExit("Merge finished but wrote no config.json — vLLM will reject %s"
                         % output_dir)
    logger.info("Done. Load it with: python inference.py --model %s", output_dir)


if __name__ == "__main__":
    main()
