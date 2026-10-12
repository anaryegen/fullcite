import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constrained_decoding.constrained_decoder import ConstrainedDecodingConfig  # noqa: E402
from grpo.citations import completion_text, parse_completion  # noqa: E402
from grpo.data import load_dataset_splits  # noqa: E402
from grpo.monitor import Diagnostics, GRPOMonitorCallback  # noqa: E402
from grpo.nli import build_support_scorer  # noqa: E402
from constrained_decoding.citation_format import DEFAULT_FORMAT  # noqa: E402
from grpo.rewards import (  # noqa: E402
    COMMON_CORE_WEIGHTS,
    COMPONENTS,
    FINE_GRAINED,
    capability_matched_weights,
    Example,
    RewardCfg,
    RewardWeights,
    gate_format,
    support_pairs,
)

logger = logging.getLogger("grpo.train")


@dataclass
class RunCfg:
    model: str = "runs/sft-qwen3-8b-r32-w1"       # SFT checkpoint, or a base model for RL-only
    data: str = "data/snippet_citation_train.json"
    output_dir: str = "runs/grpo-v1"

    # GRPO
    num_generations: int = 8                       # group size (8-16)
    max_completion_length: int = 512
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    steps_per_generation: int = 1
    learning_rate: float = 1.0e-6
    beta: float = 0.04                             # KL penalty against the SFT reference
    temperature: float = 0.9                       # rollout diversity
    top_p: float = 0.95
    max_steps: int = 1000
    save_steps: int = 100
    logging_steps: int = 5
    seed: int = 42
    eval_size: int = 128
    max_rows: int = 0
    scale_rewards: str = "group"
    rollout: str = "hf"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_max_model_length: int = 4096
    fsa_cache_capacity: int = 64

    # LoRA
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    report_to: str = "none"
    citation_format: str = DEFAULT_FORMAT
    reward_policy: str = "as_written"

    reward: RewardCfg = field(default_factory=RewardCfg)


def load_cfg(path: Optional[str]) -> RunCfg:
    cfg = RunCfg()
    if not path:
        return cfg
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    reward_raw = raw.pop("reward", {}) or {}
    weights_raw = reward_raw.pop("weights", {}) or {}
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise ValueError("Unknown config key: %s" % key)
        setattr(cfg, key, value)
    for key, value in reward_raw.items():
        if not hasattr(cfg.reward, key):
            raise ValueError("Unknown reward config key: %s" % key)
        setattr(cfg.reward, key, value)
    for key, value in weights_raw.items():
        if not hasattr(cfg.reward.weights, key):
            raise ValueError("Unknown reward weight: %s" % key)
        setattr(cfg.reward.weights, key, value)
    return cfg

class RewardRuntime:
    """State the reward adapters share: the step counter, scorer, diagnostics."""

    def __init__(self, cfg: RewardCfg, support_scorer=None):
        self.cfg = cfg
        self.step = 0
        self.support_scorer = support_scorer
        self.diagnostics = Diagnostics()

    def fine_grained_active(self) -> bool:
        """Curriculum stage 2 gate.

        Stage 1 trains the gate and the base layer only; the fine-grained terms
        are muted until `curriculum_finegrained_step`.  Muting here rather than
        by zeroing `reward_weights` keeps each component's logged curve alive
        through the transition, so the effect of switching stages is visible.
        """
        return self.step >= self.cfg.curriculum_finegrained_step

    def support_throughput(self) -> Dict[str, float]:
        return self.support_scorer.throughput_report() if self.support_scorer else {}


def build_examples_from_columns(columns: Dict[str, List], size: int) -> List[Example]:
    return [
        Example(
            question=columns["question"][i],
            docs=columns["docs"][i],
            gold_doc_ids=columns["gold_doc_ids"][i],
            gold_spans=columns["gold_spans"][i],
            gold_answer=columns["gold_answer"][i],
            question_type=columns["question_type"][i],
        )
        for i in range(size)
    ]

def parse_for(runtime: RewardRuntime, text: str, example) -> "ParsedCompletion":
    """Parse one completion under the run's citation representation.

    Routed through here rather than calling `parse_completion(text)` directly in
    each adapter, because the two arguments that resolve a citation's document --
    the prompt's document order for `marker`, and a snippet resolver for
    `snippet_only` -- have to be supplied identically to every component, or the
    same completion would score against different attributions in different terms.
    """
    documents = {str(d["id"]): d["content"] for d in example.docs}
    return parse_completion(
        text,
        runtime.cfg.citation_format,
        doc_order=sorted(documents),
        resolver=lambda snippet: {
            doc_id for doc_id, content in documents.items() if snippet in content
        },
    )


def make_component_adapter(name: str, runtime: RewardRuntime):
    """Vectorize one pure component into TRL's reward-function signature."""
    pure_fn, _, is_fine_grained = COMPONENTS[name]

    def adapter(completions, **columns) -> List[float]:
        texts = [completion_text(c) for c in completions]
        examples = build_examples_from_columns(columns, len(texts))
        if is_fine_grained and not runtime.fine_grained_active():
            return [0.0] * len(texts)

        scores = []
        for text, example in zip(texts, examples):
            parsed = parse_for(runtime, text, example)
            gate = gate_format(parsed, runtime.cfg, runtime.step)
            scores.append(gate * pure_fn(example, parsed, runtime.cfg) if gate else 0.0)
        return scores

    adapter.__name__ = name
    return adapter


def make_support_adapter(runtime: RewardRuntime):
    """R_support: pure pair extraction, one batched call to the learned scorer."""

    def adapter(completions, **columns) -> List[float]:
        texts = [completion_text(c) for c in completions]
        if not runtime.fine_grained_active():
            return [0.0] * len(texts)

        examples = build_examples_from_columns(columns, len(texts))

        gates, indices, pairs = [], [], []
        for text, example in zip(texts, examples):
            parsed = parse_for(runtime, text, example)
            gate = gate_format(parsed, runtime.cfg, runtime.step)
            gates.append(gate)
            current = []
            if gate > 0.0:
                # The premise is the quote where the representation has one and
                # the cited document otherwise, so this needs the example too.
                for pair in support_pairs(parsed, example, runtime.cfg):
                    current.append(len(pairs))
                    pairs.append(pair)
            indices.append(current)

        scores = runtime.support_scorer.score(pairs)
        return [
            gate * (sum(scores[i] for i in idx) / len(idx)) if idx and gate else 0.0
            for gate, idx in zip(gates, indices)
        ]

    adapter.__name__ = "R_support"
    return adapter


def make_gate_adapter(runtime: RewardRuntime):
    """Weight-0 component: logs the gate pass-rate and drives the diagnostics."""

    def adapter(completions, **columns) -> List[float]:
        texts = [completion_text(c) for c in completions]
        examples = build_examples_from_columns(columns, len(texts))
        runtime.diagnostics.update(texts, examples, runtime.cfg)
        return [
            gate_format(parse_for(runtime, text, example), runtime.cfg, runtime.step)
            for text, example in zip(texts, examples)
        ]

    adapter.__name__ = "gate_format"
    return adapter


def build_reward_funcs(runtime: RewardRuntime):
    """(functions, weights) for TRL. Zero-weight components are left out entirely."""
    weights_cfg = runtime.cfg.weights
    funcs, weights = [make_gate_adapter(runtime)], [0.0]

    for name, (_, attribute, _) in COMPONENTS.items():
        weight = getattr(weights_cfg, attribute)
        if weight == 0.0:
            continue
        funcs.append(make_component_adapter(name, runtime))
        weights.append(weight)

    if weights_cfg.support != 0.0:
        if runtime.support_scorer is None:
            raise ValueError("support weight is non-zero but no scorer was built")
        funcs.append(make_support_adapter(runtime))
        weights.append(weights_cfg.support)

    return funcs, weights


def check_model_path(model: str) -> None:
    """Reject an adapter directory with an actionable message.

    `sft/train.py` saves an adapter, not a model, so a run directory holds
    `adapter_config.json` and no `config.json`.  GRPO needs real weights: both
    `AutoConfig.from_pretrained` (for LoRA target resolution) and vLLM resolve a
    local path by looking for `config.json`, and vLLM's failure mode is the
    misleading "Invalid repository ID or local directory specified".  Merge
    first with `scripts/merge_lora.sh`.
    """
    path = Path(model)
    if not path.is_dir():
        return  # a Hub id
    if (path / "config.json").exists():
        return
    if (path / "adapter_config.json").exists():
        raise SystemExit(
            "%s is a LoRA adapter directory, not a model. GRPO starts from "
            "merged weights: run `ADAPTER=%s bash scripts/merge_lora.sh` and point "
            "--model at the resulting *-merged directory." % (model, model)
        )
    raise SystemExit("%s has no config.json and is not an adapter directory." % model)


def resolve_resume(resume, output_dir: str) -> Optional[str]:
    """Turn --resume into a concrete checkpoint path, and say which one it is.

    A long GRPO run is worth restarting rather than repeating: the trainer state,
    the optimizer and the RNG streams all live in the checkpoint, so a resumed run
    continues the schedule instead of re-warming the gate from step 0.
    """
    if resume is None:
        return None
    if resume is not True:
        path = Path(resume)
        if not path.is_dir():
            raise SystemExit("--resume %s is not a directory." % resume)
        logger.info("Resuming from %s", path)
        return str(path)

    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    if last is None:
        raise SystemExit(
            "--resume was passed without a checkpoint and %s holds none." % output_dir
        )
    logger.info("Resuming from the newest checkpoint in %s: %s", output_dir, last)
    return last


def resolve_lora_targets(model_name: str) -> List[str]:
    """Explicit LoRA target names, with multimodal towers excluded.

    Reuses the SFT resolver so a Gemma-3 checkpoint does not pick up adapters on
    its SigLIP vision tower, whose projections share the names q_proj/k_proj/
    v_proj with the language model.
    """
    import torch  # noqa: F401  (imported for side effects in accelerate)
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from sft.train import DEFAULT_TARGET_SUFFIXES, resolve_target_modules

    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        )
    return resolve_target_modules(skeleton, set(DEFAULT_TARGET_SUFFIXES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML overriding RunCfg / RewardCfg")
    parser.add_argument("--model", help="override cfg.model (SFT checkpoint or base)")
    parser.add_argument("--output-dir", help="override cfg.output_dir")
    parser.add_argument("--rollout", choices=["hf", "vllm", "vllm-fsa"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--report-to")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT",
                        help="resume from CHECKPOINT, or from the newest checkpoint in "
                             "the output directory when passed without a value")
    parser.add_argument("--allow-eval-nli", action="store_true",
                        help="permit using the evaluation NLI checkpoint as the reward scorer")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_cfg(args.config)
    for key in ("model", "output_dir", "rollout", "max_steps", "max_rows", "report_to"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)

    if args.allow_eval_nli:
        import grpo.nli as nli_module

        nli_module.assert_distinct_from_eval = lambda name: None

    check_model_path(cfg.model)

    # One representation for the whole run: the prompt the dataset is built with
    # and the grammar the reward parses must be the same, so the reward's copy is
    # derived here rather than being settable on its own in the YAML.
    if cfg.reward.citation_format not in (DEFAULT_FORMAT, cfg.citation_format):
        raise SystemExit(
            "reward.citation_format (%r) contradicts citation_format (%r); set it "
            "once, at the top level" % (cfg.reward.citation_format, cfg.citation_format)
        )
    cfg.reward.citation_format = cfg.citation_format

    if cfg.reward_policy == "common_core":
        cfg.reward.weights = replace(COMMON_CORE_WEIGHTS)
    elif cfg.reward_policy == "capability_matched":
        cfg.reward.weights = capability_matched_weights(
            cfg.reward.weights, cfg.citation_format
        )
    elif cfg.reward_policy != "as_written":
        raise SystemExit(
            "unknown reward_policy %r; choose as_written, common_core or "
            "capability_matched" % cfg.reward_policy
        )

    logger.info("Run config:\n%s", yaml.dump(asdict(cfg), sort_keys=False))

    decoding_config = ConstrainedDecodingConfig(
        min_snippet_length=cfg.reward.snippet_min_chars,
        max_snippet_length=cfg.reward.snippet_max_chars,
        citation_format=cfg.citation_format,
    )

    train_dataset, eval_dataset, data_stats = load_dataset_splits(
        cfg.data,
        decoding_config,
        eval_size=cfg.eval_size,
        max_rows=cfg.max_rows,
        seed=cfg.seed,
    )

    support_scorer = None
    if cfg.reward.weights.support != 0.0:
        support_scorer = build_support_scorer(cfg.reward)
    runtime = RewardRuntime(cfg.reward, support_scorer)

    reward_funcs, reward_weights = build_reward_funcs(runtime)
    logger.info(
        "Active reward components: %s",
        [(f.__name__, w) for f, w in zip(reward_funcs, reward_weights)],
    )

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    # Match the SFT prompt exactly: Qwen3 renders an empty thinking block when
    # the flag is off, and a rollout prompt that differed from the SFT prompt
    # would start the policy off-distribution.
    chat_template_kwargs = (
        {"enable_thinking": False}
        if "enable_thinking" in (tokenizer.chat_template or "")
        else {}
    )

    grpo_args = GRPOConfig(
        output_dir=cfg.output_dir,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        steps_per_generation=cfg.steps_per_generation,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        seed=cfg.seed,
        bf16=True,
        scale_rewards=cfg.scale_rewards,
        reward_weights=reward_weights,
        chat_template_kwargs=chat_template_kwargs,
        log_completions=True,
        num_completions_to_print=2,
        report_to=cfg.report_to,
        use_vllm=cfg.rollout in ("vllm", "vllm-fsa"),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_max_model_length=cfg.vllm_max_model_length,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    peft_config = None
    if cfg.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolve_lora_targets(cfg.model),
        )

    rollout_func = None
    if cfg.rollout == "vllm-fsa":
        from grpo.rollouts import FSARollout, install_fsa_logits_processor

        install_fsa_logits_processor()
        rollout_func = FSARollout(
            tokenizer=tokenizer,
            docs_by_prompt={
                row["prompt"][-1]["content"]: row["docs"]
                for row in list(train_dataset) + list(eval_dataset)
            },
            config=decoding_config,
            max_new_tokens=cfg.max_completion_length,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            cache_capacity=cfg.fsa_cache_capacity,
            chat_template_kwargs=chat_template_kwargs,
        )

    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=rollout_func,
    )
    monitor = GRPOMonitorCallback(runtime)
    trainer.add_callback(monitor)
    trainer.callback_handler.callbacks.remove(monitor)
    trainer.callback_handler.callbacks.insert(0, monitor)

    trainer.train(resume_from_checkpoint=resolve_resume(args.resume, cfg.output_dir))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir / "final"))
    with open(output_dir / "grpo_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(cfg),
                "reward_components": [f.__name__ for f in reward_funcs],
                "reward_weights": reward_weights,
                "data_stats": dict(data_stats),
            },
            handle,
            indent=2,
        )
    logger.info("Saved GRPO adapter to %s", output_dir / "final")


if __name__ == "__main__":
    main()
