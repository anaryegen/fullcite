import os
os.environ["VLLM_USE_V1"] = "1"

import argparse
import logging
import sys
from pathlib import Path
import json
from tqdm import tqdm
# Make the repo root importable regardless of the working directory
sys.path.insert(0, str(Path(__file__).parent))

from constrained_decoding.citation_format import DEFAULT_FORMAT, REPRESENTATIONS
from constrained_decoding import (
    Document,
    ConstrainedDecodingConfig,
    DecodingStrategy,
)
from sft.prompts import build_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> list:
    """Read the dataset, accepting either a JSON array or JSONL."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = [json.loads(line) for line in open(data_path, 'r', encoding='utf-8').readlines()]
    logger.info(f"Loaded {len(data)} data points from {data_path}")
    return data


def create_documents(documents) -> list[Document]:
    """Create Document objects from sample data."""
    return [
        Document(
            document_id=str(doc["id"]),
            text=doc["content"]
        )
        for doc in documents[:5] if doc['id'] and doc['content']
    ]


class Checkpoint:

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.partial_path = f"{output_path}.partial"
        self.results = []
        if os.path.exists(self.partial_path):
            with open(self.partial_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            logger.info(
                f"Resuming from {self.partial_path}: "
                f"{len(self.results)} samples already done"
            )

    def save(self) -> None:
        # Write-then-rename so an interruption mid-write cannot truncate the
        # file we would resume from.
        tmp_path = f"{self.partial_path}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.partial_path)

    def finish(self) -> None:
        logger.info(f"Saving {len(self.results)} results to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        # The run completed, so the resume file has served its purpose.
        if os.path.exists(self.partial_path):
            os.remove(self.partial_path)


def default_tensor_parallel_size() -> int:

    import torch

    n = torch.cuda.device_count()
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def build_engine(model_name: str, max_model_len: int, tensor_parallel_size: int = 0):
    """Load the base vLLM engine, or return None if it cannot be loaded."""
    from config import VLLMEngineConfig, SamplingConfig
    from models import VLLMEngine

    if tensor_parallel_size <= 0:
        tensor_parallel_size = default_tensor_parallel_size()

    logger.info("1. Initializing vLLM Engine...")
    logger.info(f"   tensor_parallel_size={tensor_parallel_size}")
    try:
        vllm_config = VLLMEngineConfig(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
        )
        engine = VLLMEngine(
            vllm_config=vllm_config,
            sampling_config=SamplingConfig()
        )
        logger.info(f"   Model loaded: {model_name}")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        return None

PROMPT_CONFIG = ConstrainedDecodingConfig(
    min_snippet_length=20,
    max_snippet_length=512,
)

CITATION_FORMAT = PROMPT_CONFIG.citation_format


def set_citation_format(name: str) -> None:
    """Select the citation representation for both prompt and decoding."""
    global PROMPT_CONFIG, CITATION_FORMAT
    CITATION_FORMAT = name
    PROMPT_CONFIG = ConstrainedDecodingConfig(
        min_snippet_length=PROMPT_CONFIG.min_snippet_length,
        max_snippet_length=PROMPT_CONFIG.max_snippet_length,
        citation_format=name,
    )


def build_inference_prompt(data_point) -> str:
    
    documents = create_documents(data_point.get("parsed_documents", []))
    return build_prompt(data_point['body'], documents, PROMPT_CONFIG)


def run_constrained(model_name: str, output_path: str, data_path: str, batch_size: int,
                    tensor_parallel_size: int = 0, max_model_len: int = 0):
    """
    Run constrained generation.

    Args:
        model_name: HuggingFace model name or path
        output_path: Path to save the generated results
        data_path: Path to the dataset file
        batch_size: Questions submitted to vLLM per scheduling round
        max_model_len: Context window; 0 keeps this mode's default (4096)
    """
    logger.info("=" * 60)
    logger.info("CONSTRAINED DECODING")
    logger.info("=" * 60)

    from config import SamplingConfig
    from constrained_decoding.vllm_integration import (
        ConstrainedVLLMEngine,
        ConstrainedGenerationConfig
    )

    base_engine = build_engine(model_name, max_model_len=max_model_len or 4096,
                               tensor_parallel_size=tensor_parallel_size)
    if base_engine is None:
        return

    logger.info("2. Initializing Constrained Engine...")

    strategy = DecodingStrategy.HYBRID
    needs_tries = strategy == DecodingStrategy.LOGIT_BIAS
    logger.info(f"   Using {strategy.name} decoding strategy")

    decoder_config = ConstrainedDecodingConfig(
        strategy=strategy,
        min_snippet_length=20,
        max_snippet_length=512,
        build_trie=needs_tries,
        build_token_trie=needs_tries,
        enforce_at_least_one_citation=True,
        max_retries=3,
        citation_format=CITATION_FORMAT,
    )

    gen_config = ConstrainedGenerationConfig(
        temperature=0.7,  # Higher temperature for more natural text
        max_tokens=512,
        require_at_least_one_citation=True,
        max_retries=3
    )

    if needs_tries:
        logger.info("   LOGIT_BIAS uses per-question logit processors; forcing batch_size=1")
        batch_size = 1

    data = load_data(data_path)
    ckpt = Checkpoint(output_path)
    pending = data[len(ckpt.results):]

    logger.info("3. Generating Answers with Constraints...")
    with tqdm(total=len(data), initial=len(ckpt.results), desc="Generating") as pbar:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            engines = []
            prompts = []
            for data_point in chunk:
                documents = create_documents(data_point.get("parsed_documents", []))
                constrained_engine = ConstrainedVLLMEngine(
                    vllm_engine=base_engine,
                    documents=documents,
                    decoder_config=decoder_config
                )
                engines.append(constrained_engine)
                prompts.append(
                    build_prompt(
                        data_point['body'],
                        documents,
                        decoder_config,
                        document_index=constrained_engine.decoder.document_index,
                    )
                )

            states = [
                {
                    "best_output": None,
                    "best_valid_count": -1,
                    "best_verification": None,
                    "retries": 0,
                }
                for _ in chunk
            ]
    
            active = list(range(len(chunk)))
            temperature = gen_config.temperature

            for attempt in range(gen_config.max_retries):
                if not active:
                    break

                # Adjust temperature on retries
                if attempt > 0:
                    temperature = min(temperature * 1.5, 2.0)

                sampling_params = SamplingConfig(
                    temperature=temperature,
                    top_p=gen_config.top_p,
                    top_k=gen_config.top_k,
                    max_tokens=gen_config.max_tokens,
                    repetition_penalty=gen_config.repetition_penalty
                )

                logit_processors = None
                if needs_tries:
                    processor = engines[active[0]].decoder.get_logit_processor()
                    if processor is not None:
                        logit_processors = [processor]
                    else:
                        logger.warning(
                            "LOGIT_BIAS strategy requested but no tokenizer available; "
                            "falling back to post-validation"
                        )

                outputs = base_engine.generate(
                    [prompts[i] for i in active],
                    sampling_params,
                    logit_processors=logit_processors
                )

                still_active = []
                for i, output in zip(active, outputs):
                    constrained_engine = engines[i]

                    # HYBRID: attempt to fix invalid citations before verifying
                    if strategy == DecodingStrategy.HYBRID:
                        output, _ = constrained_engine.decoder._fix_citations(output)

                    verification = constrained_engine.verifier.verify(output)

                    # Track best output
                    state = states[i]
                    valid_count = verification.stats.get('valid_citations', 0)
                    if valid_count > state["best_valid_count"]:
                        state["best_valid_count"] = valid_count
                        state["best_output"] = output
                        state["best_verification"] = verification

                    accepted = verification.is_valid and (
                        not gen_config.require_at_least_one_citation
                        or verification.stats.get('total_citations', 0) > 0
                    )
                    if accepted:
                        continue

                    state["retries"] += 1
                    logger.debug(f"Attempt {attempt + 1} failed: {verification.errors}")
                    still_active.append(i)

                active = still_active

            for data_point, state, constrained_engine in zip(chunk, states, engines):
                # Keep the last generated text for debugging even if falling back
                last_generated_text = state["best_output"]
                best_output = state["best_output"]
                best_verification = state["best_verification"]

                if best_output is None or (
                    gen_config.require_at_least_one_citation
                    and state["best_valid_count"] == 0
                ):
                    if best_output is not None:
                        logger.debug(
                            f"Falling back to default message. Last generated text was:\n{best_output}"
                        )
                    best_output = gen_config.fallback_message
                    best_verification = constrained_engine.verifier.verify(best_output)

                ckpt.results.append({
                    "question": data_point['body'],
                    "type": data_point['type'],
                    "generated_text": best_output,
                    "is_valid": best_verification.is_valid,
                    "retries": state["retries"],
                    "verification": best_verification.to_dict(),
                    "last_generated_text": last_generated_text
                })

            ckpt.save()
            pbar.update(len(chunk))

    ckpt.finish()

    logger.info("=" * 60)
    logger.info("CONSTRAINED GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


def run_inference(model_name: str, output_path: str, data_path: str, batch_size: int,
                  tensor_parallel_size: int = 0, max_model_len: int = 0):

    logger.info("=" * 60)
    logger.info("CLASSIC INFERENCE (Non-Constrained)")
    logger.info("=" * 60)

    base_engine = build_engine(model_name, max_model_len=max_model_len or 8096,
                               tensor_parallel_size=tensor_parallel_size)
    if base_engine is None:
        return

    data = load_data(data_path)
    ckpt = Checkpoint(output_path)
    pending = data[len(ckpt.results):]

    logger.info("2. Generating Answers (Non-Constrained)...")
    with tqdm(total=len(data), initial=len(ckpt.results), desc="Generating") as pbar:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            prompts = [build_inference_prompt(dp) for dp in chunk]

            # One generate() call per batch rather than per question.  vLLM
            # schedules up to max_num_seqs (256) sequences concurrently, so
            # submitting a single prompt at a time left the GPU almost entirely
            # idle; the engine admits as many of these as the KV cache holds and
            # queues the rest itself.
            outputs = base_engine.generate(prompts)

            for data_point, result in zip(chunk, outputs):
                generated_text = result if isinstance(result, str) else result.get('text', str(result))
                ckpt.results.append({
                    "question": data_point['body'],
                    "type": data_point['type'],
                    "generated_text": generated_text,
                    "is_valid": None,  # No validation for non-constrained
                    "retries": 0,
                    "verification": None,
                    "last_generated_text": None
                })

            ckpt.save()
            pbar.update(len(chunk))

    ckpt.finish()

    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


def str2bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "t", "yes", "y")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cited answers with a (fine-tuned) model, optionally under citation-constrained decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file for saving generated results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Questions submitted to vLLM per scheduling round; also the "
             "crash-resume checkpoint granularity"
    )
    parser.add_argument(
        "--citation-format",
        default=DEFAULT_FORMAT,
        choices=sorted(REPRESENTATIONS),
        help="Citation representation to emit and enforce. Must match the one "
             "the checkpoint was trained with; the prompt's instructions come "
             "from it, so a mismatch asks the model for one form and scores "
             "another"
    )
    parser.add_argument(
        "--constrained",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use constrained decoding (true) or classic inference (false)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="Number of GPUs to shard the model over. 0 (default) picks the "
             "largest power of two that fits the visible devices; a 70B model "
             "in bf16 needs at least 4 x A100-80GB"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Context window handed to vLLM. 0 (default) keeps each mode's "
             "own default (8096 free / 4096 constrained); set it lower for a "
             "model with a shorter window, e.g. 4096 for Llama-2"
    )
    args = parser.parse_args()
    set_citation_format(args.citation_format)

    if args.constrained:
        run_constrained(args.model, args.output, args.data, args.batch_size,
                        args.tensor_parallel_size, args.max_model_len)
    else:
        run_inference(args.model, args.output, args.data, args.batch_size,
                      args.tensor_parallel_size, args.max_model_len)
