import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constrained_decoding import Document  # noqa: E402
from constrained_decoding.constrained_decoder import (  # noqa: E402
    ConstrainedDecoder,
    ConstrainedDecodingConfig,
)

logger = logging.getLogger("grpo.rollouts")

# Populated in the training process; read by the logits processor running in the
# co-located vLLM worker (same process, see module docstring).
FSA_REGISTRY: Dict[str, ConstrainedDecoder] = {}


class DecoderCache:
    """LRU of ConstrainedDecoders keyed by prompt.

    Building the token trie costs seconds per question, and a question's
    documents never change, so the cache turns that into a one-off cost per
    prompt rather than a per-step one.  Capacity is bounded because each trie
    holds on the order of 10^5 token sequences.
    """

    def __init__(self, tokenizer, config: ConstrainedDecodingConfig, capacity: int = 64):
        self.tokenizer = tokenizer
        self.config = config
        self.capacity = capacity
        self._entries: "OrderedDict[str, ConstrainedDecoder]" = OrderedDict()
        self._token_vocab: Optional[Dict[int, str]] = None
        self.hits = 0
        self.misses = 0

    def _vocab(self) -> Dict[int, str]:
        if self._token_vocab is None:
            logger.info("Building token vocab for constrained rollouts...")
            vocab = {}
            for token_id in range(self.tokenizer.vocab_size):
                try:
                    vocab[token_id] = self.tokenizer.decode([token_id])
                except Exception:
                    continue
            self._token_vocab = vocab
            logger.info("Token vocab built: %d tokens", len(vocab))
        return self._token_vocab

    def get(self, key: str, documents: List[Dict[str, str]]) -> ConstrainedDecoder:
        if key in self._entries:
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]

        self.misses += 1
        decoder = ConstrainedDecoder(
            documents=[
                Document(document_id=str(d["id"]), text=d["content"]) for d in documents
            ],
            tokenizer=self.tokenizer,
            config=self.config,
            token_vocab=self._vocab(),
        )
        self._entries[key] = decoder
        while len(self._entries) > self.capacity:
            evicted, _ = self._entries.popitem(last=False)
            FSA_REGISTRY.pop(evicted, None)
        return decoder


def build_logits_processor_class():
    """Define the engine-level processor lazily, so vLLM is only imported if used."""
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

    class CitationFSALogitsProcessor(AdapterLogitsProcessor):
        """Applies a request's citation FSA mask to its logits."""

        def is_argmax_invariant(self) -> bool:
            # It masks tokens, so it can change which token is the argmax.
            return False

        def new_req_logits_processor(self, params):
            key = (getattr(params, "extra_args", None) or {}).get("fsa_key")
            if key is None:
                return None  # unconstrained request
            decoder = FSA_REGISTRY.get(key)
            if decoder is None:
                logger.warning("No FSA decoder registered for key %s", key)
                return None
            processor = decoder.get_logit_processor()
            if processor is None:
                return None

            def request_processor(output_token_ids, logits):
                # Same contract as the HF transformers logits processor: the
                # processor sees generated ids only.
                return processor(list(output_token_ids), logits)

            return request_processor

    return CitationFSALogitsProcessor


def install_fsa_logits_processor() -> None:
    """Register the FSA processor with the engine TRL is about to build.

    Must be called before GRPOTrainer constructs its vLLM backend.
    """
    import functools

    from trl.generation import vllm_generation

    if getattr(vllm_generation.LLM, "_fsa_patched", False):
        return

    processor_class = build_logits_processor_class()
    original_llm = vllm_generation.LLM

    @functools.wraps(original_llm)
    def llm_with_fsa(*args, **kwargs):
        existing = list(kwargs.pop("logits_processors", []) or [])
        existing.append(processor_class)
        if kwargs.get("distributed_executor_backend") != "external_launcher":
            raise RuntimeError(
                "FSA-constrained rollouts need the co-located in-process engine "
                "(distributed_executor_backend='external_launcher'); got %r. The "
                "decoder registry would not be visible to the worker otherwise."
                % kwargs.get("distributed_executor_backend")
            )
        return original_llm(*args, logits_processors=existing, **kwargs)

    llm_with_fsa._fsa_patched = True
    vllm_generation.LLM = llm_with_fsa
    logger.info("FSA logits processor installed into the TRL vLLM backend.")


class FSARollout:
    """`rollout_func` producing FSA-constrained rollouts through TRL's engine.

    TRL synchronises the policy weights into the co-located engine immediately
    before calling this, so the rollouts are on-policy.
    """

    def __init__(
        self,
        tokenizer,
        docs_by_prompt: Dict[str, List[Dict[str, str]]],
        config: ConstrainedDecodingConfig,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        cache_capacity: int = 64,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.docs_by_prompt = docs_by_prompt
        self.cache = DecoderCache(tokenizer, config, capacity=cache_capacity)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.chat_template_kwargs = chat_template_kwargs or {}

    def _render(self, prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    @staticmethod
    def _content(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        return prompt[-1]["content"]

    def __call__(self, prompts: List[Any], trainer) -> Dict[str, List]:
        from vllm import SamplingParams

        llm = trainer.vllm_generation.llm

        rendered, keys = [], []
        for prompt in prompts:
            content = self._content(prompt)
            documents = self.docs_by_prompt.get(content)
            key = str(abs(hash(content)))
            if documents is not None:
                # Registering before generate() is what makes the decoder
                # reachable from the logits processor by key.
                FSA_REGISTRY[key] = self.cache.get(key, documents)
            else:
                logger.warning("No documents for prompt; rollout will be unconstrained.")
                key = None
            rendered.append(self._render(prompt))
            keys.append(key)

        outputs = llm.generate(
            rendered,
            [
                SamplingParams(
                    n=1,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                    logprobs=0,
                    extra_args={"fsa_key": key} if key else None,
                )
                for key in keys
            ],
            use_tqdm=False,
        )

        prompt_ids, completion_ids, logprobs = [], [], []
        for output in outputs:
            best = output.outputs[0]
            prompt_ids.append(list(output.prompt_token_ids))
            completion_ids.append(list(best.token_ids))
            logprobs.append(
                [
                    entry[token].logprob
                    for token, entry in zip(best.token_ids, best.logprobs or [])
                ]
                if best.logprobs
                else [0.0] * len(best.token_ids)
            )

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "fsa_cache_hit_rate": [
                self.cache.hits / max(1, self.cache.hits + self.cache.misses)
            ] * len(outputs),
        }
