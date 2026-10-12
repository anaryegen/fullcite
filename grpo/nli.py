import logging
import os
import time
from typing import List, Sequence, Tuple

import torch

logger = logging.getLogger("grpo.nli")


def resolve_device(device: str) -> str:
    """Turn a bare "cuda" into this rank's concrete device.

    The scorer is constructed before the trainer has set the per-process CUDA
    device, so a bare "cuda" resolves to cuda:0 for the *weights* on every rank,
    while the inputs are later moved to whatever device the rank ended up on.
    On rank 0 that happens to agree and on every other rank it is a device
    mismatch at the first forward pass.  Pinning both ends to LOCAL_RANK here
    removes the ambiguity.
    """
    if device == "cuda" and torch.cuda.is_available():
        return "cuda:%s" % os.environ.get("LOCAL_RANK", "0")
    return device

# The checkpoint the evaluation scores support with.
EVAL_NLI_MODEL = "microsoft/deberta-v3-base-mnli-fever-anli"


class SupportScorer:
    """Scores (premise, hypothesis) pairs in [0, 1]."""

    name = "base"

    def __init__(self):
        self.pairs_scored = 0
        self.seconds = 0.0

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        start = time.perf_counter()
        scores = self._score(pairs)
        self.seconds += time.perf_counter() - start
        self.pairs_scored += len(pairs)
        return scores

    def _score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError

    @property
    def pairs_per_second(self) -> float:
        return self.pairs_scored / self.seconds if self.seconds > 0 else 0.0

    def throughput_report(self) -> dict:
        return {
            "support/pairs_scored": self.pairs_scored,
            "support/pairs_per_second": round(self.pairs_per_second, 1),
        }


class NLIScorer(SupportScorer):
    """P(entailment) with premise = snippet, hypothesis = claim."""

    name = "nli"

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 512,
    ):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        assert_distinct_from_eval(model_name)
        device = resolve_device(device)
        logger.info("Loading support scorer (NLI): %s on %s", model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if "cuda" in device else torch.float32
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
            .to(device)
            .eval()
        )
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Label order differs between NLI checkpoints; never assume index 0 or 2.
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        entailment = [i for i, label in labels.items() if "entail" in label]
        if not entailment:
            raise ValueError(
                "%s has no entailment label (id2label=%s); it is not an NLI head."
                % (model_name, labels)
            )
        self.entailment_index = entailment[0]

    @torch.no_grad()
    def _score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)
            scores.extend(probabilities[:, self.entailment_index].float().cpu().tolist())
        return scores


class CosineScorer(SupportScorer):
    """Sentence-embedding cosine, rescaled to [0, 1]. Ablation arm only."""

    name = "cosine"

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 128):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        device = resolve_device(device)
        logger.info("Loading support scorer (cosine): %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    @torch.no_grad()
    def _score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        premises = self.model.encode(
            [p for p, _ in pairs], batch_size=self.batch_size,
            convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        )
        hypotheses = self.model.encode(
            [h for _, h in pairs], batch_size=self.batch_size,
            convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False,
        )
        cosine = (premises * hypotheses).sum(dim=-1)
        return ((cosine + 1.0) / 2.0).clamp(0.0, 1.0).float().cpu().tolist()


def assert_distinct_from_eval(model_name: str) -> None:
    if model_name.strip() == EVAL_NLI_MODEL:
        raise ValueError(
            "Reward NLI model %s is the checkpoint the evaluation scores with. "
            "Training against the evaluation scorer makes the reported support "
            "gain self-referential. Choose a different checkpoint, or pass "
            "--allow-eval-nli if this is deliberate." % model_name
        )


def build_support_scorer(cfg, device: str = None):
    """Instantiate the scorer named by the reward config."""
    device = device or cfg.nli_device
    if cfg.support_scorer == "cosine":
        return CosineScorer(cfg.cosine_model, device=device)
    if cfg.support_scorer == "nli":
        return NLIScorer(
            cfg.nli_model,
            device=device,
            batch_size=cfg.nli_batch_size,
            max_length=cfg.nli_max_length,
        )
    raise ValueError("Unknown support_scorer: %s" % cfg.support_scorer)
