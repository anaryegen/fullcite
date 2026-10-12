import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from transformers import TrainerCallback

from grpo.citations import ParsedCompletion, doc_positions, parse_completion
from grpo.rewards import Example, RewardCfg, jaccard

logger = logging.getLogger("grpo.monitor")

# Thresholds at which a pattern stops being noise and starts being the policy.
ALERTS = {
    "gate/pass_rate": ("below", 0.05, "gate collapse: nearly every rollout fails to parse"),
    "diag/snippet_len_at_floor": ("above", 0.5, "snippet-length collapse to the minimum"),
    "diag/first_sentence_doc1": ("above", 0.5, "first-sentence-of-doc-1 spam"),
    "diag/snippet_dup_rate": ("above", 0.5, "snippet duplication across claims"),
    "diag/claim_snippet_overlap": ("above", 0.8, "claims are paraphrases of their snippets"),
    "diag/refusal_rate": ("above", 0.3, "policy is learning to refuse"),
}


@dataclass
class Diagnostics:
    """Accumulates rollout statistics over a generation batch."""

    counts: Counter = field(default_factory=Counter)
    sums: Counter = field(default_factory=Counter)
    position_counts: Counter = field(default_factory=Counter)

    def reset(self) -> None:
        self.counts.clear()
        self.sums.clear()
        self.position_counts.clear()

    def update(
        self,
        texts: Sequence[str],
        examples: Sequence[Example],
        cfg: RewardCfg,
    ) -> None:
        for text, example in zip(texts, examples):
            parsed = parse_completion(text)
            self._update_one(parsed, example, cfg)

    def _update_one(
        self,
        parsed: ParsedCompletion,
        example: Example,
        cfg: RewardCfg,
    ) -> None:
        self.counts["completions"] += 1
        self.counts["gate_pass"] += int(parsed.parses)
        self.counts["malformed"] += parsed.n_malformed
        self.counts["refusals"] += int(parsed.is_refusal)
        self.sums["answer_chars"] += len(parsed.answer_text)

        if not parsed.citations:
            self.counts["uncited_completions"] += 1
            return

        self.counts["citations"] += len(parsed.citations)
        positions = doc_positions(example.docs)
        documents = {str(d["id"]): d["content"] for d in example.docs}

        # The snippet diagnostics exist to catch ways of gaming a *quote*; a
        # representation that emits none cannot exhibit them, and feeding it
        # through here would report a snippet length of 0 and a 100% at-floor
        # rate, tripping the degeneracy alert on every step of a healthy run.
        track_snippets = cfg.format.carries_snippet

        if track_snippets:
            self.counts["snippet_tracked"] += len(parsed.citations)

        for citation in parsed.citations:
            if track_snippets:
                self.sums["snippet_chars"] += len(citation.snippet)
                if len(citation.snippet) <= cfg.snippet_min_chars + 5:
                    self.counts["snippet_at_floor"] += 1

            position = positions.get(citation.doc_id)
            if position is not None:
                self.position_counts[position] += 1
                document = documents.get(citation.doc_id, "")
                # "Opening of the first document" — the highest-prior guess.
                if (
                    track_snippets
                    and position == 0
                    and document.startswith(citation.snippet[:40])
                ):
                    self.counts["first_sentence_doc1"] += 1

            if citation.claim and track_snippets:
                self.sums["claim_snippet_overlap"] += jaccard(citation.claim, citation.snippet)
                self.counts["claims"] += 1

        snippets = [c.snippet for c in parsed.citations] if track_snippets else []
        if len(snippets) >= 2:
            pairs = 0
            duplicates = 0
            for i in range(len(snippets)):
                for j in range(i + 1, len(snippets)):
                    pairs += 1
                    duplicates += jaccard(snippets[i], snippets[j]) >= cfg.dup_jaccard_threshold
            self.counts["snippet_pairs"] += pairs
            self.counts["snippet_dups"] += duplicates

    def summary(self) -> Dict[str, float]:
        n = self.counts["completions"]
        if not n:
            return {}

        citations = self.counts["citations"]
        report = {
            "gate/pass_rate": self.counts["gate_pass"] / n,
            "gate/parse_failure_rate": 1.0 - self.counts["gate_pass"] / n,
            "gate/malformed_per_completion": self.counts["malformed"] / n,
            "diag/citations_per_completion": citations / n,
            "diag/uncited_rate": self.counts["uncited_completions"] / n,
            "diag/refusal_rate": self.counts["refusals"] / n,
            "diag/answer_chars": self.sums["answer_chars"] / n,
        }
        # Keyed on snippets actually tracked, not on citations: a representation
        # that emits no quote would otherwise report snippet_chars 0 and an
        # at-floor rate of 0, which read as a healthy quote distribution rather
        # than as the absence of one.
        tracked = self.counts["snippet_tracked"]
        if tracked:
            report["diag/snippet_chars"] = self.sums["snippet_chars"] / tracked
            report["diag/snippet_len_at_floor"] = self.counts["snippet_at_floor"] / tracked
            report["diag/first_sentence_doc1"] = self.counts["first_sentence_doc1"] / tracked
        if self.counts["claims"]:
            report["diag/claim_snippet_overlap"] = (
                self.sums["claim_snippet_overlap"] / self.counts["claims"]
            )
        if self.counts["snippet_pairs"]:
            report["diag/snippet_dup_rate"] = (
                self.counts["snippet_dups"] / self.counts["snippet_pairs"]
            )
        report["diag/doc_position_entropy"] = self.position_entropy()
        return report

    def position_entropy(self) -> float:
        """Normalized entropy of the cited-document positions.

        1.0 means citations are spread evenly over the retrieved list; near 0
        means the policy has settled on one slot regardless of the question.
        """
        total = sum(self.position_counts.values())
        if total == 0 or len(self.position_counts) < 2:
            return 0.0
        entropy = -sum(
            (c / total) * math.log(c / total) for c in self.position_counts.values() if c
        )
        return entropy / math.log(len(self.position_counts))


def check_alerts(report: Dict[str, float]) -> List[str]:
    """Degenerate patterns visible in this report."""
    triggered = []
    for key, (direction, threshold, message) in ALERTS.items():
        if key not in report:
            continue
        value = report[key]
        if (direction == "above" and value > threshold) or (
            direction == "below" and value < threshold
        ):
            triggered.append("%s (%s=%.3f)" % (message, key, value))
    return triggered


class GRPOMonitorCallback(TrainerCallback):
    """Keeps the reward runtime in step and folds diagnostics into the logs."""

    def __init__(self, runtime):
        self.runtime = runtime

    def on_train_begin(self, args, state, control, **kwargs):
        # on_step_end only fires once a step has completed, so on a resumed run the
        # first rollout would be scored at step 0 -- with the graded gate back in
        # its warmup and the curriculum back in stage 1. Seed the counter instead.
        self.runtime.step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        self.runtime.step = state.global_step
        if (
            self.runtime.cfg.graded_gate_warmup_steps
            and state.global_step == self.runtime.cfg.graded_gate_warmup_steps
        ):
            logger.info(
                "Step %d: graded gate warmup over, hardening to the binary gate.",
                state.global_step,
            )
        if (
            self.runtime.cfg.curriculum_finegrained_step
            and state.global_step == self.runtime.cfg.curriculum_finegrained_step
        ):
            logger.info(
                "Step %d: curriculum stage 2, fine-grained layer now active.",
                state.global_step,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        report = self.runtime.diagnostics.summary()
        if not report:
            return
        report.update(self.runtime.support_throughput())
        logs.update({k: round(v, 4) for k, v in report.items()})

        for alert in check_alerts(report):
            logger.warning("[degeneracy] step %d: %s", state.global_step, alert)
        self.runtime.diagnostics.reset()
