import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rouge_score import rouge_scorer  # noqa: E402

from get_train_data import tokenize as repo_tokenize  # noqa: E402
from constrained_decoding.citation_format import (  # noqa: E402
    DEFAULT_FORMAT,
    get_format,
)
from grpo.citations import (  # noqa: E402
    ParsedCitation,
    ParsedCompletion,
    doc_lookup,
    doc_positions,
    parse_completion,
)

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

YES_RE = re.compile(r"\b(yes|correct|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|not|incorrect|false)\b", re.IGNORECASE)



@dataclass
class RewardWeights:
    answer: float = 1.0
    doc: float = 1.0
    span: float = 1.0
    support: float = 1.0
    omission: float = 0.5
    verbose: float = 0.0
    dup: float = 0.0
    coverage: float = 0.0


@dataclass
class RewardCfg:
    weights: RewardWeights = field(default_factory=RewardWeights)

    # gate
    graded_gate_warmup_steps: int = 0     # >0: soft gate for the first N steps
    gate_collapse_threshold: float = 0.05  # pass-rate below this triggers a warning

    # span
    snippet_min_chars: int = 20
    snippet_max_chars: int = 512
    wrong_doc_penalty: float = 0.25       # alpha in the dense R_doc variant

    # support
    support_scorer: str = "nli"           # "nli" (default) or "cosine" (ablation arm)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_batch_size: int = 64
    nli_max_length: int = 512
    nli_device: str = "cuda"

    # shaping
    omission_yesno_scale: float = 2.0
    verbose_slack_ratio: float = 1.5
    dup_jaccard_threshold: float = 0.8
    late_position_index: int = 2          # docs at or after this index count as "late"

    # curriculum: fine-grained layer is muted until this step (0 = never muted)
    curriculum_finegrained_step: int = 0

    # representation
    citation_format: str = DEFAULT_FORMAT
    support_premise: str = "auto"

    @property
    def format(self):
        return get_format(self.citation_format)

    @property
    def effective_support_premise(self) -> str:
        if self.support_premise != "auto":
            return self.support_premise
        return "snippet" if self.format.carries_snippet else "document"


@dataclass
class Example:
    """The context a reward is allowed to see."""

    question: str
    docs: List[Dict[str, str]]            # [{"id": str, "content": str}, ...]
    gold_doc_ids: List[str]
    gold_spans: List[str]
    gold_answer: str
    question_type: str = "summary"        # yesno | factoid | list | summary




def token_set(text: str) -> set:
    """Punctuation-stripped lowercase tokens, matching the repo's aligner."""
    return set(repo_tokenize(text)[0])


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def verbatimness(snippet: str, document: str) -> float:
    """How much of the snippet the cited document actually contains.

    An exact substring scores 1.0.  Otherwise the score is the longest
    contiguous run of snippet tokens found in the document, divided by the
    snippet length — the same longest-common-run measure the post-hoc aligner
    uses.  Contiguity matters: a bag-of-words overlap would give full credit to
    a snippet stitched together from words scattered across the document, which
    is exactly the hallucination this term is meant to catch.
    """
    if not snippet.strip():
        return 0.0
    if snippet in document:
        return 1.0

    snippet_tokens = repo_tokenize(snippet)[0]
    doc_tokens = repo_tokenize(document)[0]
    if not snippet_tokens or not doc_tokens:
        return 0.0

    matcher = SequenceMatcher(None, snippet_tokens, doc_tokens, autojunk=False)
    match = matcher.find_longest_match(0, len(snippet_tokens), 0, len(doc_tokens))
    return match.size / len(snippet_tokens)


def rouge_l_f(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    return _ROUGE.score(reference, prediction)["rougeL"].fmeasure


def yesno_label(text: str) -> Optional[str]:
    """First polarity word in the text, or None if it commits to neither."""
    yes, no = YES_RE.search(text), NO_RE.search(text)
    if yes and no:
        return "yes" if yes.start() < no.start() else "no"
    if yes:
        return "yes"
    if no:
        return "no"
    return None



def gate_format(parsed: ParsedCompletion, cfg: RewardCfg, step: int = 0) -> float:
    """1 if the answer parses under the citation grammar, else 0.

    During warmup the gate is graded (the fraction of citation attempts that are
    well formed) so that a policy which cannot yet produce the format still sees
    a gradient; it re-hardens automatically at `graded_gate_warmup_steps`.
    """
    if step < cfg.graded_gate_warmup_steps:
        return parsed.graded_gate
    return 1.0 if parsed.parses else 0.0

def r_answer(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """Downstream correctness of the answer text, citations stripped.

    Exact polarity match for yes/no questions, ROUGE-L against the gold answer
    otherwise.  (Macro-F1 is the batch-level reporting metric; per rollout the
    dense equivalent is the per-example indicator, which is what GRPO needs.)

    This term is what stops the policy from optimizing citation quality by
    saying less and less: a citation-only reward degrades answer recall.
    """
    answer = parsed.answer_text
    if not answer:
        return 0.0
    if example.question_type == "yesno":
        predicted, gold = yesno_label(answer), yesno_label(example.gold_answer)
        return float(predicted is not None and predicted == gold)
    return rouge_l_f(answer, example.gold_answer)


def r_doc(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """F1 of cited document ids against gold, with a dense wrong-doc penalty.

    Deliberately blind to snippet quality: keeping "cited the right document"
    and "quoted the right words" on separate axes is the point of the
    decomposition, and it is what makes the two curves diagnosable apart.
    """
    if not parsed.citations:
        return 0.0

    gold = {str(d) for d in example.gold_doc_ids}
    # `doc_ids` rather than `doc_id`, so that a representation which recovers its
    # document instead of naming it is scored on what it actually identified.  For
    # the forms that name one document this is the same single-element set, so the
    # arithmetic below is unchanged.
    cited = {d for c in parsed.citations for d in c.doc_ids}
    if not gold:
        return 0.0

    hits = len(cited & gold)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Dense per-citation signal: F1 alone gives identical credit however many
    # times a wrong document is cited.  A citation none of whose candidate
    # documents is gold counts as wrong, which also prices an unresolvable quote.
    wrong = sum(1 for c in parsed.citations if not (set(c.doc_ids) & gold))
    return max(0.0, f1 - cfg.wrong_doc_penalty * wrong / len(parsed.citations))



def span_score(
    citation: ParsedCitation,
    documents: Dict[str, str],
    gold_doc_ids: set,
    gold_spans: Sequence[str],
    cfg: RewardCfg,
) -> float:
    """Span quality for one citation, conditioned on the document being right.

    The conditioning is what blocks the "copy anything verbatim" hack: quoting a
    document perfectly earns nothing unless that document is one the answer
    should have cited.
    """
    # Any candidate document that is both gold and present.  For a representation
    # that names its document this is that document; for one that recovers it from
    # the quote, the quote still has to land in a document the answer should have
    # cited -- which is what keeps the "copy anything verbatim" hack blocked when
    # there is no document id to condition on.
    candidates = [
        d for d in (citation.doc_ids or ())
        if d in gold_doc_ids and d in documents
    ]
    if not candidates:
        return 0.0
    if not cfg.snippet_min_chars <= len(citation.snippet) <= cfg.snippet_max_chars:
        return 0.0

    verbatim = max(
        verbatimness(citation.snippet, documents[d]) for d in candidates
    )
    alignment = max((jaccard(citation.snippet, g) for g in gold_spans), default=0.0)
    return 0.5 * verbatim + 0.5 * alignment


def r_span(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """Mean span quality over the answer's citations.

    Undefined, not zero, for a representation that emits no quote: there is no
    span to score.  Such arms deregister this component rather than carrying it at
    0.0 -- substituting the whole cited document would score verbatimness 1.0
    trivially and jaccard-to-gold near 0, i.e. a meaningless near-constant.
    """
    if not cfg.format.carries_snippet or not parsed.citations:
        return 0.0
    documents = doc_lookup(example.docs)
    gold = {str(d) for d in example.gold_doc_ids}
    scores = [
        span_score(c, documents, gold, example.gold_spans, cfg)
        for c in parsed.citations
    ]
    return sum(scores) / len(scores)


def support_pairs(
    parsed: ParsedCompletion,
    example: Optional[Example] = None,
    cfg: Optional[RewardCfg] = None,
) -> List[Tuple[str, str]]:
    """(premise, hypothesis=claim) pairs for the support scorer.

    Pure and model-free; the scoring itself is batched by the caller.  Claim and
    snippet are generated jointly, which inflates embedding similarity between
    them and makes it gameable by paraphrase, so entailment is the default
    scorer and cosine is kept only as an ablation arm.

    The premise is the quote where there is one and the cited document otherwise.
    Dropping the term for the quote-free representations instead would leave them
    with no groundedness signal at all, which would make the comparison a test of
    how many reward terms each arm got rather than of the representation.
    """
    premise_kind = cfg.effective_support_premise if cfg is not None else "snippet"
    if premise_kind == "snippet":
        return [
            (c.snippet, c.claim)
            for c in parsed.citations
            if c.claim.strip() and c.snippet.strip()
        ]

    documents = doc_lookup(example.docs) if example is not None else {}
    pairs = []
    for citation in parsed.citations:
        if not citation.claim.strip():
            continue
        # One premise per citation, not per candidate document: an ambiguous
        # reference should not get several attempts at entailing its claim.
        texts = [documents[d] for d in citation.doc_ids if d in documents]
        if not texts:
            continue
        pairs.append((max(texts, key=len), citation.claim))
    return pairs


def p_omission(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """Penalty for answering with no citations at all.

    Scaled up for yes/no questions, where a bare polarity answer already
    maximizes R_answer and the policy has no other reason to cite.
    """
    if parsed.is_refusal or parsed.citations:
        return 0.0
    return -(cfg.omission_yesno_scale if example.question_type == "yesno" else 1.0)


def p_verbose(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """Penalty for snippets far longer than the gold span they match.

    Counters "cite the whole paragraph", which otherwise raises verbatimness and
    entailment together at no cost.
    """
    if not cfg.format.carries_snippet or not parsed.citations:
        return 0.0

    penalties = []
    for citation in parsed.citations:
        best = max(example.gold_spans, key=lambda g: jaccard(citation.snippet, g), default="")
        if not best:
            penalties.append(0.0)
            continue
        ratio = len(citation.snippet) / max(1, len(best))
        excess = ratio - cfg.verbose_slack_ratio
        penalties.append(-min(1.0, excess) if excess > 0 else 0.0)
    return sum(penalties) / len(penalties)


def p_dup(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """Penalty for near-identical snippets across claims.

    A support-maximizing policy will otherwise attach its single best-entailing
    snippet to every claim it makes.
    """
    if not cfg.format.carries_snippet:
        return 0.0
    snippets = [c.snippet for c in parsed.citations]
    if len(snippets) < 2:
        return 0.0

    duplicates = sum(
        1
        for i in range(len(snippets))
        for j in range(i + 1, len(snippets))
        if jaccard(snippets[i], snippets[j]) >= cfg.dup_jaccard_threshold
    )
    pairs = len(snippets) * (len(snippets) - 1) / 2
    return -duplicates / pairs


def r_coverage(example: Example, parsed: ParsedCompletion, cfg: RewardCfg) -> float:
    """Unique gold-document coverage, with a bonus for documents placed late.

    Retrieval-order primacy means the easy reward is concentrated in the first
    couple of documents; the late-position term pays for reaching past them.
    """
    gold = {str(d) for d in example.gold_doc_ids}
    if not gold:
        return 0.0

    cited = {c.doc_id for c in parsed.citations}
    coverage = len(cited & gold) / len(gold)

    positions = doc_positions(example.docs)
    late_gold = {d for d in gold if positions.get(d, 0) >= cfg.late_position_index}
    late = len(cited & late_gold) / len(late_gold) if late_gold else 0.0
    return 0.7 * coverage + 0.3 * late

COMPONENTS = {
    "R_answer": (r_answer, "answer", False),
    "R_doc": (r_doc, "doc", False),
    "R_span": (r_span, "span", True),
    "P_omission": (p_omission, "omission", False),
    "P_verbose": (p_verbose, "verbose", True),
    "P_dup": (p_dup, "dup", True),
    "R_coverage": (r_coverage, "coverage", True),
}

FINE_GRAINED = {"R_span", "R_support", "P_verbose", "P_dup", "R_coverage"}
SNIPPET_DEPENDENT = {"R_span", "P_verbose", "P_dup"}


def supported_components(citation_format: str = DEFAULT_FORMAT) -> set:
    """Which reward components a representation can actually define."""
    names = set(COMPONENTS) | {"R_support"}
    if not get_format(citation_format).carries_snippet:
        names -= SNIPPET_DEPENDENT
    return names


COMMON_CORE_WEIGHTS = RewardWeights(
    answer=1.0, doc=1.0, span=0.0, support=0.0,
    omission=0.5, verbose=0.0, dup=0.0, coverage=0.0,
)


def capability_matched_weights(
    base: RewardWeights,
    citation_format: str = DEFAULT_FORMAT,
    positive_total: float = 4.0,
) -> RewardWeights:
    """`base` restricted to what this representation supports, then renormalised.

    Renormalisation is not cosmetic.  TRL's advantages are group-mean-centred, so
    a component that is identically zero across a rollout group contributes no
    gradient -- dropping terms does not directly handicap an arm.  What it does do
    is shrink the reward's magnitude against a *fixed* KL coefficient (`beta`),
    which means the arms with fewer terms would train under a tighter effective
    KL constraint.  That is not "the same training budget", so the surviving
    positive weights are rescaled to keep the reward/KL ratio constant, and the
    penalties are scaled by the same factor so their relative price is preserved.
    """
    supported = supported_components(citation_format)
    fields = {
        "answer": "R_answer", "doc": "R_doc", "span": "R_span",
        "support": "R_support", "omission": "P_omission",
        "verbose": "P_verbose", "dup": "P_dup", "coverage": "R_coverage",
    }
    kept = {
        attr: (getattr(base, attr) if name in supported else 0.0)
        for attr, name in fields.items()
    }

    positive = sum(
        value for attr, value in kept.items()
        if attr not in ("omission", "verbose", "dup")
    )
    if positive > 0:
        scale = positive_total / positive
        kept = {attr: value * scale for attr, value in kept.items()}
    return RewardWeights(**kept)


def score_all(
    example: Example,
    text: str,
    cfg: RewardCfg,
    step: int = 0,
    doc_order: Sequence[str] = (),
) -> Dict[str, float]:
    """Every model-free component for one completion, gate included.

    Convenience for offline analysis and tests; training goes through the TRL
    adapters in `grpo/train_grpo.py`.
    """
    documents = doc_lookup(example.docs)
    parsed = parse_completion(
        text,
        cfg.citation_format,
        doc_order=doc_order or sorted(documents),
        resolver=lambda snippet: {
            doc_id for doc_id, content in documents.items() if snippet in content
        },
    )
    gate = gate_format(parsed, cfg, step)
    scores = {"gate_format": gate}
    supported = supported_components(cfg.citation_format)
    for name, (fn, _, _) in COMPONENTS.items():
        if name not in supported:
            continue
        scores[name] = gate * fn(example, parsed, cfg)
    return scores
