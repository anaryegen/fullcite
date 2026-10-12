import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constrained_decoding.citation_format import (  # noqa: E402
    DEFAULT_FORMAT,
    CitationFormat,
    get_format,
)
from constrained_decoding.citation_verifier import CitationVerifier  # noqa: E402
from constrained_decoding.constrained_decoder import (  # noqa: E402
    ConstrainedDecodingConfig,
)

CITATION_RE = CitationVerifier.CITATION_PATTERN
MALFORMED_PATTERNS = CitationVerifier.MALFORMED_PATTERNS

REFUSAL = ConstrainedDecodingConfig().fallback_message.lower().rstrip(".")


@dataclass
class ParsedCitation:
    """One citation plus the claim it is attached to."""

    claim: str          # text since the previous citation — what this one supports
    doc_id: str         # the single resolved document, or "" when none/ambiguous
    snippet: str
    order: int          # position in the answer, for primacy diagnostics
    
    doc_ids: Tuple[str, ...] = ()
    index: Optional[int] = None   # 1-based prompt position, for `marker`

    @property
    def is_ambiguous(self) -> bool:
        return len(self.doc_ids) > 1


@dataclass
class ParsedCompletion:
    citations: List[ParsedCitation]
    n_malformed: int
    is_refusal: bool
    raw: str
    citation_format: str = DEFAULT_FORMAT

    @property
    def parses(self) -> bool:
        """Hard gate: the output contains no malformed citation attempts.

        The gate is *format validity*, not "did it cite something".  Requiring at
        least one citation here would make `P_omission` unreachable: an answer
        with no citations would fail the gate, the gate would zero the whole
        reward including the omission penalty, and the term meant to punish
        not citing could never fire.  Splitting them keeps each term doing
        distinct work — the gate rejects broken syntax, P_omission prices
        missing citations, and an uncited answer still scores zero on every
        citation-dependent component.

        A refusal passes for the same reason: declining to answer is a
        legitimate output under the prompt's own instructions, and zeroing it
        would train the policy to fabricate citations rather than abstain.
        """
        return self.n_malformed == 0

    @property
    def graded_gate(self) -> float:
        """Soft gate for warmup: fraction of citation attempts that are well formed.

        An output with no citation attempts at all scores 1.0 here for the same
        reason it passes the hard gate — nothing it wrote is malformed, and the
        omission is priced by `P_omission`.
        """
        if self.is_refusal:
            return 1.0
        total = len(self.citations) + self.n_malformed
        return len(self.citations) / total if total else 1.0

    @property
    def answer_text(self) -> str:
        """The answer with citations removed — what R_answer is scored on.

        Stripped with *this* representation's pattern.  Using the default here
        would leave `[1]` markers in the text that R_answer is ROUGE'd against,
        charging that arm for its own citations while `full` is scored clean.
        """
        pattern = get_format(self.citation_format).pattern
        return re.sub(r"\s+", " ", pattern.sub(" ", self.raw)).strip()


def count_malformed(text: str, citation_format: str = DEFAULT_FORMAT) -> int:
    """Citation attempts that the grammar refuses.

    Valid citations are removed first so that a well-formed citation is not also
    counted as a malformed one by the looser patterns.  Which attempts count as
    malformed is representation-specific, and getting this wrong is not a small
    error: `[doc:ID]` is `full`'s "Missing snippet" but is `doc_only`'s *correct*
    output, so scoring that arm with full's malformed set would zero its format
    gate on every citation it emitted and train it to stop citing.
    """
    return get_format(citation_format).count_malformed(text)


def parse_completion(
    text: str,
    citation_format: str = DEFAULT_FORMAT,
    doc_order: Sequence[str] = (),
    resolver=None,
) -> ParsedCompletion:
    """Parse a completion under one citation representation.

    `doc_order` and `resolver` are how a citation that does not name a document
    gets one: a `marker` index is a position in the prompt's document listing, and
    a `snippet_only` quote is located in the documents.  Both default to nothing,
    in which case those citations keep an empty `doc_ids` rather than a guessed
    one -- a reward computed from an invented attribution is worse than one
    computed from none.
    """
    fmt = get_format(citation_format)
    citations: List[ParsedCitation] = []
    last_end = 0
    for order, ref in enumerate(
        fmt.parse(text, doc_order=doc_order, resolver=resolver)
    ):
        citations.append(
            ParsedCitation(
                claim=text[last_end:ref.span[0]].strip(),
                doc_id=ref.doc_id,
                snippet=ref.snippet or "",
                order=order,
                doc_ids=ref.doc_ids,
                index=ref.index,
            )
        )
        last_end = ref.span[1]

    return ParsedCompletion(
        citations=citations,
        n_malformed=count_malformed(text, citation_format),
        is_refusal=REFUSAL in text.lower(),
        raw=text,
        citation_format=fmt.name,
    )


def completion_text(completion) -> str:
    """TRL hands back a string or a chat-format list; accept either."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def doc_lookup(docs: Sequence[Dict[str, str]]) -> Dict[str, str]:
    return {str(d["id"]): d["content"] for d in docs}


def doc_positions(docs: Sequence[Dict[str, str]]) -> Dict[str, int]:
    return {str(d["id"]): i for i, d in enumerate(docs)}
