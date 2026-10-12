import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

SENTENCE_RE = re.compile(r'\S.*?(?:[.!?]+(?=\s|$)|$)', re.DOTALL)


def _dedupe(values: Sequence[Optional[int]]) -> List[int]:
    """Unique values, order preserved, `None` dropped."""
    seen: Set[int] = set()
    out: List[int] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


@dataclass(frozen=True)
class CitationRef:
    """One parsed citation, in whatever detail its representation carries.

    `doc_ids` is a tuple rather than a single id because `snippet_only` recovers
    the document by locating the quote, and a quote can occur in more than one of
    the provided documents.  An empty tuple means the citation named no document
    and none could be recovered; more than one means the reference is genuinely
    ambiguous.  Collapsing either case to a single id would report a precision
    number the representation has not earned.
    """

    doc_ids: Tuple[str, ...]
    snippet: Optional[str]
    index: Optional[int]
    span: Tuple[int, int]
    raw: str

    @property
    def doc_id(self) -> str:
        """The single resolved document id, or `''`.

        Kept so that callers written against the one-document-per-citation world
        (`grpo/rewards.py`, `grpo/monitor.py`) keep working unchanged.
        """
        return self.doc_ids[0] if len(self.doc_ids) == 1 else ""

    @property
    def is_ambiguous(self) -> bool:
        return len(self.doc_ids) > 1


@dataclass(frozen=True)
class CitationFormat:
    """A citation surface form: how to write it, match it, and misspell it."""

    name: str
    carries_doc_id: bool
    carries_snippet: bool
    doc_ref_kind: str          # "id" | "index" | "resolved"
    attachment: str            # "inline" | "after_sentence"

    # --- surface ---
    pattern: re.Pattern
    malformed: Tuple[Tuple[re.Pattern, str], ...]
    render_template: str
    # Group numbers rather than names, so `full`'s regex stays the exact object
    # (and the exact pattern string) it was before this module existed.
    doc_group: Optional[int]
    snippet_group: Optional[int]
    index_group: Optional[int]
    # The literal that opens a citation, consumed by the FSA one character at a
    # time.  None for `marker`, which opens on a digit rather than a keyword.
    opening_literal: Optional[str]
    instruction_body: str



    def render(
        self,
        *,
        doc_id: Optional[str] = None,
        snippet: Optional[str] = None,
        doc_index: Optional[int] = None,
    ) -> str:
        """Render one citation.

        Raises rather than silently dropping a field: a representation asked to
        render something it cannot express is a bug in the caller, and a citation
        that quietly lost its snippet would corrupt a training target.
        """
        if self.carries_doc_id and self.doc_ref_kind == "id" and not doc_id:
            raise ValueError("%s requires doc_id" % self.name)
        if self.carries_snippet and not snippet:
            raise ValueError("%s requires snippet" % self.name)
        if self.doc_ref_kind == "index" and doc_index is None:
            raise ValueError("%s requires doc_index" % self.name)
        return self.render_template % {
            "doc_id": doc_id,
            "snippet": snippet,
            "index": doc_index,
        }

    def assemble(
        self,
        pieces: Sequence[tuple],
        doc_order: Sequence[str] = (),
    ) -> str:
        """Join an answer out of text and citation pieces.

        `pieces` are `("text", str)` and `("cite", doc_id, snippet)`, where
        `doc_id` is already remapped and `snippet` already sanitised — the
        citations are identical across representations, and only their rendering
        differs.  That is the whole point of routing the join through here.

        Spacing is collapsed on text pieces only, never on the joined answer:
        dropping a citation can leave a double space behind, but the same
        substitution applied afterwards would rewrite snippets whose source
        document contains a run of spaces, and they would stop being verbatim.
        """
        if self.attachment == "after_sentence":
            return self._assemble_after_sentence(pieces, doc_order)

        out: List[str] = []
        for piece in pieces:
            if piece[0] == "text":
                out.append(re.sub(r"[ \t]+", " ", piece[1]))
            else:
                _, doc_id, snippet = piece
                out.append(self.render(
                    doc_id=doc_id,
                    snippet=snippet,
                    doc_index=self._index_of(doc_id, doc_order),
                ))
        return "".join(out)

    def _index_of(self, doc_id: str, doc_order: Sequence[str]) -> Optional[int]:
        if self.doc_ref_kind != "index":
            return None
        order = [str(d) for d in doc_order]
        try:
            return order.index(str(doc_id)) + 1
        except ValueError:
            raise ValueError(
                "document %r is not in the prompt's document order" % doc_id
            ) from None

    def _assemble_after_sentence(
        self,
        pieces: Sequence[tuple],
        doc_order: Sequence[str],
    ) -> str:
        """Attach citations after the sentence they fall inside (ALCE style).

        The gold citations sit at claim boundaries mid-answer, but a marker
        representation cites per sentence, so each citation has to migrate to the
        end of its sentence and duplicates within a sentence collapse.

        Markers go *after* the terminating punctuation rather than before it, as
        ALCE writes them.  The evaluator attaches citations to the sentence they
        follow, and splits prose on the same regex used here; with the marker
        inside the sentence, `"A [1]."` splits into prose "A", a citation, and a
        second prose unit "." which the evaluator counts as an uncited sentence.
        Placing the marker after the period keeps this representation inside the
        evaluator's existing attachment contract without changing the evaluator.
        """
        text_parts: List[str] = []
        pending: List[Tuple[int, int]] = []   # (offset in joined text, index)
        length = 0
        for piece in pieces:
            if piece[0] == "text":
                collapsed = re.sub(r"[ \t]+", " ", piece[1])
                text_parts.append(collapsed)
                length += len(collapsed)
            else:
                _, doc_id, _snippet = piece
                pending.append((length, self._index_of(doc_id, doc_order)))

        joined = "".join(text_parts)
        sentences = [m.span() for m in SENTENCE_RE.finditer(joined)]
        if not sentences:
            # No sentence to attach to (empty or punctuation-free answer); keep
            # the citations rather than silently dropping them.
            markers = _dedupe([index for _, index in pending])
            tail = "".join(self.render(doc_index=i) for i in markers)
            return (joined + " " + tail).strip() if markers else joined

        # A citation supports the text *before* it, so it belongs to the last
        # sentence that had already begun at its offset.  Keying on sentence ends
        # instead would be wrong: a citation written just after a period sits in
        # the whitespace *between* two sentences, past the first one's end but
        # before the second one's start, and would migrate forward by one
        # sentence -- attaching every citation to the claim after the one it was
        # written for.
        buckets: List[List[int]] = [[] for _ in sentences]
        for offset, index in pending:
            slot = 0
            for candidate, (start, _end) in enumerate(sentences):
                if start < offset:
                    slot = candidate
                else:
                    break
            buckets[slot].append(index)

        out: List[str] = []
        cursor = 0
        for slot, (start, end) in enumerate(sentences):
            out.append(joined[cursor:end])
            markers = _dedupe(buckets[slot])
            if markers:
                out.append(" " + "".join(
                    self.render(doc_index=i) for i in markers
                ))
            cursor = end
        out.append(joined[cursor:])

        # Moving the citation off the gap it was written in leaves the space that
        # followed it next to the one introduced above.  Collapsing runs of spaces
        # is safe here and nowhere else in this module: this representation emits
        # no quoted span, so there is no verbatim text that rewriting whitespace
        # could invalidate.  Leaving them would make the arm differ from the others
        # in its prose as well as its citations.
        return re.sub(r"[ \t]{2,}", " ", "".join(out))

    def spans(self, text: str) -> List[Tuple[int, int]]:
        """Character spans of every well-formed citation, whole bracket included.

        This is what the SFT loss mask is built from, so it must cover the
        bracket and not just the quoted span.
        """
        return [match.span() for match in self.pattern.finditer(text)]

    def strip(self, text: str) -> str:
        """`text` with its citations removed."""
        return self.pattern.sub("", text)

    def instructions(
        self,
        *,
        example_ids: str,
        fallback_message: str,
        n_docs: int = 0,
    ) -> str:
        """The citation-format block of the prompt.

        Every representation keeps the same scaffolding sentence and the same six
        numbered rules; only the format line and rules 2-3 differ.  Holding the
        rule count fixed keeps instruction length from becoming a second
        difference between the arms.
        """
        return self.instruction_body % {
            "example_ids": example_ids,
            "fallback_message": fallback_message,
            "n_docs": n_docs,
        }

    def document_header(self, doc_id: str, index: int) -> str:
        """The per-document header in the prompt's document listing.

        `marker` shows the index, because a representation that cites by position
        must be able to *see* the position; inferring it by counting would make
        the arm a test of arithmetic rather than of the representation.
        """
        if self.doc_ref_kind == "index":
            return "Document [%d] %s:" % (index, doc_id)
        return "Document %s:" % doc_id

    def parse(
        self,
        text: str,
        *,
        doc_order: Sequence[str] = (),
        resolver: Optional[Callable[[str], Set[str]]] = None,
    ) -> List[CitationRef]:
        """Every well-formed citation in `text`.

        `doc_order` is the document order of the prompt that produced `text`, used
        to turn a `marker` index back into a document id.  `resolver` maps a
        verbatim snippet to the ids of the documents containing it, used by
        `snippet_only`.  Both are optional: without them the affected citations
        come back with an empty `doc_ids`, which is honest, rather than a guessed
        id, which is not.
        """
        refs: List[CitationRef] = []
        for match in self.pattern.finditer(text):
            snippet = (
                match.group(self.snippet_group).strip()
                if self.snippet_group is not None
                else None
            )

            index: Optional[int] = None
            doc_ids: Tuple[str, ...] = ()

            if self.doc_group is not None:
                doc_ids = (match.group(self.doc_group).strip(),)
            elif self.index_group is not None:
                index = int(match.group(self.index_group))
                if 1 <= index <= len(doc_order):
                    doc_ids = (str(doc_order[index - 1]),)
            elif snippet and resolver is not None:
                doc_ids = tuple(sorted(resolver(snippet)))

            refs.append(
                CitationRef(
                    doc_ids=doc_ids,
                    snippet=snippet,
                    index=index,
                    span=match.span(),
                    raw=match.group(0),
                )
            )
        return refs

    def count_malformed(self, text: str) -> int:
        """Citation attempts this representation refuses.

        Well-formed citations are removed first, so that a valid citation is not
        also counted as a malformed one by the looser patterns.
        """
        residue = self.pattern.sub("", text)
        return sum(len(pattern.findall(residue)) for pattern, _ in self.malformed)

    def find_malformed(self, text: str) -> Dict[str, List[int]]:
        """Malformed-attempt positions, keyed by what is wrong with them."""
        residue = self.pattern.sub("", text)
        found: Dict[str, List[int]] = {}
        for pattern, label in self.malformed:
            positions = [m.start() for m in pattern.finditer(residue)]
            if positions:
                found[label] = positions
        return found


_FULL_PATTERN = re.compile(r'\[doc:([^\s\]]+)\s+snippet:"([^"]+)"\]')

_FULL_MALFORMED = (
    (re.compile(r'\[doc:([^\s\]]*)\]'), "Missing snippet"),
    (re.compile(r'\[doc:\s*\]'), "Empty document ID"),
    (re.compile(r'\[doc:([^\s\]]+)\s+snippet:\]'), "Empty snippet"),
    (re.compile(r'\[doc:([^\s\]]+)\s+snippet:([^"\]]+)\]'), "Unquoted snippet"),
)

_FULL_INSTRUCTIONS = """You are answering questions based on the provided documents.
For every factual claim, you MUST include a citation in this exact format:
[doc:<document_id> snippet:"<exact text from document>"]

Rules:
1. The document_id must be one of the provided documents (e.g., %(example_ids)s)
2. The snippet must be an EXACT verbatim quote from the referenced document
3. Do not invent or paraphrase snippets
4. Make sure to answer the question directly and avoid over-citation by citing unrelated parts of the document
5. If you cannot find supporting evidence, say: "%(fallback_message)s"
6. It is guaranteed that the documents are relevant and have enough information to provide answer with citations.
"""

FULL = CitationFormat(
    name="full",
    carries_doc_id=True,
    carries_snippet=True,
    doc_ref_kind="id",
    attachment="inline",
    pattern=_FULL_PATTERN,
    malformed=_FULL_MALFORMED,
    render_template='[doc:%(doc_id)s snippet:"%(snippet)s"]',
    doc_group=1,
    snippet_group=2,
    index_group=None,
    opening_literal="[doc:",
    instruction_body=_FULL_INSTRUCTIONS,
)



# def get_format(name) -> CitationFormat:
#     """Look up a representation, or say which ones exist."""
#     if isinstance(name, CitationFormat):
#         return name
#     try:
#         return REPRESENTATIONS[name]
#     except KeyError:
#         raise KeyError(
#             "unknown citation format %r; available: %s"
#             % (name, ", ".join(sorted(REPRESENTATIONS)))
#         ) from None
