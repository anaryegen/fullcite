import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from constrained_decoding.citation_format import DEFAULT_FORMAT, get_format

from .prompts import (
    QUOTE,
    build_document_index,
    build_prompt,
    chat_head,
    chat_tail,
    citation_spans,
    make_documents,
    render_citation,
)

# Citations as written by get_train_data.py.
LEGACY_OPENER = re.compile(r"\[doc: (\d+) snippet: ")

SENTENCE_END = re.compile(r"[.!?]\s")


def split_legacy_answer(
    answer: str,
    contents: Dict[str, str],
) -> List[Tuple[str, ...]]:
    """Split an answer into ("text", s) and ("cite", doc_id, snippet) pieces.
    """
    pieces: List[Tuple[str, ...]] = []
    position = 0

    for match in LEGACY_OPENER.finditer(answer):
        if match.start() < position:
            continue  # opener sits inside a snippet already consumed
        content = contents.get(match.group(1))
        if content is None:
            continue

        rest = answer[match.end():]
        end = None
        for bracket in (m.start() for m in re.finditer(r"\]", rest)):
            if rest[:bracket] in content:
                end = bracket
        if end is None:
            continue

        pieces.append(("text", answer[position:match.start()]))
        pieces.append(("cite", match.group(1), rest[:end]))
        position = match.end() + end + 1

    pieces.append(("text", answer[position:]))
    return pieces


def synthetic_document_id(content: str, taken: set) -> str:
    """A deterministic 8-digit id with the same surface form as a PubMed id."""
    digest = int(hashlib.blake2b(content.encode("utf-8"), digest_size=8).hexdigest(), 16)
    candidate = 10_000_000 + digest % 90_000_000
    while str(candidate) in taken:
        candidate = 10_000_000 + (candidate + 1) % 90_000_000
    return str(candidate)


def sanitize_snippet(
    snippet: str,
    content: str,
    min_chars: int,
    max_chars: int,
) -> Optional[str]:
    """Coerce a snippet into something the citation grammar can express.
    """
    if QUOTE in snippet:
        # A snippet cannot contain the delimiter; keep its longest quote-free run.
        snippet = max(snippet.split(QUOTE), key=len)

    snippet = snippet.strip()

    if len(snippet) > max_chars:
        window = snippet[:max_chars + 1]
        cuts = [m.start() + 1 for m in SENTENCE_END.finditer(window)]
        cut = cuts[-1] if cuts else window.rfind(" ")
        if cut <= 0:
            return None
        snippet = snippet[:cut].strip()

    if not min_chars <= len(snippet) <= max_chars:
        return None
    if snippet not in content:
        return None
    return snippet


@dataclass
class SFTExample:
    """One prompt/answer pair, plus provenance for inspection."""

    question: str
    prompt: str
    answer: str
    documents: List[Dict[str, str]] = field(default_factory=list)
    num_citations: int = 0
    citation_format: str = DEFAULT_FORMAT
    length_reference_answer: str = ""
    length_reference_prompt: str = ""


def build_examples(
    records: Sequence[Dict[str, Any]],
    config,
    doc_id_style: str = "synthetic",
    max_documents: int = 0,
    keep_uncited: bool = False,
    stats: Optional[Counter] = None,
    length_filter_reference: Optional[str] = None,
) -> List[SFTExample]:
    """Convert raw training records into prompt/answer pairs.

    Args:
        records: parsed `snippet_citation_train.json`.
        config: ConstrainedDecodingConfig supplying the snippet length bounds
            and the fallback message, so training and decoding agree on them.
        doc_id_style: "synthetic" for PubMed-shaped ids, "index" to keep 1..N.
        max_documents: keep at most this many documents (0 = all).
        keep_uncited: keep records whose citations were all dropped.
        length_filter_reference: render each answer under this representation as
            well, and use *its* length for the max_seq_len decision.  Pass "full"
            across an ablation so every arm trains on the same examples.
    """
    stats = stats if stats is not None else Counter()
    examples: List[SFTExample] = []
    fmt = get_format(getattr(config, "citation_format", DEFAULT_FORMAT))
    reference_fmt = (
        get_format(length_filter_reference) if length_filter_reference else None
    )
    use_reference = reference_fmt is not None and reference_fmt is not fmt
    reference_config = (
        replace(config, citation_format=reference_fmt.name) if use_reference else None
    )

    for record in records:
        documents = record["parsed_documents"]
        if max_documents:
            documents = documents[:max_documents]

        taken: set = set()
        remap: Dict[str, str] = {}
        contents: Dict[str, str] = {}
        for doc in documents:
            old = str(doc["id"])
            if doc_id_style == "synthetic":
                new = synthetic_document_id(doc["content"], taken)
            else:
                new = old
            taken.add(new)
            remap[old] = new
            contents[old] = doc["content"]

        pieces: List[tuple] = []
        n_citations = 0
        for piece in split_legacy_answer(record["answer"], contents):
            if piece[0] == "text":
                pieces.append(piece)
                continue

            _, old_id, snippet = piece
            stats["citations_seen"] += 1
            if old_id not in remap:
                stats["citations_dropped_document_removed"] += 1
                continue

            clean = sanitize_snippet(
                snippet,
                contents[old_id],
                config.min_snippet_length,
                config.max_snippet_length,
            )
            if clean is None:
                stats["citations_dropped_unrepresentable"] += 1
                continue
            if clean != snippet:
                stats["citations_repaired"] += 1

            pieces.append(("cite", remap[old_id], clean))
            n_citations += 1

    
        docs = make_documents(
            [{"id": remap[str(d["id"])], "content": d["content"]} for d in documents]
        )
        doc_order = sorted(d.document_id for d in docs)

        answer = fmt.assemble(pieces, doc_order).strip()
        reference_answer = (
            reference_fmt.assemble(pieces, doc_order).strip() if use_reference else ""
        )
        if n_citations == 0:
            stats["records_without_citations"] += 1
            if not keep_uncited:
                continue

        if not docs:
            stats["records_without_documents"] += 1
            continue

        prompt = build_prompt(record["question"], docs, config)
        stats["records_kept"] += 1
        stats["citations_kept"] += n_citations
        examples.append(
            SFTExample(
                question=record["question"],
                prompt=prompt,
                answer=answer,
                documents=[{"id": d.document_id, "content": d.text} for d in docs],
                num_citations=n_citations,
                citation_format=fmt.name,
                length_reference_answer=reference_answer,
                length_reference_prompt=(
                    build_prompt(record["question"], docs, reference_config)
                    if use_reference else ""
                ),
            )
        )

    return examples



def encode_example(
    tokenizer,
    example: SFTExample,
    max_seq_len: int,
    citation_weight: float,
) -> Optional[Dict[str, List]]:
    """Tokenize one example into ids, labels and per-token loss weights.
    """
    head = chat_head(tokenizer, example.prompt)
    tail = chat_tail(tokenizer, example.prompt, example.answer)

    head_ids = tokenizer(
        head, add_special_tokens=not tokenizer.chat_template
    )["input_ids"]
    encoded = tokenizer(tail, add_special_tokens=False, return_offsets_mapping=True)
    tail_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    own_len = len(head_ids) + len(tail_ids)
    fit_len = own_len
    if example.length_reference_prompt or example.length_reference_answer:
        reference_prompt = example.length_reference_prompt or example.prompt
        reference_answer = example.length_reference_answer or example.answer
        reference_head = chat_head(tokenizer, reference_prompt)
        reference_tail = chat_tail(tokenizer, reference_prompt, reference_answer)
        fit_len = (
            len(tokenizer(
                reference_head, add_special_tokens=not tokenizer.chat_template
            )["input_ids"])
            + len(tokenizer(reference_tail, add_special_tokens=False)["input_ids"])
        )

    if max(own_len, fit_len) > max_seq_len:
        return None

    spans = get_format(example.citation_format).spans(example.answer)
    weights = [1.0] * len(tail_ids)
    n_citation_tokens = 0
    for i, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            weights[i] = citation_weight
            n_citation_tokens += 1

    return {
        "input_ids": head_ids + tail_ids,
        "labels": [-100] * len(head_ids) + list(tail_ids),
        "loss_weights": [0.0] * len(head_ids) + weights,
        # Bookkeeping only; SFTCollator does not forward it to the model.
        "num_citation_tokens": n_citation_tokens,
    }


class CitationSFTDataset(Dataset):
    """Pre-tokenized examples; short enough that encoding up front is cheapest."""

    def __init__(
        self,
        tokenizer,
        examples: Sequence[SFTExample],
        max_seq_len: int,
        citation_weight: float,
        stats: Optional[Counter] = None,
    ):
        stats = stats if stats is not None else Counter()
        self.features: List[Dict[str, List]] = []
        for example in examples:
            encoded = encode_example(tokenizer, example, max_seq_len, citation_weight)
            if encoded is None:
                stats["examples_too_long"] += 1
                continue
            self.features.append(encoded)
            stats["tokens"] += len(encoded["input_ids"])
            stats["answer_tokens"] += sum(w > 0 for w in encoded["loss_weights"])
            stats["citation_tokens"] += encoded["num_citation_tokens"]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List]:
        return self.features[index]


@dataclass
class SFTCollator:
    """Right-pads a batch and carries the loss weights alongside the labels."""

    pad_token_id: int

    def __call__(self, features: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        batch = {
            "input_ids": [], "attention_mask": [], "labels": [], "loss_weights": [],
        }
        for feature in features:
            pad = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append([1] * len(feature["input_ids"]) + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
            batch["loss_weights"].append(feature["loss_weights"] + [0.0] * pad)

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float),
        }
