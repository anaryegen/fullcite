import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constrained_decoding import Document, DocumentIndex  # noqa: E402
from constrained_decoding.citation_verifier import CitationVerifier  # noqa: E402
from constrained_decoding.constrained_decoder import (  # noqa: E402
    ConstrainedDecoder,
    ConstrainedDecodingConfig,
)


CITATION_RE = CitationVerifier.CITATION_PATTERN

QUOTE = '"'


def render_citation(document_id: str, snippet: str) -> str:
    """Render one citation in the canonical format."""
    return '[doc:%s snippet:"%s"]' % (document_id, snippet)


def citation_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of every well-formed citation in `text`."""
    return [match.span() for match in CITATION_RE.finditer(text)]


class _OrderedIndex:
    """Document index proxy that hands out document ids in a stable order."""

    def __init__(self, index: DocumentIndex):
        self._index = index

    def get_all_document_ids(self) -> List[str]:
        return sorted(self._index.get_all_document_ids())

    def get_document(self, document_id: str):
        return self._index.get_document(document_id)

    def __len__(self) -> int:
        return len(self._index)


class _PromptContext:

    build_constrained_prompt = ConstrainedDecoder.build_constrained_prompt
    _format_documents = ConstrainedDecoder._format_documents
    _get_format_instructions = ConstrainedDecoder._get_format_instructions

    def __init__(self, document_index, config: ConstrainedDecodingConfig):
        self.document_index = _OrderedIndex(document_index)
        self.config = config


def make_documents(documents: Sequence[Dict[str, Any]]) -> List[Document]:
    """Build Document objects from `parsed_documents` entries.
    """
    return [
        Document(document_id=str(doc["id"]), text=doc["content"])
        for doc in documents
        if str(doc.get("id", "")).strip() and doc.get("content")
    ]


def build_document_index(
    documents: Sequence[Document],
    config: ConstrainedDecodingConfig,
) -> DocumentIndex:
    return DocumentIndex(
        documents=list(documents),
        min_snippet_length=config.min_snippet_length,
        max_snippet_length=config.max_snippet_length,
        precompute_snippets=False,
    )


def build_prompt(
    question: str,
    documents: Sequence[Document],
    config: ConstrainedDecodingConfig,
    document_index: DocumentIndex = None,
) -> str:
    """The task prompt, identical for training and for all three decoding modes."""
    if document_index is None:
        document_index = build_document_index(documents, config)
    context = _PromptContext(document_index, config)
    return context.build_constrained_prompt(question, include_format_instructions=True)

def supports_thinking_toggle(tokenizer) -> bool:
    """True for templates with a thinking mode that has to be switched off."""
    return "enable_thinking" in (tokenizer.chat_template or "")


def _template_kwargs(tokenizer) -> Dict[str, Any]:
    return {"enable_thinking": False} if supports_thinking_toggle(tokenizer) else {}


def chat_head(tokenizer, prompt: str) -> str:
    
    if not tokenizer.chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **_template_kwargs(tokenizer),
    )


def chat_tail(tokenizer, prompt: str, answer: str) -> str:
    
    if not tokenizer.chat_template:
        return answer + (tokenizer.eos_token or "")

    kwargs = _template_kwargs(tokenizer)
    head = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        **kwargs,
    )
    if full.startswith(head):
        return full[len(head):]


    return answer + (tokenizer.eos_token or "")


def turn_end_ids(tokenizer) -> List[int]:
    
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    control = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    tail = chat_tail(tokenizer, "Q", "A")
    for token_id in tokenizer(tail, add_special_tokens=False)["input_ids"]:
        if token_id not in control or token_id in ids:
            continue
        if not tokenizer.decode([token_id]).strip():
            continue
        ids.append(token_id)
    return ids


def iter_documents_for_prompt(document_index: DocumentIndex) -> Iterable[str]:
    return sorted(document_index.get_all_document_ids())
