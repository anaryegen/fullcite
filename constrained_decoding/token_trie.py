import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)


@dataclass
class TokenTrieNode:
    """
    One node in the token-level trie.

    children:       token_id → child node
    is_end:         True if a valid snippet ends here
    document_ids:   doc IDs for snippets ending at this node
    reachable_docs: union of all doc IDs reachable from this node (including here).
                    Precomputed during insert so get_valid_next_token_ids() is O(1).
    """
    children: Dict[int, "TokenTrieNode"] = field(default_factory=dict)
    is_end: bool = False
    document_ids: Set[str] = field(default_factory=set)
    reachable_docs: Set[str] = field(default_factory=set)


class TokenLevelSnippetTrie:
    """
    Token-ID indexed prefix trie for snippet-constrained decoding.

    Each root-to-node path represents a token ID sequence. Nodes marked
    `is_end` are complete valid snippets. Every node also carries
    `reachable_docs` — the union of all doc IDs reachable from it — so
    filtering by document is a single set-membership test per child.

    """

    def __init__(self) -> None:
        self.root = TokenTrieNode()
        self._total_sequences = 0

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def insert(self, token_ids: List[int], doc_id: str) -> None:
        """Insert a token sequence with its associated document ID."""
        node = self.root
        node.reachable_docs.add(doc_id)
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TokenTrieNode()
            node = node.children[tid]
            node.reachable_docs.add(doc_id)
        node.is_end = True
        node.document_ids.add(doc_id)
        self._total_sequences += 1

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_node(self, token_ids: List[int]) -> Optional[TokenTrieNode]:
        node = self.root
        for tid in token_ids:
            node = node.children.get(tid)
            if node is None:
                return None
        return node

    def get_valid_next_token_ids(
        self,
        token_prefix: List[int],
        doc_id: Optional[str] = None,
    ) -> Set[int]:
        """
        Return token IDs that can legally extend token_prefix.

        If doc_id is given, only returns tokens that lead to at least one
        snippet belonging to that document.
        """
        node = self.get_node(token_prefix)
        if node is None:
            return set()
        if doc_id is None:
            return set(node.children.keys())
        return {
            tid
            for tid, child in node.children.items()
            if doc_id in child.reachable_docs
        }

    def can_end(
        self, token_prefix: List[int], doc_id: Optional[str] = None
    ) -> bool:
        """Return True if token_prefix is itself a complete valid snippet."""
        node = self.get_node(token_prefix)
        if node is None or not node.is_end:
            return False
        return doc_id is None or doc_id in node.document_ids

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_sequences": self._total_sequences,
            "root_children": len(self.root.children),
        }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def build_from_documents(
        cls,
        documents: Any,           # List[Document]
        tokenizer: Any,
        min_chars: int = 20,
        max_chars: int = 512,
        token_step: int = 1,
    ) -> "TokenLevelSnippetTrie":
        """
        Build a TokenLevelSnippetTrie from a list of Document objects.

        For each document the full text is tokenized once. Every contiguous
        token subsequence whose decoded character length falls in
        [min_chars, max_chars] is inserted into the trie.

        Args:
            documents:   List of Document objects (need .text and .document_id).
            tokenizer:   HuggingFace tokenizer.
            min_chars:   Minimum character length of a valid snippet.
            max_chars:   Maximum character length of a valid snippet.
            token_step:  Stride for start positions (in tokens). 1 = full
                         token-aligned coverage. Higher values trade recall
                         for build speed on very long documents.
        """
        trie = cls()
        total_inserted = 0

        for doc in documents:
            # Tokenize without special tokens so indices map cleanly to text.
            doc_tokens: List[int] = tokenizer.encode(
                doc.text, add_special_tokens=False
            )
            n = len(doc_tokens)

            for i in range(0, n, token_step):
                for j in range(i + 1, n + 1):
                    snippet_tokens = doc_tokens[i:j]
                    snippet_text = tokenizer.decode(snippet_tokens)
                    char_len = len(snippet_text)

                    if char_len < min_chars:
                        continue        # keep extending j
                    if char_len > max_chars:
                        break           # further extension only makes it longer

                    trie.insert(snippet_tokens, doc.document_id)
                    total_inserted += 1

            logger.debug(
                f"Indexed document {doc.document_id} "
                f"({n} tokens, {len(doc.text)} chars)"
            )

        logger.info(
            f"TokenLevelSnippetTrie: {total_inserted} sequences "
            f"from {len(documents)} documents, "
            f"{len(trie.root.children)} distinct first-tokens"
        )
        return trie
