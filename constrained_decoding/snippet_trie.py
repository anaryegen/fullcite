import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrieNode:
    """
    Node in the snippet trie.

    Attributes:
        children: Mapping from character to child nodes
        is_end: Whether this node marks the end of a valid snippet
        document_ids: Set of document IDs that contain snippets ending here
        snippet_count: Number of snippets that pass through this node
    """
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False
    document_ids: Set[str] = field(default_factory=set)
    snippet_count: int = 0


class SnippetTrie:
    """
    Trie data structure for efficient snippet prefix matching.

    This trie is used during constrained decoding to:
    1. Validate that partial snippets can lead to complete valid snippets
    2. Get all valid next characters given a prefix
    3. Check if a complete snippet is valid for a given document

    The trie supports document-specific constraints, allowing validation
    that a snippet belongs to a specific document.
    """

    def __init__(self):
        """Initialize an empty trie."""
        self.root = TrieNode()
        self._total_snippets = 0
        self._total_nodes = 1  # Count root

    def insert(self, snippet: str, document_id: str) -> None:
        """
        Insert a snippet into the trie.

        Args:
            snippet: The snippet text to insert
            document_id: The document ID this snippet belongs to
        """
        if not snippet:
            return

        node = self.root
        for char in snippet:
            if char not in node.children:
                node.children[char] = TrieNode()
                self._total_nodes += 1
            node = node.children[char]
            node.snippet_count += 1

        node.is_end = True
        node.document_ids.add(document_id)
        self._total_snippets += 1

    def insert_all_substrings(
        self,
        text: str,
        document_id: str,
        min_length: int = 10,
        max_length: int = 500,
        step: int = 1
    ) -> int:
        """
        Insert all valid substrings of a text into the trie.

        Args:
            text: The source text
            document_id: The document ID
            min_length: Minimum snippet length
            max_length: Maximum snippet length
            step: Step size for starting positions (1 = all positions)

        Returns:
            Number of snippets inserted
        """
        text_len = len(text)
        count = 0

        for start in range(0, text_len, step):
            for length in range(min_length, min(max_length + 1, text_len - start + 1)):
                snippet = text[start:start + length]
                self.insert(snippet, document_id)
                count += 1

        return count

    def search(self, snippet: str) -> Optional[Set[str]]:
        """
        Search for a complete snippet in the trie.

        Args:
            snippet: The snippet to search for

        Returns:
            Set of document IDs containing this snippet, or None if not found
        """
        node = self._find_node(snippet)
        if node is None or not node.is_end:
            return None
        return node.document_ids.copy()

    def has_prefix(self, prefix: str) -> bool:
        """
        Check if any snippet starts with the given prefix.

        Args:
            prefix: The prefix to check

        Returns:
            True if any snippet has this prefix
        """
        return self._find_node(prefix) is not None

    def has_prefix_for_document(self, prefix: str, document_id: str) -> bool:
        """
        Check if any snippet for a specific document starts with the prefix.

        Args:
            prefix: The prefix to check
            document_id: The document ID to check against

        Returns:
            True if any snippet for the document has this prefix
        """
        node = self._find_node(prefix)
        if node is None:
            return False

        # Check if this node or any descendant has the document_id
        return self._has_document_in_subtree(node, document_id)

    def _has_document_in_subtree(self, node: TrieNode, document_id: str) -> bool:
        """Recursively check if document_id exists in subtree."""
        if document_id in node.document_ids:
            return True

        for child in node.children.values():
            if self._has_document_in_subtree(child, document_id):
                return True

        return False

    def get_valid_next_chars(self, prefix: str) -> Set[str]:
        """
        Get all valid next characters given a prefix.

        Args:
            prefix: The current prefix

        Returns:
            Set of characters that can follow the prefix
        """
        node = self._find_node(prefix)
        if node is None:
            return set()
        return set(node.children.keys())

    def get_valid_next_chars_for_document(
        self,
        prefix: str,
        document_id: str
    ) -> Set[str]:
        """
        Get valid next characters for a specific document.

        Args:
            prefix: The current prefix
            document_id: The document ID to constrain to

        Returns:
            Set of characters that can follow the prefix for this document
        """
        node = self._find_node(prefix)
        if node is None:
            return set()

        valid_chars = set()
        for char, child in node.children.items():
            if self._has_document_in_subtree(child, document_id):
                valid_chars.add(char)

        return valid_chars

    def can_complete(self, prefix: str) -> bool:
        """
        Check if the prefix can be completed to a valid snippet.

        Args:
            prefix: The current prefix

        Returns:
            True if there exists a valid completion
        """
        node = self._find_node(prefix)
        if node is None:
            return False

        # Check if this node is an end or has children
        return node.is_end or len(node.children) > 0

    def is_complete_snippet(self, text: str) -> bool:
        """
        Check if the text is a complete valid snippet.

        Args:
            text: The text to check

        Returns:
            True if this is a complete valid snippet
        """
        node = self._find_node(text)
        return node is not None and node.is_end

    def is_valid_snippet_for_document(self, snippet: str, document_id: str) -> bool:
        """
        Check if a snippet is valid for a specific document.

        Args:
            snippet: The snippet to validate
            document_id: The document ID to check against

        Returns:
            True if the snippet is valid for the document
        """
        node = self._find_node(snippet)
        if node is None or not node.is_end:
            return False
        return document_id in node.document_ids

    def get_completions(
        self,
        prefix: str,
        max_completions: int = 100,
        document_id: Optional[str] = None
    ) -> List[str]:
        """
        Get all valid completions for a prefix.

        Args:
            prefix: The prefix to complete
            max_completions: Maximum number of completions to return
            document_id: Optional document ID to filter by

        Returns:
            List of complete snippets that start with the prefix
        """
        node = self._find_node(prefix)
        if node is None:
            return []

        completions = []
        self._collect_completions(
            node, prefix, completions, max_completions, document_id
        )
        return completions

    def get_completions_for_document(
        self,
        prefix: str,
        document_id: str,
        max_completions: int = 100
    ) -> List[str]:
        """
        Get all valid completions for a prefix filtered by document ID.

        Args:
            prefix: The prefix to complete
            document_id: The document ID to filter by
            max_completions: Maximum number of completions to return

        Returns:
            List of complete snippets that start with the prefix for the given document
        """
        return self.get_completions(prefix, max_completions, document_id)

    def _collect_completions(
        self,
        node: TrieNode,
        current: str,
        completions: List[str],
        max_completions: int,
        document_id: Optional[str]
    ) -> None:
        """Recursively collect completions from a node."""
        if len(completions) >= max_completions:
            return

        if node.is_end:
            if document_id is None or document_id in node.document_ids:
                completions.append(current)

        for char, child in node.children.items():
            if len(completions) >= max_completions:
                break
            self._collect_completions(
                child, current + char, completions, max_completions, document_id
            )

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node corresponding to a prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the trie."""
        return {
            "total_snippets": self._total_snippets,
            "total_nodes": self._total_nodes,
        }

    @classmethod
    def from_documents(
        cls,
        documents: List[Tuple[str, str]],
        min_length: int = 10,
        max_length: int = 500,
        step: int = 1
    ) -> "SnippetTrie":
        """
        Build a trie from a list of documents.

        Args:
            documents: List of (document_id, text) tuples
            min_length: Minimum snippet length
            max_length: Maximum snippet length
            step: Step size for substring extraction

        Returns:
            SnippetTrie instance
        """
        trie = cls()
        total_inserted = 0

        for doc_id, text in documents:
            count = trie.insert_all_substrings(
                text, doc_id, min_length, max_length, step
            )
            total_inserted += count
            logger.debug(f"Inserted {count} snippets from document {doc_id}")

        logger.info(f"Built trie with {total_inserted} total snippets")
        return trie


class TokenAwareSnippetTrie(SnippetTrie):
    """
    Extension of SnippetTrie that is aware of tokenization.

    This class bridges the gap between character-level snippet matching
    and token-level decoding constraints by maintaining mappings between
    tokens and their character representations.
    """

    def __init__(self, tokenizer=None, token_vocab: Optional[Dict[int, str]] = None):
        """
        Initialize a token-aware trie.

        Args:
            tokenizer: HuggingFace tokenizer for token-to-text mapping
            token_vocab: Pre-built {token_id: text} mapping. If provided,
                         the tokenizer vocab scan is skipped entirely.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self._token_to_text: Dict[int, str] = {}

        if token_vocab is not None:
            self._token_to_text = token_vocab
            logger.info(f"Using pre-built token mapping: {len(token_vocab)} tokens")
        elif tokenizer:
            self._build_token_mapping()

    def _build_token_mapping(self) -> None:
        """Build mapping from token IDs to their text representations."""
        if not self.tokenizer:
            return

        vocab_size = self.tokenizer.vocab_size
        for token_id in range(vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
                self._token_to_text[token_id] = text
            except Exception:
                continue

        logger.info(f"Built token mapping for {len(self._token_to_text)} tokens")

    def get_valid_token_ids(
        self,
        prefix: str,
        document_id: Optional[str] = None
    ) -> Set[int]:
        """
        Get all valid token IDs that can follow a prefix.

        Args:
            prefix: The current character-level prefix
            document_id: Optional document ID constraint

        Returns:
            Set of valid token IDs
        """
        if not self.tokenizer:
            return set()

        # Get valid next characters
        if document_id:
            valid_chars = self.get_valid_next_chars_for_document(prefix, document_id)
        else:
            valid_chars = self.get_valid_next_chars(prefix)

        if not valid_chars:
            return set()

        # Find tokens that start with valid characters
        valid_tokens = set()
        for token_id, token_text in self._token_to_text.items():
            if not token_text:
                continue

            # Check if token text starts with a valid character
            first_char = token_text[0] if token_text else ""
            if first_char in valid_chars:
                # Verify the full token is a valid continuation
                if self.has_prefix(prefix + token_text):
                    if document_id is None or self.has_prefix_for_document(
                        prefix + token_text, document_id
                    ):
                        valid_tokens.add(token_id)

        return valid_tokens
