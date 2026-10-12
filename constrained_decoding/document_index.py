import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Iterator
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a single document with ID and content.

    Attributes:
        document_id: Unique identifier for the document
        text: Full text content of the document
        metadata: Optional metadata dictionary
    """
    document_id: str
    text: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate document on creation."""
        if not self.document_id:
            raise ValueError("document_id cannot be empty")
        if not self.text:
            raise ValueError("text cannot be empty")

    def __hash__(self):
        return hash(self.document_id)

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.document_id == other.document_id


class DocumentIndex:
    """
    Efficient index for document retrieval and snippet validation.

    This class provides:
    - Fast document ID lookup
    - Substring matching for snippet validation
    - Precomputed valid snippet positions

    The index is optimized for the constrained decoding use case where
    we need to quickly validate if a snippet exists in a document.
    """

    def __init__(
        self,
        documents: List[Document],
        min_snippet_length: int = 10,
        max_snippet_length: int = 500,
        precompute_snippets: bool = False,
        snippet_step: int = 1,
    ):
        """
        Initialize the document index.

        Args:
            documents: List of Document objects to index
            min_snippet_length: Minimum allowed snippet length in characters
            max_snippet_length: Maximum allowed snippet length in characters
            precompute_snippets: Whether to precompute all valid snippets
            snippet_step: Step size for precomputing snippets (1 = all positions)
        """
        self.min_snippet_length = min_snippet_length
        self.max_snippet_length = max_snippet_length
        self.snippet_step = snippet_step

        # Document storage
        self._documents: Dict[str, Document] = {}
        self._document_ids: Set[str] = set()

        # Snippet index for fast validation
        self._snippet_hashes: Dict[str, Set[str]] = {}  # hash -> set of doc_ids
        self._precomputed_snippets: Dict[str, Set[str]] = {}  # doc_id -> set of snippet hashes

        # Index all documents
        for doc in documents:
            self._add_document(doc)

        # Optionally precompute all valid snippets
        if precompute_snippets:
            self._precompute_all_snippets()

        logger.info(f"DocumentIndex initialized with {len(self._documents)} documents")

    def _add_document(self, document: Document) -> None:
        """Add a document to the index."""
        if document.document_id in self._documents:
            logger.warning(f"Document {document.document_id} already exists, updating")

        self._documents[document.document_id] = document
        self._document_ids.add(document.document_id)

    def _precompute_all_snippets(self) -> None:
        """
        Precompute all valid snippet hashes for each document.

        This enables O(1) snippet validation but uses more memory.
        Only recommended for smaller document sets or when using
        larger snippet_step values.
        """
        logger.info("Precomputing snippet hashes...")

        for doc_id, doc in self._documents.items():
            text = doc.text
            text_len = len(text)
            self._precomputed_snippets[doc_id] = set()

            # Generate snippets of various lengths
            for start in range(0, text_len, self.snippet_step):
                for length in range(self.min_snippet_length,
                                   min(self.max_snippet_length + 1, text_len - start + 1)):
                    snippet = text[start:start + length]
                    snippet_hash = self._hash_snippet(snippet)
                    self._precomputed_snippets[doc_id].add(snippet_hash)

                    # Also store in global hash index
                    if snippet_hash not in self._snippet_hashes:
                        self._snippet_hashes[snippet_hash] = set()
                    self._snippet_hashes[snippet_hash].add(doc_id)

        total_snippets = sum(len(s) for s in self._precomputed_snippets.values())
        logger.info(f"Precomputed {total_snippets} snippet hashes")

    @staticmethod
    def _hash_snippet(snippet: str) -> str:
        """Create a hash of a snippet for fast lookup."""
        return hashlib.md5(snippet.encode('utf-8')).hexdigest()

    def is_valid_document_id(self, document_id: str) -> bool:
        """
        Check if a document ID exists in the index.

        Args:
            document_id: The document ID to check
        """
        return document_id in self._document_ids

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: The document ID to retrieve

        Returns:
            The Document object if found, None otherwise
        """
        return self._documents.get(document_id)

    def get_all_document_ids(self) -> Set[str]:
        """Get all valid document IDs."""
        return self._document_ids.copy()

    def is_valid_snippet(
        self,
        document_id: str,
        snippet: str,
        case_sensitive: bool = False
    ) -> bool:
        """
        Check if a snippet exists in a document.

        Args:
            document_id: The document ID to check against
            snippet: The text snippet to validate
            case_sensitive: If True, match exactly; if False, use case-insensitive matching

        Returns:
            True if the snippet exists in the document, False otherwise
        """
        # First check if document exists
        if document_id not in self._documents:
            return False

        # Check snippet length constraints
        snippet_len = len(snippet)
        if snippet_len < self.min_snippet_length or snippet_len > self.max_snippet_length:
            return False

        # If we have precomputed snippets, use hash lookup (case-sensitive only)
        if self._precomputed_snippets and case_sensitive:
            snippet_hash = self._hash_snippet(snippet)
            return snippet_hash in self._precomputed_snippets.get(document_id, set())

        # Do direct substring search
        doc = self._documents[document_id]

        if case_sensitive:
            return snippet in doc.text
        else:
            # Case-insensitive matching: find if the snippet exists in the document
            # regardless of case, and return True if found
            return snippet.lower() in doc.text.lower()

    def find_snippet_position(
        self,
        document_id: str,
        snippet: str,
        case_sensitive: bool = False
    ) -> Optional[Tuple[int, int]]:
        """
        Find the position of a snippet in a document.

        Args:
            document_id: The document ID to search in
            snippet: The text snippet to find
            case_sensitive: If True, match exactly; if False, use case-insensitive matching

        Returns:
            Tuple of (start, end) positions if found, None otherwise
        """
        if document_id not in self._documents:
            return None

        doc = self._documents[document_id]

        if case_sensitive:
            pos = doc.text.find(snippet)
        else:
            pos = doc.text.lower().find(snippet.lower())

        if pos == -1:
            return None

        return (pos, pos + len(snippet))

    def get_corrected_snippet(
        self,
        document_id: str,
        snippet: str,
        case_sensitive: bool = False
    ) -> Optional[str]:
        """
        Get the actual snippet text from the document that matches the input.

        This is useful when case-insensitive matching is used, to get the
        exact text as it appears in the document.

        Args:
            document_id: The document ID to search in
            snippet: The text snippet to find
            case_sensitive: If True, match exactly; if False, use case-insensitive matching

        Returns:
            The actual snippet text from the document, or None if not found
        """
        position = self.find_snippet_position(document_id, snippet, case_sensitive)
        if position is None:
            return None

        doc = self._documents[document_id]
        return doc.text[position[0]:position[1]]

    def get_valid_snippets_at_position(
        self,
        document_id: str,
        position: int,
        max_count: int = 100
    ) -> List[str]:
        """
        Get all valid snippets starting at a specific position.

        This is useful for constrained decoding when we need to know
        what snippets are valid continuations.

        Args:
            document_id: The document ID
            position: Starting position in the document
            max_count: Maximum number of snippets to return

        Returns:
            List of valid snippets starting at the position
        """
        if document_id not in self._documents:
            return []

        doc = self._documents[document_id]
        text = doc.text
        text_len = len(text)

        if position >= text_len:
            return []

        snippets = []
        for length in range(self.min_snippet_length,
                          min(self.max_snippet_length + 1, text_len - position + 1)):
            snippet = text[position:position + length]
            snippets.append(snippet)
            if len(snippets) >= max_count:
                break

        return snippets

    def find_documents_containing_snippet(self, snippet: str) -> Set[str]:
        """
        Find all documents that contain a given snippet.

        Args:
            snippet: The text snippet to search for

        Returns:
            Set of document IDs that contain the snippet
        """
        matching_docs = set()

        # If we have precomputed hashes, use them
        if self._snippet_hashes:
            snippet_hash = self._hash_snippet(snippet)
            return self._snippet_hashes.get(snippet_hash, set()).copy()

        # Otherwise, search all documents
        for doc_id, doc in self._documents.items():
            if snippet in doc.text:
                matching_docs.add(doc_id)

        return matching_docs

    def iterate_all_snippets(
        self,
        document_id: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Iterator[Tuple[str, int, int]]:
        """
        Iterate over all valid snippets in a document.

        Args:
            document_id: The document ID
            min_length: Minimum snippet length (uses index default if None)
            max_length: Maximum snippet length (uses index default if None)

        Yields:
            Tuples of (snippet, start_pos, end_pos)
        """
        if document_id not in self._documents:
            return

        doc = self._documents[document_id]
        text = doc.text
        text_len = len(text)

        min_len = min_length or self.min_snippet_length
        max_len = max_length or self.max_snippet_length

        for start in range(text_len):
            for length in range(min_len, min(max_len + 1, text_len - start + 1)):
                snippet = text[start:start + length]
                yield (snippet, start, start + length)

    def get_document_text(self, document_id: str) -> Optional[str]:
        """Get the full text of a document."""
        doc = self._documents.get(document_id)
        return doc.text if doc else None

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self._documents)

    def __contains__(self, document_id: str) -> bool:
        """Check if a document ID is in the index."""
        return document_id in self._document_ids

    @classmethod
    def from_dict_list(
        cls,
        documents: List[Dict[str, str]],
        id_key: str = "document_id",
        text_key: str = "text",
        **kwargs
    ) -> "DocumentIndex":
        """
        Create a DocumentIndex from a list of dictionaries.

        Args:
            documents: List of dictionaries with document data
            id_key: Key for document ID in dictionaries
            text_key: Key for document text in dictionaries
            **kwargs: Additional arguments passed to __init__

        Returns:
            DocumentIndex instance
        """
        doc_objects = []
        for doc_dict in documents:
            doc = Document(
                document_id=doc_dict[id_key],
                text=doc_dict[text_key],
                metadata={k: v for k, v in doc_dict.items() if k not in [id_key, text_key]}
            )
            doc_objects.append(doc)

        return cls(doc_objects, **kwargs)
