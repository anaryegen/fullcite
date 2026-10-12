import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum, auto

from .citation_format import (
    DEFAULT_FORMAT,
    REPRESENTATIONS,
    CitationFormat,
    get_format,
)
from .document_index import DocumentIndex

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of citation verification."""
    VALID = auto()
    INVALID_DOCUMENT_ID = auto()
    INVALID_SNIPPET = auto()
    MALFORMED_SYNTAX = auto()
    SNIPPET_TOO_SHORT = auto()
    SNIPPET_TOO_LONG = auto()


@dataclass
class CitationInfo:
    """
    Information about a single citation.

    Attributes:
        document_id: The cited document ID
        snippet: The cited snippet text
        start_pos: Start position in the generated text
        end_pos: End position in the generated text
        status: Verification status
        error_message: Optional error description
    """
    document_id: str
    snippet: str
    start_pos: int
    end_pos: int
    status: VerificationStatus = VerificationStatus.VALID
    error_message: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status == VerificationStatus.VALID


@dataclass
class VerificationResult:
    """
    Result of verifying citations in generated text.

    Attributes:
        is_valid: Whether all citations are valid
        citations: List of CitationInfo for each citation found
        errors: List of error messages
        warnings: List of warning messages
        stats: Dictionary of verification statistics
    """
    is_valid: bool
    citations: List[CitationInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"VerificationResult({status}, "
            f"{len(self.citations)} citations, "
            f"{len(self.errors)} errors)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "citations": [
                {
                    "document_id": c.document_id,
                    "snippet": c.snippet,
                    "start_pos": c.start_pos,
                    "end_pos": c.end_pos,
                    "status": c.status.name,
                    "is_valid": c.is_valid,
                    "error_message": c.error_message
                }
                for c in self.citations
            ],
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats
        }


class CitationVerifier:
    """
    Verifies citations in generated text.

    This class provides comprehensive verification including:
    - Syntax validation
    - Document ID validation
    - Snippet existence validation
    - Optional snippet similarity checking

    Example usage:
        ```python
        verifier = CitationVerifier(document_index)
        result = verifier.verify(generated_text)

        if result.is_valid:
            print("All citations valid!")
        else:
            for error in result.errors:
                print(f"Error: {error}")
        ```
    """

    # The default representation's surface form, kept as class attributes because
    # sft/prompts.py and grpo/citations.py read them off this class by name in
    # order to guarantee the trainer and the reward can only accept what the
    # evaluator accepts.  These are the *same objects* the registry holds, so
    # that guarantee now spans the registry too rather than being duplicated.
    CITATION_PATTERN = REPRESENTATIONS[DEFAULT_FORMAT].pattern
    MALFORMED_PATTERNS = list(REPRESENTATIONS[DEFAULT_FORMAT].malformed)

    def __init__(
        self,
        document_index: DocumentIndex,
        min_snippet_length: int = 10,
        max_snippet_length: int = 500,
        check_similar_snippets: bool = False,
        similarity_threshold: float = 1.0
    ):
        """
        Initialize the citation verifier.

        Args:
            document_index: Index of valid documents
            min_snippet_length: Minimum allowed snippet length
            max_snippet_length: Maximum allowed snippet length
            check_similar_snippets: Check for similar snippets on failure
            similarity_threshold: Threshold for snippet similarity
        """
        self.document_index = document_index
        self.min_snippet_length = min_snippet_length
        self.max_snippet_length = max_snippet_length
        self.check_similar_snippets = check_similar_snippets
        self.similarity_threshold = similarity_threshold

        logger.info(f"CitationVerifier initialized with {len(document_index)} documents")

    def verify(self, text: str) -> VerificationResult:
        """
        Verify all citations in the given text.

        Args:
            text: The generated text containing citations

        Returns:
            VerificationResult with validation details
        """
        citations = []
        errors = []
        warnings = []

        # Find all valid-format citations
        for match in self.CITATION_PATTERN.finditer(text):
            doc_id = match.group(1)
            snippet = match.group(2)
            start_pos = match.start()
            end_pos = match.end()

            # Validate this citation
            citation_info = self._verify_citation(doc_id, snippet, start_pos, end_pos)
            citations.append(citation_info)

            if not citation_info.is_valid:
                errors.append(citation_info.error_message)

        # Check for malformed citations
        malformed = self._find_malformed_citations(text)
        for pattern_name, positions in malformed.items():
            for pos in positions:
                errors.append(f"Malformed citation at position {pos}: {pattern_name}")

        # Calculate statistics
        stats = self._calculate_stats(citations, text)

        # Generate warnings
        if stats['total_citations'] == 0:
            warnings.append("No citations found in text")

        if stats.get('unique_documents', 0) < len(self.document_index) * 0.1:
            warnings.append("Citations reference very few of the available documents")

        is_valid = len(errors) == 0 and all(c.is_valid for c in citations)

        return VerificationResult(
            is_valid=is_valid,
            citations=citations,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    def _verify_citation(
        self,
        doc_id: str,
        snippet: str,
        start_pos: int,
        end_pos: int
    ) -> CitationInfo:
        """Verify a single citation."""
        # Check document ID
        if not self.document_index.is_valid_document_id(doc_id):
            return CitationInfo(
                document_id=doc_id,
                snippet=snippet,
                start_pos=start_pos,
                end_pos=end_pos,
                status=VerificationStatus.INVALID_DOCUMENT_ID,
                error_message=f"Document ID '{doc_id}' does not exist"
            )

        # Check snippet length
        if len(snippet) < self.min_snippet_length:
            return CitationInfo(
                document_id=doc_id,
                snippet=snippet,
                start_pos=start_pos,
                end_pos=end_pos,
                status=VerificationStatus.SNIPPET_TOO_SHORT,
                error_message=(
                    f"Snippet too short ({len(snippet)} chars, "
                    f"minimum {self.min_snippet_length})"
                )
            )

        if len(snippet) > self.max_snippet_length:
            return CitationInfo(
                document_id=doc_id,
                snippet=snippet,
                start_pos=start_pos,
                end_pos=end_pos,
                status=VerificationStatus.SNIPPET_TOO_LONG,
                error_message=(
                    f"Snippet too long ({len(snippet)} chars, "
                    f"maximum {self.max_snippet_length})"
                )
            )

        # Check snippet exists in document (case-insensitive by default)
        if not self.document_index.is_valid_snippet(doc_id, snippet):
            error_msg = f"Snippet not found in document '{doc_id}'"

            # Optionally check for similar snippets
            if self.check_similar_snippets:
                similar = self._find_similar_snippet(doc_id, snippet)
                if similar:
                    error_msg += f". Did you mean: '{similar[:50]}...'?"

            return CitationInfo(
                document_id=doc_id,
                snippet=snippet,
                start_pos=start_pos,
                end_pos=end_pos,
                status=VerificationStatus.INVALID_SNIPPET,
                error_message=error_msg
            )

        # Citation is valid - get the corrected snippet from the document
        # (in case there was a case mismatch that we accepted)
        corrected_snippet = self.document_index.get_corrected_snippet(doc_id, snippet)
        if corrected_snippet is None:
            corrected_snippet = snippet  # Fallback to original

        return CitationInfo(
            document_id=doc_id,
            snippet=corrected_snippet,  # Use the corrected snippet from the document
            start_pos=start_pos,
            end_pos=end_pos,
            status=VerificationStatus.VALID
        )

    def _find_malformed_citations(self, text: str) -> Dict[str, List[int]]:
        """Find malformed citation attempts in text."""
        malformed = {}

        # Remove valid citations first to avoid false positives
        text_without_valid = self.CITATION_PATTERN.sub("", text)

        for pattern, name in self.MALFORMED_PATTERNS:
            matches = pattern.finditer(text_without_valid)
            positions = [m.start() for m in matches]
            if positions:
                malformed[name] = positions

        return malformed

    def _find_similar_snippet(
        self,
        doc_id: str,
        snippet: str,
        max_candidates: int = 5
    ) -> Optional[str]:
        """
        Find a similar snippet in the document.

        Uses simple substring matching to find the closest match.
        """
        doc_text = self.document_index.get_document_text(doc_id)
        if not doc_text:
            return None

        # Try to find a substring that contains most of the snippet words
        snippet_words = set(snippet.lower().split())
        best_match = None
        best_score = 0

        # Sliding window approach
        window_size = len(snippet) + 50  # Allow some flexibility
        for i in range(0, len(doc_text) - window_size + 1, 10):
            window = doc_text[i:i + window_size]
            window_words = set(window.lower().split())

            # Calculate word overlap
            overlap = len(snippet_words & window_words)
            score = overlap / len(snippet_words) if snippet_words else 0

            if score > best_score:
                best_score = score
                best_match = window

        if best_score >= self.similarity_threshold:
            return best_match

        return None

    def _calculate_stats(
        self,
        citations: List[CitationInfo],
        text: str
    ) -> Dict[str, Any]:
        """Calculate verification statistics."""
        valid_citations = [c for c in citations if c.is_valid]
        invalid_citations = [c for c in citations if not c.is_valid]

        unique_docs = set(c.document_id for c in citations)
        unique_valid_docs = set(c.document_id for c in valid_citations)

        # Status breakdown
        status_counts = {}
        for c in citations:
            status_name = c.status.name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        return {
            "total_citations": len(citations),
            "valid_citations": len(valid_citations),
            "invalid_citations": len(invalid_citations),
            "unique_documents": len(unique_docs),
            "unique_valid_documents": len(unique_valid_docs),
            "text_length": len(text),
            "citation_density": len(citations) / max(len(text), 1) * 1000,
            "status_breakdown": status_counts
        }

    def verify_batch(self, texts: List[str]) -> List[VerificationResult]:
        """
        Verify citations in multiple texts.

        Args:
            texts: List of generated texts

        Returns:
            List of VerificationResult, one per text
        """
        return [self.verify(text) for text in texts]

    def get_document_coverage(self, text: str) -> Dict[str, List[str]]:
        """
        Get which documents are cited and with what snippets.

        Args:
            text: The generated text

        Returns:
            Dictionary mapping document IDs to lists of cited snippets
        """
        coverage = {}

        for match in self.CITATION_PATTERN.finditer(text):
            doc_id = match.group(1)
            snippet = match.group(2)

            if doc_id not in coverage:
                coverage[doc_id] = []
            coverage[doc_id].append(snippet)

        return coverage

    def extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citations from text as dictionaries.

        Args:
            text: The generated text

        Returns:
            List of citation dictionaries
        """
        citations = []

        for match in self.CITATION_PATTERN.finditer(text):
            citations.append({
                "document_id": match.group(1),
                "snippet": match.group(2),
                "full_match": match.group(0),
                "start": match.start(),
                "end": match.end()
            })

        return citations

    def format_report(self, result: VerificationResult) -> str:
        """
        Format a verification result as a human-readable report.

        Args:
            result: The verification result

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "CITATION VERIFICATION REPORT",
            "=" * 60,
            f"Overall Status: {'VALID' if result.is_valid else 'INVALID'}",
            "",
            "Statistics:",
            f"  Total Citations: {result.stats.get('total_citations', 0)}",
            f"  Valid Citations: {result.stats.get('valid_citations', 0)}",
            f"  Invalid Citations: {result.stats.get('invalid_citations', 0)}",
            f"  Unique Documents: {result.stats.get('unique_documents', 0)}",
            "",
        ]

        if result.errors:
            lines.append("Errors:")
            for error in result.errors:
                lines.append(f"  - {error}")
            lines.append("")

        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        lines.append("Citation Details:")
        for i, citation in enumerate(result.citations, 1):
            status = "✓" if citation.is_valid else "✗"
            lines.append(f"  {i}. [{status}] doc:{citation.document_id}")
            lines.append(f"      Snippet: \"{citation.snippet[:50]}...\"")
            if not citation.is_valid:
                lines.append(f"      Error: {citation.error_message}")

        lines.append("=" * 60)
        return "\n".join(lines)
