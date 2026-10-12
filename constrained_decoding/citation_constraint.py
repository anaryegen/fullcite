import logging
import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple, Any

from .citation_format import DEFAULT_FORMAT, CitationFormat, get_format
from .document_index import DocumentIndex
from .snippet_trie import SnippetTrie

logger = logging.getLogger(__name__)


class CitationState(Enum):
    """States in the citation parsing automaton."""
    FREE_TEXT = auto()          # Normal text, no citation in progress
    CITATION_START = auto()     # Saw '[', expecting 'd'
    DOC_KEYWORD = auto()        # Parsing 'doc:'
    DOCUMENT_ID = auto()        # Parsing document ID
    SPACE_AFTER_ID = auto()     # Expecting space after doc ID
    SNIPPET_KEYWORD = auto()    # Parsing 'snippet:'
    SNIPPET_QUOTE_START = auto() # Expecting opening quote
    SNIPPET_CONTENT = auto()    # Inside snippet content
    SNIPPET_QUOTE_END = auto()  # Expecting closing quote
    CITATION_END = auto()       # Expecting ']'
    ESCAPE_CHAR = auto()        # Processing escape character in snippet
    MARKER_INDEX = auto()       # Parsing the digits of a `[1]` style marker


@dataclass
class CitationParseState:
    """
    Tracks the current state of citation parsing.
    """
    state: CitationState = CitationState.FREE_TEXT
    partial_keyword: str = ""      # Partial match of 'doc:', 'snippet:', etc.
    current_doc_id: str = ""       # Document ID being parsed
    current_snippet: str = ""      # Snippet content being parsed
    escape_next: bool = False      # Next char is escaped
    bracket_depth: int = 0         # Track nested brackets (for edge cases)

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = CitationState.FREE_TEXT
        self.partial_keyword = ""
        self.current_doc_id = ""
        self.current_snippet = ""
        self.escape_next = False
        self.bracket_depth = 0

    def copy(self) -> "CitationParseState":
        """Create a copy of the current state."""
        return CitationParseState(
            state=self.state,
            partial_keyword=self.partial_keyword,
            current_doc_id=self.current_doc_id,
            current_snippet=self.current_snippet,
            escape_next=self.escape_next,
            bracket_depth=self.bracket_depth
        )


class CitationGrammar:
    """
    Formal grammar definition for the citation format.

    The grammar enforces:
    - Exact citation syntax: [doc:<id> snippet:"<text>"]
    - Valid document IDs from the document index
    - Verbatim snippets that exist in the referenced document
    """

    # Characters allowed in document IDs
    VALID_DOC_ID_CHARS = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-."
    )

    def __init__(
        self,
        document_index: DocumentIndex,
        snippet_trie: Optional[SnippetTrie] = None,
        allow_nested_quotes: bool = False,
        max_snippet_length: int = 500,
        citation_format: str = DEFAULT_FORMAT,
    ):
        """
        Initialize the citation grammar.

        Args:
            document_index: Index of valid documents
            snippet_trie: Optional trie for snippet validation
            allow_nested_quotes: Whether to allow escaped quotes in snippets
            max_snippet_length: Maximum allowed snippet length
            citation_format: Which citation representation to accept
        """
        self.document_index = document_index
        self.snippet_trie = snippet_trie
        self.allow_nested_quotes = allow_nested_quotes
        self.max_snippet_length = max_snippet_length
        self.format: CitationFormat = get_format(citation_format)

        # The keyword the automaton matches after '[', i.e. the representation's
        # opening literal minus the bracket itself.  None for `marker`, which
        # opens on a digit rather than a keyword.
        self._opening = (
            self.format.opening_literal[1:]
            if self.format.opening_literal else None
        )

        # Precompute valid document ID prefixes for efficiency
        self._doc_id_prefixes: Set[str] = set()
        for doc_id in document_index.get_all_document_ids():
            doc_id=str(doc_id)
            for i in range(1, len(doc_id) + 1):
                self._doc_id_prefixes.add(doc_id[:i])

        # Marker indices are positions in the prompt's document listing, so this
        # must be the same sorted order `_format_documents` and sft/prompts.py
        # emit, or the model would be shown one numbering and constrained to
        # another.
        self._doc_order: List[str] = sorted(
            str(d) for d in document_index.get_all_document_ids()
        )
        self._valid_markers: Set[str] = {
            str(i) for i in range(1, len(self._doc_order) + 1)
        }
        self._marker_prefixes: Set[str] = {
            marker[:i]
            for marker in self._valid_markers
            for i in range(1, len(marker) + 1)
        }

        logger.info(
            f"CitationGrammar initialized with {len(document_index)} documents "
            f"for citation format {self.format.name!r}"
        )

    def is_valid_doc_id_char(self, char: str) -> bool:
        """Check if a character is valid in a document ID."""
        return char in self.VALID_DOC_ID_CHARS

    def is_valid_doc_id_prefix(self, prefix: str) -> bool:
        """Check if a prefix could lead to a valid document ID."""
        return prefix in self._doc_id_prefixes

    def is_valid_doc_id(self, doc_id: str) -> bool:
        """Check if a complete document ID is valid."""
        return self.document_index.is_valid_document_id(doc_id)

    def is_valid_snippet_prefix(
        self,
        prefix: str,
        document_id: Optional[str]
    ) -> bool:
        """
        Check if a snippet prefix could lead to a valid snippet.

        Args:
            prefix: The current snippet content
            document_id: The document ID being referenced, or None to accept a
                prefix of any provided document.  `snippet_only` quotes without
                naming a document, so there is nothing to scope the lookup to
                until the quote is resolved after the fact.

        Returns:
            True if the prefix could be completed to a valid snippet
        """
        if not prefix:
            return True  # Empty prefix is always valid

        # Use trie if available for efficiency
        if self.snippet_trie:
            if document_id is None:
                return self.snippet_trie.has_prefix(prefix)
            return self.snippet_trie.has_prefix_for_document(prefix, document_id)

        # Fall back to document text search
        if document_id is None:
            return any(
                prefix in (self.document_index.get_document_text(doc_id) or "")
                for doc_id in self._doc_order
            )

        doc_text = self.document_index.get_document_text(document_id)
        if doc_text is None:
            return False

        return prefix in doc_text

    def is_valid_complete_snippet(
        self,
        snippet: str,
        document_id: Optional[str]
    ) -> bool:
        """
        Check if a complete snippet is valid for a document.

        Args:
            snippet: The complete snippet text
            document_id: The document ID being referenced, or None to accept a
                snippet verbatim in any provided document.

        Returns:
            True if the snippet exists verbatim in the document
        """
        if document_id is None:
            # Per-document so that the length bounds and the case-insensitive
            # match are applied exactly as they are for a named document; a bare
            # substring scan here would quietly admit snippets the verifier later
            # rejects for being too short.
            return any(
                self.document_index.is_valid_snippet(doc_id, snippet)
                for doc_id in self._doc_order
            )
        return self.document_index.is_valid_snippet(document_id, snippet)

    def get_valid_next_chars(
        self,
        parse_state: CitationParseState
    ) -> Set[str]:
        """
        Get all valid next characters given the current parse state.

        This is the core method for constrained decoding, determining
        what characters can follow the current generation state.

        Args:
            parse_state: Current parsing state

        Returns:
            Set of valid next characters
        """
        state = parse_state.state

        if state == CitationState.FREE_TEXT:
            # In free text, any character is valid
            # But '[' starts a potential citation
            return None  # None means all characters valid

        elif state == CitationState.CITATION_START:
            # After '[', the representation's opening keyword must follow.  A
            # marker has no keyword: it opens straight onto the digits of a
            # document position.
            if self._opening is None:
                return {
                    digit for digit in "0123456789"
                    if digit in self._marker_prefixes
                }
            pos = len(parse_state.partial_keyword)
            if pos < len(self._opening):
                return {self._opening[pos]}
            return set()

        elif state == CitationState.DOC_KEYWORD:
            # Parsing 'doc:', must match exactly
            pos = len(parse_state.partial_keyword)
            if pos < len("doc:"):
                return {"doc:"[pos]}
            return set()

        elif state == CitationState.DOCUMENT_ID:
            # Parsing document ID
            valid_chars = set()

            # Characters that continue the document ID
            current_id = parse_state.current_doc_id
            for doc_id in self.document_index.get_all_document_ids():
                if doc_id.startswith(current_id) and len(doc_id) > len(current_id):
                    valid_chars.add(doc_id[len(current_id)])

            if self.is_valid_doc_id(current_id):
                if self.format.carries_snippet:
                    # A space, then the snippet keyword.
                    valid_chars.add(' ')
                else:
                    # The citation is complete at the document id.  This single
                    # branch is the whole accepting path for `doc_only`; without
                    # it `]` is never a legal continuation and the arm cannot
                    # emit a citation at all.
                    valid_chars.add(']')

            return valid_chars

        elif state == CitationState.SPACE_AFTER_ID:
            # After document ID, must have ' snippet:"'
            pos = len(parse_state.partial_keyword)
            expected = ' snippet:"'
            if pos < len(expected):
                return {expected[pos]}
            return set()

        elif state == CitationState.SNIPPET_KEYWORD:
            # Parsing 'snippet:"'
            pos = len(parse_state.partial_keyword)
            expected = 'snippet:"'
            if pos < len(expected):
                return {expected[pos]}
            return set()

        elif state == CitationState.SNIPPET_CONTENT:
            # Inside snippet, constrained by document content
            if parse_state.escape_next:
                # After backslash, allow quote or backslash
                return {'"', '\\'}

            # Get valid next chars from trie or document
            valid_chars = set()

            # None means "any provided document", which is the right scope when
            # the representation quotes without naming one.
            scope = parse_state.current_doc_id or None

            if self.snippet_trie:
                if scope is None:
                    valid_chars = self.snippet_trie.get_valid_next_chars(
                        parse_state.current_snippet
                    )
                else:
                    valid_chars = self.snippet_trie.get_valid_next_chars_for_document(
                        parse_state.current_snippet,
                        scope,
                    )
            else:
                # Fall back to checking document text
                current = parse_state.current_snippet
                scoped_ids = self._doc_order if scope is None else [scope]
                for doc_id in scoped_ids:
                    doc_text = self.document_index.get_document_text(doc_id)
                    if not doc_text:
                        continue
                    for i in range(len(doc_text) - len(current)):
                        if doc_text[i:i+len(current)] == current:
                            if i + len(current) < len(doc_text):
                                valid_chars.add(doc_text[i + len(current)])

            # Quote to end snippet (if valid)
            if self.is_valid_complete_snippet(
                parse_state.current_snippet,
                scope
            ):
                valid_chars.add('"')

            # Backslash for escaping
            if self.allow_nested_quotes:
                valid_chars.add('\\')

            return valid_chars

        elif state == CitationState.MARKER_INDEX:
            # Digits that keep the index inside 1..N, and ']' once it names a
            # document that actually exists.  An out-of-range marker is therefore
            # unreachable rather than merely penalised.
            current = parse_state.current_doc_id
            valid_chars = {
                digit for digit in "0123456789"
                if (current + digit) in self._marker_prefixes
            }
            if current in self._valid_markers:
                valid_chars.add(']')
            return valid_chars

        elif state == CitationState.SNIPPET_QUOTE_END:
            # After closing quote, must have ']'
            return {']'}

        elif state == CitationState.CITATION_END:
            # Citation complete, go back to free text
            return None

        return None  # Default: all characters valid

    def process_char(
        self,
        char: str,
        parse_state: CitationParseState
    ) -> Tuple[CitationParseState, bool]:
        """
        Process a character and update the parse state.

        Args:
            char: The character to process
            parse_state: Current parsing state

        Returns:
            Tuple of (new_state, is_valid) where is_valid indicates
            if the character was valid in the current context
        """
        new_state = parse_state.copy()
        state = new_state.state

        if state == CitationState.FREE_TEXT:
            if char == '[':
                new_state.state = CitationState.CITATION_START
                new_state.partial_keyword = ""
            # Any other char stays in FREE_TEXT
            return new_state, True

        elif state == CitationState.CITATION_START:
            # `marker` opens on a digit; the keyword representations open on their
            # own literal.  Completing that literal lands in DOCUMENT_ID for the
            # forms that name a document and straight in SNIPPET_CONTENT for the
            # form that only quotes -- which is why snippet_only's literal
            # includes its opening quote.
            if self._opening is None:
                if char.isdigit() and char in self._marker_prefixes:
                    new_state.state = CitationState.MARKER_INDEX
                    new_state.current_doc_id = char
                    new_state.partial_keyword = ""
                    return new_state, True
                new_state.reset()
                return new_state, False

            expected = self._opening
            pos = len(new_state.partial_keyword)

            if pos < len(expected) and char == expected[pos]:
                new_state.partial_keyword += char
                if new_state.partial_keyword == expected:
                    new_state.partial_keyword = ""
                    new_state.state = (
                        CitationState.DOCUMENT_ID
                        if self.format.carries_doc_id
                        else CitationState.SNIPPET_CONTENT
                    )
                return new_state, True
            else:
                # Invalid citation start, go back to free text
                new_state.reset()
                return new_state, False

        elif state == CitationState.DOCUMENT_ID:
            if (
                char == ' '
                and self.format.carries_snippet
                and self.is_valid_doc_id(new_state.current_doc_id)
            ):
                new_state.state = CitationState.SNIPPET_KEYWORD
                new_state.partial_keyword = ""
                return new_state, True
            elif (
                char == ']'
                and not self.format.carries_snippet
                and self.is_valid_doc_id(new_state.current_doc_id)
            ):
                # `doc_only` ends here.  Under `full` this same string is
                # MALFORMED_PATTERNS[0], "Missing snippet"; which of the two it is
                # depends entirely on the representation in force.
                new_state.reset()
                return new_state, True
            elif self.is_valid_doc_id_char(char):
                new_id = new_state.current_doc_id + char
                if self.is_valid_doc_id_prefix(new_id):
                    new_state.current_doc_id = new_id
                    return new_state, True
            # Invalid document ID char
            new_state.reset()
            return new_state, False

        elif state == CitationState.SNIPPET_KEYWORD:
            expected = 'snippet:"'
            pos = len(new_state.partial_keyword)

            if pos < len(expected) and char == expected[pos]:
                new_state.partial_keyword += char
                if new_state.partial_keyword == expected:
                    new_state.state = CitationState.SNIPPET_CONTENT
                    new_state.partial_keyword = ""
                return new_state, True
            # Invalid snippet keyword
            new_state.reset()
            return new_state, False

        elif state == CitationState.MARKER_INDEX:
            if char == ']' and new_state.current_doc_id in self._valid_markers:
                new_state.reset()
                return new_state, True
            if char.isdigit():
                extended = new_state.current_doc_id + char
                if extended in self._marker_prefixes:
                    new_state.current_doc_id = extended
                    return new_state, True
            new_state.reset()
            return new_state, False

        elif state == CitationState.SNIPPET_CONTENT:
            if new_state.escape_next:
                # Accept escaped character
                new_state.current_snippet += char
                new_state.escape_next = False
                return new_state, True

            if char == '\\' and self.allow_nested_quotes:
                new_state.escape_next = True
                return new_state, True

            if char == '"':
                # End of snippet
                if self.is_valid_complete_snippet(
                    new_state.current_snippet,
                    new_state.current_doc_id or None
                ):
                    new_state.state = CitationState.SNIPPET_QUOTE_END
                    return new_state, True
                # Invalid snippet
                new_state.reset()
                return new_state, False

            # Regular character - check if valid continuation
            new_snippet = new_state.current_snippet + char
            if len(new_snippet) > self.max_snippet_length:
                new_state.reset()
                return new_state, False

            if self.is_valid_snippet_prefix(
                new_snippet, new_state.current_doc_id or None
            ):
                new_state.current_snippet = new_snippet
                return new_state, True

            # Invalid snippet continuation
            new_state.reset()
            return new_state, False

        elif state == CitationState.SNIPPET_QUOTE_END:
            if char == ']':
                new_state.state = CitationState.FREE_TEXT
                new_state.reset()
                return new_state, True
            # Missing closing bracket
            new_state.reset()
            return new_state, False

        return new_state, True


class CitationConstraint:
    """
    High-level interface for citation constraints in decoding.

    This class provides methods for integrating citation constraints
    with various decoding strategies (greedy, beam search, sampling).
    """

    def __init__(
        self,
        document_index: DocumentIndex,
        snippet_trie: Optional[SnippetTrie] = None,
        tokenizer: Any = None,
        min_snippet_length: int = 10,
        max_snippet_length: int = 500,
        allow_uncited_text: bool = True,
        fallback_message: str = "Cannot answer using provided documents.",
        token_vocab: Optional[Dict[int, str]] = None,
        citation_format: str = DEFAULT_FORMAT,
    ):
        """
        Initialize citation constraints.

        Args:
            document_index: Index of valid documents
            snippet_trie: Optional trie for snippet validation
            tokenizer: Tokenizer for token-level constraints
            min_snippet_length: Minimum snippet length
            max_snippet_length: Maximum snippet length
            allow_uncited_text: Allow free text without citations
            fallback_message: Message when no valid citation possible
            token_vocab: Pre-built {token_id: text} mapping. If provided,
                         the tokenizer vocab scan is skipped entirely.
            citation_format: Which citation representation to accept
        """
        self.document_index = document_index
        self.snippet_trie = snippet_trie
        self.tokenizer = tokenizer
        self.min_snippet_length = min_snippet_length
        self.max_snippet_length = max_snippet_length
        self.allow_uncited_text = allow_uncited_text
        self.fallback_message = fallback_message

        # Create grammar
        self.grammar = CitationGrammar(
            document_index=document_index,
            snippet_trie=snippet_trie,
            max_snippet_length=max_snippet_length,
            citation_format=citation_format,
        )
        self.format: CitationFormat = self.grammar.format

        # Token mappings — use pre-built vocab if available to avoid redundant scan
        if token_vocab is not None:
            self._token_to_text = token_vocab
            self._char_to_tokens: Dict[str, Set[int]] = {}
            for token_id, text in token_vocab.items():
                if text:
                    first_char = text[0]
                    if first_char not in self._char_to_tokens:
                        self._char_to_tokens[first_char] = set()
                    self._char_to_tokens[first_char].add(token_id)
            logger.info(f"Using pre-built token mapping: {len(token_vocab)} tokens")
        else:
            self._build_token_mappings()

        logger.info("CitationConstraint initialized")

    def _build_token_mappings(self) -> None:
        """Build mappings between tokens and characters."""
        self._token_to_text: Dict[int, str] = {}
        self._char_to_tokens: Dict[str, Set[int]] = {}

        if not self.tokenizer:
            return

        vocab_size = getattr(self.tokenizer, 'vocab_size', 0)
        for token_id in range(vocab_size):
            try:
                text = self.tokenizer.decode([token_id])
                if text:
                    self._token_to_text[token_id] = text
                    first_char = text[0]
                    if first_char not in self._char_to_tokens:
                        self._char_to_tokens[first_char] = set()
                    self._char_to_tokens[first_char].add(token_id)
            except Exception:
                continue

    def get_valid_token_ids(
        self,
        parse_state: CitationParseState,
        generated_text: str = ""
    ) -> Optional[Set[int]]:
        """
        Get valid token IDs given the current state.

        Args:
            parse_state: Current parsing state
            generated_text: Text generated so far

        Returns:
            Set of valid token IDs, or None if all tokens are valid
        """
        if not self.tokenizer:
            return None

        # Get valid next characters from grammar
        valid_chars = self.grammar.get_valid_next_chars(parse_state)

        if valid_chars is None:
            return None  # All tokens valid

        if not valid_chars:
            return set()  # No valid tokens

        # Map characters to tokens
        valid_tokens = set()
        for char in valid_chars:
            tokens = self._char_to_tokens.get(char, set())
            for token_id in tokens:
                token_text = self._token_to_text[token_id]
                # Verify the full token is valid
                test_state = parse_state.copy()
                all_valid = True
                for c in token_text:
                    test_state, is_valid = self.grammar.process_char(c, test_state)
                    if not is_valid:
                        all_valid = False
                        break
                if all_valid:
                    valid_tokens.add(token_id)

        return valid_tokens if valid_tokens else None

    def validate_generated_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate that generated text contains only valid citations.

        Args:
            text: The generated text to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # The pattern comes from the representation in force rather than being
        # re-typed here, so this check cannot drift from the verifier, the reward
        # or the automaton above.
        for ref in self._parse(text):
            doc_ids = ref.doc_ids
            if not doc_ids:
                if self.format.carries_doc_id:
                    errors.append("Citation names no valid document")
                elif ref.snippet is not None:
                    errors.append(
                        f"Snippet is in no provided document: "
                        f"'{ref.snippet[:50]}...'"
                    )
                continue

            for doc_id in doc_ids:
                if not self.document_index.is_valid_document_id(doc_id):
                    errors.append(f"Invalid document ID: {doc_id}")
                    continue
                if ref.snippet is None:
                    continue
                if not self.document_index.is_valid_snippet(doc_id, ref.snippet):
                    errors.append(
                        f"Invalid snippet for document {doc_id}: "
                        f"'{ref.snippet[:50]}...'"
                    )

        return len(errors) == 0, errors

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract all citations from generated text.

        Args:
            text: The generated text

        Returns:
            List of citation dictionaries with 'document_id' and 'snippet'
        """
        citations = []

        for ref in self._parse(text):
            # The dict shape is unchanged for callers (`_fix_citations` in
            # constrained_decoder.py reads these four keys).  A representation
            # that carries no snippet reports an empty one, and one that names no
            # document reports whichever document the quote was found in -- or ""
            # when the quote is ambiguous, which is honest rather than a guess.
            citations.append({
                "document_id": ref.doc_id,
                "snippet": ref.snippet or "",
                "start": ref.span[0],
                "end": ref.span[1],
            })

        return citations

    def _parse(self, text: str):
        """Parse `text` under the representation this constraint enforces."""
        return self.format.parse(
            text,
            doc_order=self.grammar._doc_order,
            resolver=self.document_index.find_documents_containing_snippet,
        )
