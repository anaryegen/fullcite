import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto

from .document_index import DocumentIndex, Document
from .snippet_trie import SnippetTrie, TokenAwareSnippetTrie
from .token_trie import TokenLevelSnippetTrie
from .citation_constraint import (
    CitationConstraint,
    CitationGrammar,
    CitationParseState,
    CitationState
)

from .citation_format import (
    DEFAULT_FORMAT,
    CitationFormat,
    get_format,
)

logger = logging.getLogger(__name__)


class DecodingStrategy(Enum):
    """Available constrained decoding strategies."""
    LOGIT_BIAS = auto()        # Modify logits to constrain generation
    GUIDED_GRAMMAR = auto()     # Use grammar-based guided decoding
    HYBRID = auto()             # Combine free text with constrained citations
    POST_VALIDATION = auto()    # Generate freely, validate after


@dataclass
class ConstrainedDecodingConfig:
    """
    Configuration for constrained decoding.

    Attributes:
        strategy: The decoding strategy to use
        min_snippet_length: Minimum allowed snippet length (characters)
        max_snippet_length: Maximum allowed snippet length (characters)
        build_token_trie: Build a token-level trie for O(1) next-token lookup
                          during snippet generation. Replaces the character trie.
        token_trie_step: Stride (in tokens) for start positions when building
                         the token trie. 1 = full coverage.
        build_trie: Build the legacy character-level trie (kept for non-logit
                    generation modes; not used by the logit processor when
                    build_token_trie=True).
        snippet_step: Step for the character trie (only relevant if build_trie=True).
        allow_uncited_text: Allow text without citations
        enforce_at_least_one_citation: Require at least one citation
        fallback_message: Message when no valid answer possible
        max_retries: Max retries on constraint violation
        temperature_on_retry: Temperature for retry attempts
        citation_format: Name of the citation representation to emit and enforce
                         (see constrained_decoding.citation_format). Defaults to
                         "full", the project's own `[doc:<id> snippet:"<text>"]`.
                         This is the only knob that changes what a citation looks
                         like; it reaches the prompt, the FSA and the verifier.
    """
    strategy: DecodingStrategy = DecodingStrategy.HYBRID
    min_snippet_length: int = 10
    max_snippet_length: int = 500
    build_token_trie: bool = True   # token-level trie (fast, recommended)
    token_trie_step: int = 1        # step in tokens; 1 = full coverage
    build_trie: bool = False        # legacy char-level trie
    snippet_step: int = 5           # step for char trie (only if build_trie=True)
    allow_uncited_text: bool = True
    enforce_at_least_one_citation: bool = True
    fallback_message: str = "Cannot answer using provided documents."
    max_retries: int = 3
    temperature_on_retry: float = 0.7
    logit_bias_strength: float = 100.0
    citation_format: str = DEFAULT_FORMAT

    def __post_init__(self):
        # Fail here rather than three layers down in the FSA, where a typo would
        # surface as a dead generation
        get_format(self.citation_format)

    @property
    def format(self) -> CitationFormat:
        """The citation representation this configuration selects."""
        return get_format(self.citation_format)


class ConstrainedDecoder:
    """
    Main constrained decoder class that integrates with vLLM.

    This decoder enforces that generated text contains only valid citations
    with correct document IDs and verbatim snippets from the source documents.

    The decoder can be used in two modes:
    1. Wrapper mode: Wraps existing generation and validates/retries
    2. Integrated mode: Modifies vLLM generation parameters for constraint enforcement

    Example usage:
        ```python
        documents = [
            Document("doc1", "The patient has diabetes mellitus type 2."),
            Document("doc2", "Hypertension is a common comorbidity.")
        ]

        decoder = ConstrainedDecoder(
            documents=documents,
            tokenizer=tokenizer,
            config=ConstrainedDecodingConfig()
        )

        # Generate with constraints
        output = decoder.generate(
            engine=vllm_engine,
            prompt="What conditions does the patient have?",
            sampling_params=sampling_config
        )
        ```
    """

    def __init__(
        self,
        documents: List[Document],
        tokenizer: Any = None,
        config: Optional[ConstrainedDecodingConfig] = None,
        token_vocab: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the constrained decoder.

        Args:
            documents: List of Document objects to use for constraints
            tokenizer: HuggingFace tokenizer for token-level operations
            config: Configuration for constrained decoding
            token_vocab: Pre-built {token_id: text} mapping shared across
                         questions. Pass this to avoid rebuilding the mapping
                         (vocab_size decode calls) on every question.
        """
        self.config = config or ConstrainedDecodingConfig()
        self.tokenizer = tokenizer

        # Build document index
        logger.info(f"Building document index for {len(documents)} documents...")
        self.document_index = DocumentIndex(
            documents=documents,
            min_snippet_length=self.config.min_snippet_length,
            max_snippet_length=self.config.max_snippet_length
        )

        # Token-level trie (recommended): O(1) next-token lookup, no char→token
        # fan-out at generation time.
        # The snippet tries exist only to constrain quoted spans, so a
        # representation that emits no span needs neither.  Leaving token_trie as
        # None also makes the SNIPPET_CONTENT branch in get_logit_processor
        # unreachable for those arms, which is correct by construction rather
        # than by an extra guard.
        wants_snippet_trie = self.config.format.carries_snippet
        self.token_trie: Optional[TokenLevelSnippetTrie] = None
        if self.config.build_token_trie and tokenizer and wants_snippet_trie:
            logger.info("Building token-level snippet trie...")
            self.token_trie = TokenLevelSnippetTrie.build_from_documents(
                documents=documents,
                tokenizer=tokenizer,
                min_chars=self.config.min_snippet_length,
                max_chars=self.config.max_snippet_length,
                token_step=self.config.token_trie_step,
            )

        # Legacy character-level trie (only built if explicitly requested).
        self.snippet_trie: Optional[SnippetTrie] = None
        if self.config.build_trie and wants_snippet_trie:
            logger.info("Building character-level snippet trie...")
            self.snippet_trie = self._build_snippet_trie(documents, token_vocab)

        # Citation constraint handler — used for format-state validation and
        # the char-level fallback path inside get_logit_processor.
        self.citation_constraint = CitationConstraint(
            document_index=self.document_index,
            snippet_trie=self.snippet_trie,
            tokenizer=tokenizer,
            min_snippet_length=self.config.min_snippet_length,
            max_snippet_length=self.config.max_snippet_length,
            fallback_message=self.config.fallback_message,
            token_vocab=token_vocab,
        )

        # Precompute closing-quote token IDs used by the token-trie path.
        # These are the tokens whose decoded text is exactly `"` — needed to
        # allow ending a snippet when can_end() returns True.
        self._closing_quote_token_ids: Set[int] = set()
        if not wants_snippet_trie:
            pass  # no quoted span in this representation; nothing to close
        elif token_vocab:
            self._closing_quote_token_ids = {
                tid for tid, text in token_vocab.items() if text == '"'
            }
        elif tokenizer:
            # Slow fallback — only runs if token_vocab was not provided
            for tid in range(getattr(tokenizer, 'vocab_size', 0)):
                try:
                    if tokenizer.decode([tid]) == '"':
                        self._closing_quote_token_ids.add(tid)
                except Exception:
                    pass

        # Special tokens needed for logit masking.
        self._special_tokens: Set[int] = set()
        if tokenizer:
            if hasattr(tokenizer, 'all_special_ids'):
                self._special_tokens = set(tokenizer.all_special_ids)
            elif hasattr(tokenizer, 'eos_token_id'):
                self._special_tokens = {tokenizer.eos_token_id}

        logger.info(f"ConstrainedDecoder initialized with strategy: {self.config.strategy}")

    def _build_snippet_trie(
        self,
        documents: List[Document],
        token_vocab: Optional[Dict[int, str]] = None,
    ) -> SnippetTrie:
        """Build a snippet trie from documents."""
        if self.tokenizer:
            trie = TokenAwareSnippetTrie(self.tokenizer, token_vocab=token_vocab)
        else:
            trie = SnippetTrie()

        for doc in documents:
            trie.insert_all_substrings(
                text=doc.text,
                document_id=doc.document_id,
                min_length=self.config.min_snippet_length,
                max_length=self.config.max_snippet_length,
                step=self.config.snippet_step
            )

        stats = trie.get_stats()
        logger.info(f"Snippet trie built: {stats}")
        return trie

    def generate(
        self,
        engine: Any,
        prompt: str,
        sampling_params: Any = None,
        context_prefix: str = "",
        **kwargs
    ) -> str:
        """
        Generate text with citation constraints.

        This is the main entry point for constrained generation. The method
        handles constraint enforcement according to the configured strategy.

        Args:
            engine: vLLM engine or compatible generation interface
            prompt: The input prompt
            sampling_params: Sampling configuration
            context_prefix: Optional prefix for document context
            **kwargs: Additional arguments passed to engine.generate()

        Returns:
            Generated text with valid citations
        """
        strategy = self.config.strategy

        if strategy == DecodingStrategy.POST_VALIDATION:
            return self._generate_with_validation(
                engine, prompt, sampling_params, **kwargs
            )
        elif strategy == DecodingStrategy.HYBRID:
            return self._generate_hybrid(
                engine, prompt, sampling_params, **kwargs
            )
        elif strategy == DecodingStrategy.LOGIT_BIAS:
            return self._generate_with_logit_bias(
                engine, prompt, sampling_params, **kwargs
            )
        else:
            # Default to validation-based approach
            return self._generate_with_validation(
                engine, prompt, sampling_params, **kwargs
            )

    def _generate_with_validation(
        self,
        engine: Any,
        prompt: str,
        sampling_params: Any,
        **kwargs
    ) -> str:
        """
        Generate with post-generation validation and retry.

        This approach:
        1. Generates text normally
        2. Validates citations in output
        3. Retries with higher temperature if invalid
        4. Falls back to error message if all retries fail
        """
        for attempt in range(self.config.max_retries):
            # Adjust temperature on retries
            if attempt > 0 and sampling_params:
                sampling_params = self._adjust_temperature(
                    sampling_params,
                    self.config.temperature_on_retry * (attempt + 1)
                )

            # Generate
            output = engine.generate(prompt, sampling_params, **kwargs)
            if isinstance(output, list):
                output = output[0]

            # Validate
            is_valid, errors = self.citation_constraint.validate_generated_text(output)

            if is_valid:
                # Check for at least one citation if required
                if self.config.enforce_at_least_one_citation:
                    citations = self.citation_constraint.extract_citations(output)
                    if not citations:
                        logger.warning(f"Attempt {attempt + 1}: No citations found")
                        continue

                logger.info(f"Valid output generated on attempt {attempt + 1}")
                return output

            logger.warning(f"Attempt {attempt + 1} validation errors: {errors}")

        # All retries failed
        logger.error("All generation attempts failed validation")
        return self.config.fallback_message

    def _generate_hybrid(
        self,
        engine: Any,
        prompt: str,
        sampling_params: Any,
        **kwargs
    ) -> str:
        """
        Hybrid generation: free text + constrained citations.

        This approach uses a two-phase generation strategy:
        1. Generate text freely using the LLM
        2. Parse the output and identify citation blocks
        3. For any invalid citations, attempt to fix them by:
           a. Finding the closest valid snippet in the target document
           b. Replacing the invalid snippet with a valid one
        4. If fixing fails, retry with adjusted parameters

        This approach maintains natural text flow while ensuring citation validity.
        """
        for attempt in range(self.config.max_retries):
            # Adjust temperature on retries
            current_params = sampling_params
            if attempt > 0 and sampling_params:
                current_params = self._adjust_temperature(
                    sampling_params,
                    self.config.temperature_on_retry * (attempt + 1)
                )

            # Generate freely
            output = engine.generate(prompt, current_params, **kwargs)
            if isinstance(output, list):
                output = output[0]

            # Validate and attempt to fix citations
            fixed_output, is_valid = self._fix_citations(output)

            if is_valid:
                # Check for at least one citation if required
                if self.config.enforce_at_least_one_citation:
                    citations = self.citation_constraint.extract_citations(fixed_output)
                    if not citations:
                        logger.warning(f"Attempt {attempt + 1}: No citations found")
                        continue

                logger.info(f"Valid output generated on attempt {attempt + 1}")
                return fixed_output

            logger.warning(f"Attempt {attempt + 1}: Could not fix all citations")

        # All retries failed
        logger.error("All generation attempts failed validation")
        return self.config.fallback_message

    def _fix_citations(self, text: str) -> Tuple[str, bool]:
        """
        Attempt to fix invalid citations in generated text.

        For each citation:
        1. Check if document ID is valid
        2. Check if snippet exists in document (case-insensitive)
        3. If snippet is valid but case differs, correct it to match document
        4. If snippet is invalid, try to find the closest matching valid snippet
        5. Replace the citation with the fixed version

        Args:
            text: Generated text with potential invalid citations

        Returns:
            Tuple of (fixed_text, all_citations_valid)
        """
        citations = self.citation_constraint.extract_citations(text)

        if not citations:
            return text, True

        all_valid = True
        fixed_text = text

        # Process citations in reverse order to preserve positions
        for citation in reversed(citations):
            doc_id = citation['document_id']
            snippet = citation['snippet']
            start = citation['start']
            end = citation['end']

            # Check document ID
            if not self.document_index.is_valid_document_id(doc_id):
                logger.warning(f"Invalid document ID: {doc_id}")
                all_valid = False
                continue

            # Check snippet (case-insensitive by default)
            if self.document_index.is_valid_snippet(doc_id, snippet):
                # Snippet is valid - but we should still correct the case to match document
                corrected_snippet = self.document_index.get_corrected_snippet(doc_id, snippet)
                if corrected_snippet and corrected_snippet != snippet:
                    # Case differs, replace with corrected version
                    new_citation = self.config.format.render(
                        doc_id=doc_id, snippet=corrected_snippet,
                        doc_index=self._document_index_of(doc_id),
                    )
                    fixed_text = fixed_text[:start] + new_citation + fixed_text[end:]
                    logger.info(f"Corrected citation case for {doc_id}")
                continue  # Citation is valid

            # Try to find a valid snippet
            fixed_snippet = self._find_closest_valid_snippet(doc_id, snippet)

            if fixed_snippet:
                # Build replacement citation
                new_citation = self.config.format.render(
                    doc_id=doc_id, snippet=fixed_snippet,
                    doc_index=self._document_index_of(doc_id),
                )
                fixed_text = fixed_text[:start] + new_citation + fixed_text[end:]
                logger.info(f"Fixed citation for {doc_id}: replaced snippet")
            else:
                logger.warning(f"Could not find valid snippet for {doc_id}")
                all_valid = False

        return fixed_text, all_valid

    def _find_closest_valid_snippet(
        self,
        document_id: str,
        target_snippet: str
    ) -> Optional[str]:
        """
        Find the closest valid snippet in a document to the target.

        Uses multiple strategies:
        1. Check if target is a substring of a valid snippet
        2. Check if a valid snippet is a substring of target
        3. Find snippets with highest character overlap
        4. Use the snippet trie if available

        Args:
            document_id: The document to search in
            target_snippet: The invalid snippet to match

        Returns:
            A valid snippet if found, None otherwise
        """
        doc_text = self.document_index.get_document_text(document_id)
        if not doc_text:
            return None

        target_lower = target_snippet.lower()
        min_len = self.config.min_snippet_length
        max_len = self.config.max_snippet_length

        # Strategy 1: Check if target exists in document (case-insensitive search)
        doc_lower = doc_text.lower()
        target_pos = doc_lower.find(target_lower)
        if target_pos != -1:
            # Found case-insensitive match, get the exact text from document
            actual_snippet = doc_text[target_pos:target_pos + len(target_snippet)]
            if min_len <= len(actual_snippet) <= max_len:
                return actual_snippet

        # Strategy 2: Find overlapping substring in document
        best_snippet = None
        best_score = 0

        # Slide a window over the document text to find best match
        for i in range(len(doc_text) - min_len + 1):
            for length in range(min_len, min(max_len, len(doc_text) - i) + 1):
                candidate = doc_text[i:i + length]
                score = self._calculate_overlap_score(target_snippet, candidate)

                if score > best_score:
                    best_score = score
                    best_snippet = candidate

                # Early exit if we found a very good match
                if score >= 0.7:
                    return best_snippet

        # Only return if we found a reasonably good match
        if best_score >= 0.7:
           return best_snippet

        # Strategy 3: Use trie for prefix matching
        if self.snippet_trie:
            # Try to find completions of target prefix
            for prefix_len in range(min(len(target_snippet), 50), min_len - 1, -1):
                prefix = target_snippet[:prefix_len]
                if self.snippet_trie.has_prefix_for_document(prefix, document_id):
                    completions = self.snippet_trie.get_completions_for_document(
                        prefix, document_id
                    )
                    if completions:
                        # Return first valid completion
                        for completion in completions:
                            if min_len <= len(completion) <= max_len:
                                return completion

        return best_snippet

    def _calculate_overlap_score(self, target: str, candidate: str) -> float:
        """
        Calculate overlap score between target and candidate strings.

        Uses a simple word-level Jaccard similarity.

        Args:
            target: Target snippet
            candidate: Candidate snippet

        Returns:
            Score between 0 and 1
        """
        target_words = set(target.lower().split())
        candidate_words = set(candidate.lower().split())

        if not target_words or not candidate_words:
            return 0.0

        intersection = len(target_words & candidate_words)
        union = len(target_words | candidate_words)

        return intersection / union if union > 0 else 0.0

    def _generate_with_logit_bias(
        self,
        engine: Any,
        prompt: str,
        sampling_params: Any,
        **kwargs
    ) -> str:
        """
        Generate using logit bias to constrain tokens.

        Attaches a stateful logit processor to every generation call so that
        invalid tokens are masked to -inf at each decoding step:
        - FREE_TEXT: all tokens allowed (no overhead)
        - Inside a citation: only grammatically valid next tokens are kept

        Falls back to post-validation if no tokenizer is available.
        """
        processor = self.get_logit_processor()
        if processor is None:
            logger.warning(
                "No tokenizer available for logit bias, falling back to validation"
            )
            return self._generate_with_validation(
                engine, prompt, sampling_params, **kwargs
            )

        for attempt in range(self.config.max_retries):
            current_params = sampling_params
            if attempt > 0 and sampling_params:
                current_params = self._adjust_temperature(
                    sampling_params,
                    self.config.temperature_on_retry * (attempt + 1)
                )

            output = engine.generate(
                prompt, current_params,
                logit_processors=[processor],
                **kwargs
            )
            if isinstance(output, list):
                output = output[0]

            # Logit bias should enforce validity, but validate to be safe
            is_valid, errors = self.citation_constraint.validate_generated_text(output)

            if is_valid:
                if self.config.enforce_at_least_one_citation:
                    citations = self.citation_constraint.extract_citations(output)
                    if not citations:
                        logger.warning(f"Attempt {attempt + 1}: No citations found")
                        continue

                logger.info(f"Valid output generated on attempt {attempt + 1}")
                return output

            logger.warning(f"Attempt {attempt + 1} validation errors: {errors}")

        logger.error("All generation attempts failed validation")
        return self.config.fallback_message

    def _adjust_temperature(self, sampling_params: Any, new_temp: float) -> Any:
        """Adjust temperature in sampling params."""
        if hasattr(sampling_params, 'temperature'):
            # Create a copy with new temperature
            from dataclasses import replace
            try:
                return replace(sampling_params, temperature=min(new_temp, 2.0))
            except Exception:
                sampling_params.temperature = min(new_temp, 2.0)
        return sampling_params

    def build_constrained_prompt(
        self,
        question: str,
        include_format_instructions: bool = True
    ) -> str:
        """
        Build a prompt that includes format instructions for citations.

        Args:
            question: The question to answer
            include_format_instructions: Whether to include format instructions

        Returns:
            Formatted prompt string
        """
        # Build document context
        doc_context = self._format_documents()

        # Format instructions
        format_instructions = ""
        if include_format_instructions:
            format_instructions = self._get_format_instructions()

        prompt = f"""{format_instructions}

Documents:
{doc_context}

Question: {question}

Answer:"""

        return prompt

    def _document_index_of(self, doc_id: str) -> Optional[int]:
        """1-based position of `doc_id` in the prompt's document listing.

        Only the marker representation needs this, but it is computed from the
        same sorted order `_format_documents` emits, so the index a repaired
        citation carries is the index the model was shown.
        """
        if self.config.format.doc_ref_kind != "index":
            return None
        order = sorted(self.document_index.get_all_document_ids())
        try:
            return order.index(str(doc_id)) + 1
        except ValueError:
            return None

    def _format_documents(self) -> str:
        """Format documents for the prompt context.

        The header comes from the citation representation: a form that cites by
        position has to show the position, or the model would have to infer it by
        counting.  Every other form renders exactly `Document <id>:` as before.
        """
        fmt = self.config.format
        parts = []
        for index, doc_id in enumerate(
            sorted(self.document_index.get_all_document_ids()), start=1
        ):
            doc = self.document_index.get_document(doc_id)
            if doc:
                parts.append(f"{fmt.document_header(doc_id, index)}\n{doc.text}")

        return "\n\n".join(parts)

    def _get_format_instructions(self) -> str:
        """Get citation format instructions for the prompt.

        The text is owned by the citation representation so that the instruction
        block, the FSA and the verifier cannot disagree about what a citation is.
        Every representation keeps the same scaffolding and the same six numbered
        rules, so instruction length does not become a second difference between
        the ablation arms.
        """
        doc_ids = list(self.document_index.get_all_document_ids())
        example_ids = ", ".join(doc_ids) if doc_ids else "doc1, doc2"

        return self.config.format.instructions(
            example_ids=example_ids,
            fallback_message=self.config.fallback_message,
            n_docs=len(doc_ids),
        )

    def get_logit_processor(self) -> Optional[Callable]:
        """
        Get a stateful logit processor for constrained decoding.

        Two paths, selected per decoding step:

        SNIPPET_CONTENT (token-trie path — fast)
            Valid next token IDs come directly from self.token_trie, an O(1)
            child-node lookup.  No character-to-token conversion at all.

        All other states (char-FSM path)
            The citation format (`[doc:ID snippet:"…"]`) is short and
            structurally fixed.  The existing char-level FSM handles it.
            `get_valid_token_ids` converts a small set of valid chars to
            token IDs — fast because the format is brief.

        The parse state is maintained incrementally: only the new characters
        added since the previous call are processed, not the full text.
        """
        if not self.tokenizer:
            return None

        # ── Closure state ─────────────────────────────────────────────
        parse_state = CitationParseState()
        prev_token_count = 0
        current_text = ""
        # Index into token_ids where the current snippet started.
        # None if we haven't entered SNIPPET_CONTENT cleanly yet.
        snippet_start_idx: Optional[int] = None

        def process_logits(token_ids: List[int], logits: Any) -> Any:
            nonlocal parse_state, prev_token_count, snippet_start_idx, current_text

            was_in_snippet = (parse_state.state == CitationState.SNIPPET_CONTENT)

            # ── Incremental char-FSM update ────────────────────────────
            # Decode only the *new* tokens added since the last call — O(new_tokens)
            # Single-token decode is compositional
            # for all standard BPE tokenizers (Llama, Qwen, Gemma).
            new_tokens = token_ids[prev_token_count:]
            prev_token_count = len(token_ids)
            new_chars = self.tokenizer.decode(new_tokens) if new_tokens else ""
            current_text += new_chars
            for char in new_chars:
                parse_state, _ = self.citation_constraint.grammar.process_char(
                    char, parse_state
                )

            now_in_snippet = (parse_state.state == CitationState.SNIPPET_CONTENT)

            # ── Track where the snippet starts in the token sequence ───
            if not was_in_snippet and now_in_snippet:
                if not parse_state.current_snippet:
                    # Clean token boundary: the `"` that opened the snippet
                    # was its own token.  Next token is the first snippet token.
                    snippet_start_idx = len(token_ids)
                else:
                    # `"` was bundled with the first snippet char in one token.
                    # Can't cleanly split — fall back to char path for this snippet.
                    snippet_start_idx = None
            elif not now_in_snippet:
                snippet_start_idx = None

            # ── FREE_TEXT: no constraints ─────────────────────────────
            if parse_state.state == CitationState.FREE_TEXT:
                return logits

            # ── SNIPPET_CONTENT: token-trie path ──────────────────────
            if (
                now_in_snippet
                and self.token_trie is not None
                and snippet_start_idx is not None
            ):
                # `or None` is load-bearing.  A representation that carries no
                # document id (snippet_only) leaves current_doc_id as "", and
                # get_valid_next_token_ids filters on `"" in reachable_docs`,
                # which is never true -- so every logit would be masked to -inf
                # and generation would die.  None means "any document", which is
                # exactly the right constraint when no document has been named.
                doc_id = parse_state.current_doc_id or None
                snippet_tokens = list(token_ids[snippet_start_idx:])

                valid_next: Set[int] = self.token_trie.get_valid_next_token_ids(
                    snippet_tokens, doc_id
                )
                # If the current prefix is already a complete snippet, also
                # allow the closing `"` token so the model can end it.
                if self.token_trie.can_end(snippet_tokens, doc_id):
                    valid_next = valid_next | self._closing_quote_token_ids

                return self._apply_mask(logits, valid_next)

            # ── Format states: char-FSM path ──────────────────────────
            valid_tokens = self.citation_constraint.get_valid_token_ids(
                parse_state, current_text
            )
            if valid_tokens is None:
                return logits
            return self._apply_mask(logits, valid_tokens)

        return process_logits

    def _apply_mask(self, logits: Any, valid_token_ids: Set[int]) -> Any:
        """Zero-out all tokens not in valid_token_ids (plus special tokens)."""
        import torch
        if not isinstance(logits, torch.Tensor):
            return logits
        mask = torch.full_like(logits, float('-inf'))
        for tid in valid_token_ids:
            if tid < len(mask):
                mask[tid] = 0.0
        for tid in self._special_tokens:
            if tid < len(mask):
                mask[tid] = 0.0
        return logits + mask

    def _parse_text_state(self, text: str) -> CitationParseState:
        """Parse text to determine current citation state."""
        state = CitationParseState()

        for char in text:
            state, _ = self.citation_constraint.grammar.process_char(char, state)

        return state

    def validate_output(self, text: str) -> Tuple[bool, List[Dict]]:
        """
        Validate generated output for citation correctness.

        Args:
            text: Generated text to validate

        Returns:
            Tuple of (is_valid, list of citation details with validation status)
        """
        citations = self.citation_constraint.extract_citations(text)
        results = []
        all_valid = True

        for citation in citations:
            doc_id = citation['document_id']
            snippet = citation['snippet']

            doc_valid = self.document_index.is_valid_document_id(doc_id)
            snippet_valid = self.document_index.is_valid_snippet(doc_id, snippet) if doc_valid else False

            result = {
                'document_id': doc_id,
                'snippet': snippet,
                'document_id_valid': doc_valid,
                'snippet_valid': snippet_valid,
                'overall_valid': doc_valid and snippet_valid
            }
            results.append(result)

            if not result['overall_valid']:
                all_valid = False

        return all_valid, results

    @classmethod
    def from_dict_documents(
        cls,
        documents: List[Dict[str, str]],
        tokenizer: Any = None,
        config: Optional[ConstrainedDecodingConfig] = None,
        id_key: str = "document_id",
        text_key: str = "text"
    ) -> "ConstrainedDecoder":
        """
        Create a ConstrainedDecoder from a list of document dictionaries.

        Args:
            documents: List of document dictionaries
            tokenizer: HuggingFace tokenizer
            config: Decoder configuration
            id_key: Key for document ID in dictionaries
            text_key: Key for document text in dictionaries

        Returns:
            ConstrainedDecoder instance
        """
        doc_objects = [
            Document(
                document_id=doc[id_key],
                text=doc[text_key]
            )
            for doc in documents
        ]
        return cls(documents=doc_objects, tokenizer=tokenizer, config=config)
