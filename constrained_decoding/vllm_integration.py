import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Callable, Union

from .document_index import DocumentIndex, Document
from .constrained_decoder import ConstrainedDecoder, ConstrainedDecodingConfig, DecodingStrategy
from .citation_verifier import CitationVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class ConstrainedGenerationConfig:
    """
    Configuration for constrained generation with vLLM.

    Combines sampling configuration with constraint settings.
    """
    # Sampling parameters
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 2048
    repetition_penalty: float = 1.1

    # Constraint parameters
    min_snippet_length: int = 10
    max_snippet_length: int = 500
    enforce_citations: bool = True
    require_at_least_one_citation: bool = True
    max_retries: int = 3
    fallback_message: str = "Cannot answer using provided documents."


class ConstrainedVLLMEngine:
    """
    Wrapper around VLLMEngine that enforces citation constraints.
    """

    def __init__(
        self,
        vllm_engine: Any,
        documents: List[Document],
        decoder_config: Optional[ConstrainedDecodingConfig] = None
    ):
        """
        Initialize the constrained vLLM engine.

        Args:
            vllm_engine: Base VLLMEngine instance
            documents: List of documents for constraint validation
            decoder_config: Configuration for the constrained decoder
        """
        self.engine = vllm_engine
        self.documents = documents

        # Get tokenizer from engine
        tokenizer = getattr(vllm_engine, 'tokenizer', None)

        # Initialize constrained decoder
        self.decoder = ConstrainedDecoder(
            documents=documents,
            tokenizer=tokenizer,
            config=decoder_config
        )

        # Initialize verifier
        self.verifier = CitationVerifier(
            document_index=self.decoder.document_index,
            min_snippet_length=decoder_config.min_snippet_length if decoder_config else 10,
            max_snippet_length=decoder_config.max_snippet_length if decoder_config else 500
        )

        logger.info(f"ConstrainedVLLMEngine initialized with {len(documents)} documents")

    def generate_with_constraints(
        self,
        prompt: str,
        config: Optional[ConstrainedGenerationConfig] = None,
        include_verification: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with citation constraints.

        Args:
            prompt: The input prompt
            config: Generation configuration
            include_verification: Include verification results in output

        Returns:
            Dictionary containing:
                - text: Generated text
                - is_valid: Whether all citations are valid
                - verification: Full verification result (if requested)
                - retries: Number of retry attempts
        """
        config = config or ConstrainedGenerationConfig()

        # Build sampling params
        from config import SamplingConfig
        sampling_params = SamplingConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            repetition_penalty=config.repetition_penalty
        )

        # Add format instructions to prompt
        full_prompt = self.decoder.build_constrained_prompt(
            question=prompt,
            include_format_instructions=True
        )

        # For LOGIT_BIAS, attach the logit processor to constrain tokens at decode time
        strategy = self.decoder.config.strategy
        logit_processors = None
        if strategy == DecodingStrategy.LOGIT_BIAS:
            processor = self.decoder.get_logit_processor()
            if processor is not None:
                logit_processors = [processor]
            else:
                logger.warning(
                    "LOGIT_BIAS strategy requested but no tokenizer available; "
                    "falling back to post-validation"
                )

        # Generate with retries
        best_output = None
        best_valid_count = -1
        retries = 0

        for attempt in range(config.max_retries):
            # Adjust temperature on retries
            if attempt > 0:
                sampling_params.temperature = min(
                    sampling_params.temperature * 1.5,
                    2.0
                )

            # Generate (with logit processor if LOGIT_BIAS strategy)
            output = self.engine.generate(
                full_prompt, sampling_params,
                logit_processors=logit_processors
            )
            if isinstance(output, list):
                output = output[0]

            # HYBRID: attempt to fix invalid citations before verifying
            if strategy == DecodingStrategy.HYBRID:
                output, _ = self.decoder._fix_citations(output)

            # Verify
            verification = self.verifier.verify(output)

            # Track best output
            valid_count = verification.stats.get('valid_citations', 0)
            if valid_count > best_valid_count:
                best_valid_count = valid_count
                best_output = output
                best_verification = verification

            if verification.is_valid:
                # Check citation requirement
                if config.require_at_least_one_citation:
                    if verification.stats.get('total_citations', 0) > 0:
                        break
                else:
                    break

            retries += 1
            logger.warning(f"Attempt {attempt + 1} failed: {verification.errors}")

        # Use best output or fallback
        # Store the last generated text for debugging even if falling back
        last_generated_text = best_output

        if best_output is None or (
            config.require_at_least_one_citation and best_valid_count == 0
        ):
            # Log the actual generated text before falling back
            if best_output is not None:
                logger.warning(f"Falling back to default message. Last generated text was:\n{best_output}")
            best_output = config.fallback_message
            best_verification = self.verifier.verify(best_output)

        result = {
            "text": best_output,
            "is_valid": best_verification.is_valid,
            "retries": retries,
            "last_generated_text": last_generated_text  # Include actual generated text for debugging
        }

        if include_verification:
            result["verification"] = best_verification.to_dict()

        return result

    def generate_batch_with_constraints(
        self,
        prompts: List[str],
        config: Optional[ConstrainedGenerationConfig] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate text for multiple prompts with constraints.

        Args:
            prompts: List of input prompts
            config: Generation configuration
            show_progress: Show progress bar

        Returns:
            List of result dictionaries
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(prompts, desc="Generating") if show_progress else prompts

        for prompt in iterator:
            result = self.generate_with_constraints(
                prompt=prompt,
                config=config,
                include_verification=True
            )
            results.append(result)

        return results

    def verify_text(self, text: str) -> VerificationResult:
        """
        Verify citations in text without generating.

        Args:
            text: Text to verify

        Returns:
            VerificationResult with validation details
        """
        return self.verifier.verify(text)

    def get_document_index(self) -> DocumentIndex:
        """Get the document index."""
        return self.decoder.document_index


def create_constrained_engine(
    model_name: str,
    documents: List[Dict[str, str]],
    num_gpus: int = 1,
    quantization: Optional[str] = None,
    **kwargs
) -> ConstrainedVLLMEngine:
    """
    Factory function to create a constrained vLLM engine.

    Args:
        model_name: HuggingFace model name or path
        documents: List of document dictionaries with 'document_id' and 'text'
        num_gpus: Number of GPUs for tensor parallelism
        quantization: Quantization method (None, 'awq', etc.)
        **kwargs: Additional arguments for VLLMEngineConfig

    Returns:
        ConstrainedVLLMEngine instance
    """
    from config import VLLMEngineConfig, SamplingConfig
    from models import VLLMEngine

    # Create vLLM config
    vllm_config = VLLMEngineConfig(
        model_name=model_name,
        tensor_parallel_size=num_gpus,
        quantization=quantization,
        **kwargs
    )

    # Create base engine
    base_engine = VLLMEngine(
        vllm_config=vllm_config,
        sampling_config=SamplingConfig()
    )

    # Convert documents
    doc_objects = [
        Document(
            document_id=doc.get('document_id', doc.get('id')),
            text=doc.get('text', doc.get('content'))
        )
        for doc in documents
    ]

    # Create decoder config
    decoder_config = ConstrainedDecodingConfig(
        min_snippet_length=kwargs.get('min_snippet_length', 10),
        max_snippet_length=kwargs.get('max_snippet_length', 500),
        build_trie=kwargs.get('build_trie', True)
    )

    # Create constrained engine
    return ConstrainedVLLMEngine(
        vllm_engine=base_engine,
        documents=doc_objects,
        decoder_config=decoder_config
    )


class ConstrainedBatchProcessor:
    """
    Batch processor for constrained generation.

    Handles large-scale constrained generation with:
    - Automatic batching
    - Progress tracking
    - Result aggregation
    - Error handling
    """

    def __init__(
        self,
        constrained_engine: ConstrainedVLLMEngine,
        batch_size: int = 16
    ):
        """
        Initialize the batch processor.

        Args:
            constrained_engine: The constrained vLLM engine
            batch_size: Number of prompts per batch
        """
        self.engine = constrained_engine
        self.batch_size = batch_size

    def process_dataset(
        self,
        data: List[Dict[str, Any]],
        question_key: str = "question",
        config: Optional[ConstrainedGenerationConfig] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a dataset with constrained generation.

        Args:
            data: List of data dictionaries containing questions
            question_key: Key for the question in each dictionary
            config: Generation configuration
            show_progress: Show progress bar

        Returns:
            List of result dictionaries with generated answers
        """
        from tqdm import tqdm

        results = []
        total = len(data)
        pbar = tqdm(total=total, desc="Processing") if show_progress else None

        for i in range(0, total, self.batch_size):
            batch = data[i:i + self.batch_size]

            for item in batch:
                question = item.get(question_key, "")
                result = self.engine.generate_with_constraints(
                    prompt=question,
                    config=config
                )

                # Combine with original data
                output = {**item, **result}
                results.append(output)

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return results

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from processed results.

        Args:
            results: List of processed results

        Returns:
            Dictionary of statistics
        """
        total = len(results)
        valid = sum(1 for r in results if r.get('is_valid', False))
        retries = sum(r.get('retries', 0) for r in results)

        citation_counts = []
        for r in results:
            verification = r.get('verification', {})
            stats = verification.get('stats', {})
            citation_counts.append(stats.get('total_citations', 0))

        return {
            "total_processed": total,
            "valid_outputs": valid,
            "invalid_outputs": total - valid,
            "validity_rate": valid / total if total > 0 else 0,
            "total_retries": retries,
            "avg_retries": retries / total if total > 0 else 0,
            "total_citations": sum(citation_counts),
            "avg_citations_per_output": sum(citation_counts) / total if total > 0 else 0,
            "min_citations": min(citation_counts) if citation_counts else 0,
            "max_citations": max(citation_counts) if citation_counts else 0
        }
