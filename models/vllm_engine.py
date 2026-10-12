import logging
from typing import List, Dict, Any, Optional, Union
import torch
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from transformers import AutoTokenizer

from config import VLLMEngineConfig, SamplingConfig

# Configure module logger
logger = logging.getLogger(__name__)


class VLLMEngine:
    def __init__(
        self,
        vllm_config: VLLMEngineConfig,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Initialize vLLM engine with specified configurations.

        Args:
            vllm_config: Configuration for vLLM engine (model, quantization, etc.)
            sampling_config: Default sampling parameters (temperature, top_p, etc.)

        Raises:
            RuntimeError: If CUDA is not available or model loading fails
            ValueError: If configuration parameters are invalid
        """
        # Store configurations
        self.config = vllm_config
        self.sampling_config = sampling_config or SamplingConfig()

        # Validate CUDA availability for GPU inference
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. vLLM requires NVIDIA GPU for inference."
            )

        # Log GPU information for debugging
        logger.info(f"Detected {torch.cuda.device_count()} CUDA device(s)")
        for i in range(torch.cuda.device_count()):
            # Get GPU properties: name, compute capability, memory
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name}, "
                f"Compute Capability: {props.major}.{props.minor}, "
                f"Memory: {props.total_memory / 1024**3:.2f} GB"
                "\n"
            )

        # Initialize vLLM engine with optimized settings
        logger.info("Initializing vLLM engine...")
        self._initialize_engine()

        # Load tokenizer for prompt processing
        logger.info("Loading tokenizer...")
        self._initialize_tokenizer()

        logger.info("vLLM engine initialized successfully")

    def _initialize_engine(self) -> None:
        """
        Initialize vLLM LLM engine with configuration.
        """
        try:
            # Convert configuration to vLLM engine arguments
            engine_args = self.config.to_dict()

            # Log configuration for debugging
            logger.info(f"vLLM engine configuration: {engine_args}")

            # Initialize vLLM LLM engine
            # This loads the model, sets up quantization, and allocates GPU memory
            self.engine = LLM(**engine_args)

            # Log successful initialization with memory stats
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Get current GPU memory usage
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(
                        f"GPU {i} memory: {allocated:.2f} GB allocated, "
                        f"{reserved:.2f} GB reserved"
                    )

        except Exception as e:
            # Log detailed error information for debugging
            logger.error(f"Failed to initialize vLLM engine: {str(e)}")
            raise RuntimeError(f"vLLM engine initialization failed: {str(e)}")

    def _initialize_tokenizer(self) -> None:
        """
        Initialize HuggingFace tokenizer for text processing.
        """
        try:
            # Load tokenizer from same model path
            # trust_remote_code allows loading custom tokenizers (e.g., Qwen)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
            )

            # Set padding token if not defined
            # Required for batch processing with variable-length sequences
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Set padding side to right for decoder-only models
            # This ensures padding doesn't interfere with generation
            self.tokenizer.padding_side = "right"

            logger.info(f"Tokenizer loaded: {type(self.tokenizer).__name__}")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise RuntimeError(f"Tokenizer loading failed: {str(e)}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingConfig] = None,
        use_tqdm: bool = False,
        logit_processors: Optional[List] = None,
    ) -> Union[str, List[str]]:
        """
        Generate text completions for given prompt(s).
        Args:
            prompts: Single prompt string or list of prompts
            sampling_params: Sampling configuration (uses default if None)
            use_tqdm: Show progress bar for batch generation

        Returns:
            Generated text(s) - single string if input is string, list if input is list

        """
        # Use default sampling config if not provided
        if sampling_params is None:
            sampling_params = self.sampling_config

        # Convert SamplingConfig to vLLM SamplingParams
        # This maps our config format to vLLM's internal format
        sampling_kwargs = dict(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            repetition_penalty=sampling_params.repetition_penalty,
            max_tokens=sampling_params.max_tokens,
            stop=sampling_params.stop if sampling_params.stop else None,
        )
        if logit_processors:
            # vllm 0.19.1's SamplingParams has no `logits_processors` field
            # (it moved to engine-level registration in this version), so
            # only pass it when there is something to pass and the
            # installed vLLM actually accepts it.
            sampling_kwargs["logits_processors"] = logit_processors
        vllm_sampling_params = SamplingParams(**sampling_kwargs)

        # Handle single prompt vs batch of prompts
        is_single = isinstance(prompts, str)
        if is_single:
            # Convert single prompt to list for uniform processing
            prompts = [prompts]

        try:
        
            logger.debug(f"Generating for {len(prompts)} prompt(s)")
            outputs: List[RequestOutput] = self.engine.generate(
                prompts,
                sampling_params=vllm_sampling_params,
                use_tqdm=use_tqdm,
            )

            generated_texts = []
            for output in outputs:
                
                generated_text = output.outputs[0].text
                generated_texts.append(generated_text)

            # Return single string if input was single string
            return generated_texts[0] if is_single else generated_texts

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"vLLM generation error: {str(e)}")

    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingConfig] = None,
        use_tqdm: bool = True,
    ) -> List[str]:
        """
        Generate text completions for batch of prompts.

        Args:
            prompts: List of prompt strings
            sampling_params: Sampling configuration (uses default if None)
            use_tqdm: Show progress bar (default: True)

        Returns:
            List of generated texts corresponding to input prompts

        """
        # Validate input is a list
        if not isinstance(prompts, list):
            raise ValueError("generate_batch requires list of prompts")

        # Use generate method with progress bar enabled
        return self.generate(prompts, sampling_params, use_tqdm)

    def get_token_length(self, text: str) -> int:
        """
        Calculate token length of text using tokenizer.

        Args:
            text: Input text string

        Returns:
            Number of tokens after tokenization
        """
        # Encode text to token IDs 
        # This matches how vLLM will tokenize the prompt
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids)

    def check_prompt_length(self, text: str) -> bool:
        """
        Check if prompt length is within model's context limit.

        Args:
            text: Input prompt text

        Returns:
            True if prompt fits within max_model_len, False otherwise
        """
        # Get token length of prompt
        token_length = self.get_token_length(text)

        # Compare with configured maximum length
        max_length = self.config.max_model_len or 4096

        # Log warning if prompt is too long
        if token_length > max_length:
            logger.warning(
                f"Prompt length ({token_length}) exceeds max_model_len ({max_length})"
            )
            return False

        return True

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage statistics.

        Returns:
            Dictionary with memory stats in GB:
                - allocated: Memory allocated by tensors
                - reserved: Memory reserved by PyTorch
                - free: Available GPU memory
        """
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}

        # Get memory stats from primary GPU (device 0)
        device = 0
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - allocated

        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total,
        }

    def __del__(self):
        """
        Cleanup when engine is destroyed.
        """
        # Free GPU memory by deleting engine
        if hasattr(self, "engine"):
            del self.engine
            # Force garbage collection to free memory immediately
            import gc
            gc.collect()
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("vLLM engine cleaned up")


class BatchInferenceEngine:
    """
    Optimized batch inference engine for processing large datasets.
    """

    def __init__(
        self,
        vllm_engine: VLLMEngine,
        batch_size: int = 32,
        enable_checkpointing: bool = False,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize batch inference engine.

        Args:
            vllm_engine: Underlying vLLM engine for generation
            batch_size: Number of prompts to process per batch
            enable_checkpointing: Save intermediate results for resume capability
            checkpoint_dir: Directory for saving checkpoints
        """
        self.engine = vllm_engine
        self.batch_size = batch_size
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if needed
        if enable_checkpointing and checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Checkpointing enabled: {checkpoint_dir}")

    def process_dataset(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingConfig] = None,
        show_progress: bool = True,
    ) -> List[str]:

        from tqdm import tqdm

        results = []
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        # Create progress bar if requested
        pbar = tqdm(total=len(prompts), desc="Processing") if show_progress else None

        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            # Get current batch
            batch_prompts = prompts[i : i + self.batch_size]

            # Generate for batch
            batch_outputs = self.engine.generate_batch(
                batch_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,  # Disable inner progress bar
            )

            # Append results
            results.extend(batch_outputs)

            # Update progress bar
            if pbar:
                pbar.update(len(batch_prompts))

            # Save checkpoint if enabled
            if self.enable_checkpointing:
                self._save_checkpoint(results, i + len(batch_prompts))

        if pbar:
            pbar.close()

        return results

    def _save_checkpoint(self, results: List[str], num_processed: int) -> None:
        """
        Save intermediate results to checkpoint file.

        Args:
            results: Current list of results
            num_processed: Number of prompts processed so far
        """
        import json
        import os

        if not self.checkpoint_dir:
            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{num_processed}.json"
        )

        with open(checkpoint_path, 'w') as f:
            json.dump({"num_processed": num_processed, "results": results}, f)

        logger.debug(f"Saved checkpoint: {checkpoint_path}")
