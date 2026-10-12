from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch
import os


@dataclass
class VLLMEngineConfig:
    """
    Configuration for vLLM inference engine with Flash Attention 2 support.

    vLLM uses PagedAttention to manage KV cache efficiently, reducing memory waste
    and enabling larger batch sizes compared to traditional LLM serving frameworks.

    Flash Attention 2 provides additional memory optimization through:
    - Tiling: Breaking attention computation into blocks
    - Recomputation: Avoiding storing full attention matrix
    - Fused kernels: Reducing memory I/O operations
    """

    # Model identifier from HuggingFace Hub or local path
    # Supports AWQ-quantized models with "-awq" suffix
    model_name: str = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "HuggingFace model ID or local path to model directory"}
    )

    # Quantization method: None, 'awq', 'gptq', 'squeezellm'
    # Set to None to use regular (non-quantized) models from HuggingFace
    quantization: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization method (set to None for regular models)"}
    )

    # Tensor parallelism: splits model across multiple GPUs along tensor dimensions
    # For 8B models: 1 GPU sufficient; 70B+ models: use 2-8 GPUs
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs for tensor parallelism (model sharding)"}
    )

    # Pipeline parallelism: splits model layers across GPUs sequentially
    # Generally less efficient than tensor parallelism; use for very large models
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs for pipeline parallelism"}
    )

    # Data type for model weights and activations
    # Options: 'float16', 'bfloat16', 'float32'
    # bfloat16 recommended for NVIDIA Ampere+ GPUs (A100, H100)
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype (bfloat16 for Ampere+, float16 for older GPUs)"}
    )

    # GPU memory utilization target (0.0-1.0)
    # Higher values allow larger batches but risk OOM
    # 0.90 recommended for production (leaves 10% buffer)
    gpu_memory_utilization: float = field(
        default=0.90,
        metadata={"help": "Fraction of GPU memory to use (0.0-1.0)"}
    )

    # Maximum number of sequences in a batch
    # vLLM automatically batches requests up to this limit
    # Larger batches improve throughput but increase latency
    max_num_seqs: int = field(
        default=256,
        metadata={"help": "Maximum batch size for continuous batching"}
    )

    # Maximum model context length (input + output tokens)
    # Should match model's training context or be smaller
    # Larger contexts consume more GPU memory for KV cache
    max_model_len: Optional[int] = field(
        default=4096,
        metadata={"help": "Maximum sequence length (input + output tokens)"}
    )

    # Maximum tokens to generate per request
    # Does not include input tokens
    max_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum output tokens per generation"}
    )

    # Enable Flash Attention 2 for memory-efficient attention
    # Requires: flash-attn package installed, CUDA 11.6+, Ampere+ GPU
    # Provides ~2-4x speedup and 10-20x memory reduction vs standard attention
    enable_flash_attention: bool = field(
        default=True,
        metadata={"help": "Enable Flash Attention 2 for optimized attention computation"}
    )

    # Trust remote code for models requiring custom implementations
    # Required for: Qwen, Baichuan, and some other models
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Allow loading models with custom code"}
    )

    # KV cache data type: 'auto', 'fp8', 'fp16'
    # 'fp8' reduces memory by 2x with minimal quality loss (H100 only)
    # 'auto' uses same dtype as model weights
    kv_cache_dtype: str = field(
        default="auto",
        metadata={"help": "Data type for KV cache (fp8 for H100, auto otherwise)"}
    )

    # Enable prefix caching for shared prompts across requests
    # Highly beneficial for RAG systems with common system prompts
    # Reduces redundant computation for repeated prompt prefixes
    enable_prefix_caching: bool = field(
        default=True,
        metadata={"help": "Cache common prompt prefixes to reduce recomputation"}
    )

    # Enforce eager mode (disable CUDA graphs)
    # CUDA graphs reduce kernel launch overhead but increase memory
    # Set to True if experiencing OOM with small batches
    enforce_eager: bool = field(
        default=True,
        metadata={"help": "Disable CUDA graphs (reduces memory, increases latency)"}
    )

    # Seed for reproducible sampling
    # None = random seed each run
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for reproducible generation"}
    )

    # Override HuggingFace model config values at runtime.
    # Useful for patching model configs that are incompatible with the current
    # vLLM version, e.g. adding a missing 'rope_type' key to rope_scaling.
    hf_overrides: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Dict of HuggingFace config overrides passed to vLLM"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for vLLM engine initialization."""
        d = {
            "model": self.model_name,
            "quantization": self.quantization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
            "kv_cache_dtype": self.kv_cache_dtype,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enforce_eager": self.enforce_eager,
            "seed": self.seed,
        }
        if self.hf_overrides:
            d["hf_overrides"] = self.hf_overrides
        return d


@dataclass
class SamplingConfig:
    """
    Sampling parameters for text generation.

    These parameters control the randomness and diversity of generated text.
    Different strategies suitable for different tasks:
    - Greedy (temperature=0): Deterministic, best for factual QA
    - Low temperature (0.1-0.3): Focused, good for medical tasks
    - Medium temperature (0.7-0.9): Balanced creativity/coherence
    - High temperature (1.0+): Very creative, may be incoherent
    """

    # Temperature controls randomness (0.0 = greedy, higher = more random)
    # Medical tasks: 0.1-0.3 recommended for factual accuracy
    temperature: float = field(
        default=0.1,
        metadata={"help": "Sampling temperature (0.0=greedy, higher=more random)"}
    )

    # Top-p (nucleus) sampling: sample from smallest set with cumulative prob >= p
    # 0.95 means consider tokens covering 95% probability mass
    # Provides better quality than pure temperature sampling
    top_p: float = field(
        default=0.95,
        metadata={"help": "Nucleus sampling threshold (0.0-1.0)"}
    )

    # Top-k sampling: sample from top-k most likely tokens
    # Limits generation to k most probable tokens at each step
    # 50 is reasonable default; -1 disables top-k
    top_k: int = field(
        default=50,
        metadata={"help": "Top-k sampling (-1 to disable)"}
    )

    # Repetition penalty: penalize repeated tokens
    # 1.0 = no penalty, >1.0 = discourage repetition
    # 1.1-1.2 recommended for reducing redundancy
    repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Repetition penalty (1.0=none, >1.0=penalize repeats)"}
    )

    # Length penalty for beam search
    # >1.0 encourages longer sequences
    # <1.0 encourages shorter sequences
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for beam search"}
    )

    # Maximum tokens to generate (per request)
    max_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum output tokens"}
    )

    # Stop sequences: generation stops when any sequence is encountered
    # Empty list = only stop at max_tokens or EOS token
    stop: List[str] = field(
        default_factory=list,
        metadata={"help": "Stop sequences for generation termination"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert sampling configuration to dictionary for vLLM."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "stop": self.stop if self.stop else None,
        }
