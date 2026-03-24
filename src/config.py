"""DynaGuard training pipeline configuration.

All hyperparameters and paths are defined here as a single dataclass.
Change ``BASE_MODEL_ID`` to swap the backbone model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration for the DynaGuard training pipeline.

    Hyperparameter defaults follow Tables 15 & 16 of the DynaGuard paper.
    """

    # ── Base model ────────────────────────────────────────────
    BASE_MODEL_ID: str = "Qwen/Qwen3-1.7B"
    """HuggingFace model ID for the causal LM backbone."""

    # ── SFT hyperparameters (Paper Table 15) ──────────────────
    SFT_LR: float = 3e-5
    """Learning rate for supervised fine-tuning."""
    SFT_BATCH_SIZE: int = 128
    """Global batch size for SFT."""
    SFT_GRAD_ACCUM: int = 8
    """Gradient accumulation steps for SFT."""
    SFT_EPOCHS: int = 1
    """Number of SFT training epochs (paper trains for 1)."""
    SFT_MAX_SEQ_LEN: int = 4096
    """Maximum sequence length for SFT."""

    # ── GRPO hyperparameters (Paper Table 16) ─────────────────
    GRPO_LR: float = 3e-6
    """Learning rate for GRPO."""
    GRPO_BATCH_SIZE: int = 64
    """Global batch size for GRPO."""
    GRPO_GRAD_ACCUM: int = 4
    """Gradient accumulation steps for GRPO."""
    GRPO_EPOCHS: int = 1
    """Number of GRPO training epochs."""
    GRPO_NUM_ROLLOUTS: int = 12
    """Number of rollouts per prompt during GRPO."""
    GRPO_MAX_NEW_TOKENS: int = 1024
    """Max generation length during GRPO rollouts."""
    GRPO_KL_COEFF: float = 1e-3
    """KL penalty coefficient for GRPO."""
    GRPO_NUM_SAMPLES: int = 11_000
    """Number of samples used for GRPO training (paper uses 11k)."""

    # ── LoRA configuration ────────────────────────────────────
    USE_LORA: bool = True
    """Enable LoRA; set False for full fine-tuning."""
    LORA_R: int = 64
    """LoRA rank."""
    LORA_ALPHA: int = 128
    """LoRA alpha scaling factor."""
    LORA_DROPOUT: float = 0.05
    """LoRA dropout rate."""

    # ── Quantization ──────────────────────────────────────────
    USE_4BIT: bool = True
    """Use 4-bit BitsAndBytes quantization to save VRAM."""

    # ── Output paths ──────────────────────────────────────────
    OUTPUT_DIR_SFT: str = "./dynaguard-sft"
    """Directory for SFT checkpoint."""
    OUTPUT_DIR_GRPO: str = "./dynaguard-grpo"
    """Directory for GRPO checkpoint."""
    OUTPUT_DIR_MERGED: str = "./dynaguard-merged"
    """Directory for final merged model."""

    # ── Weights & Biases ──────────────────────────────────────
    WANDB_PROJECT: str = "dynaguard-training"
    """W&B project name."""

    def __post_init__(self) -> None:
        """Set W&B environment variable from config."""
        os.environ["WANDB_PROJECT"] = self.WANDB_PROJECT


default_config = Config()
