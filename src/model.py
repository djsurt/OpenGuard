"""Model and tokenizer loading, LoRA configuration.

Handles loading the base causal LM with optional 4-bit quantization,
tokenizer setup, and LoRA wrapping via PEFT.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import Config


def load_tokenizer(config: Config) -> AutoTokenizer:
    """Load and configure the tokenizer for the base model.

    Sets the pad token to EOS if not already defined.

    Args:
        config: Pipeline configuration with ``BASE_MODEL_ID``.

    Returns:
        Configured tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_ID, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print(f"Chat template available: {tokenizer.chat_template is not None}")
    return tokenizer


def load_base_model(config: Config) -> AutoModelForCausalLM:
    """Load the base causal LM with optional 4-bit BnB quantization.

    Enables gradient checkpointing to reduce VRAM usage.

    Args:
        config: Pipeline configuration with ``BASE_MODEL_ID`` and
            ``USE_4BIT``.

    Returns:
        The loaded model with gradient checkpointing enabled.
    """
    bnb_config = None
    if config.USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=None,
    )

    model.gradient_checkpointing_enable()

    print(f"Model loaded: {config.BASE_MODEL_ID}")
    print(f"Parameters   : {model.num_parameters() / 1e9:.2f}B")
    print(f"Dtype        : {model.dtype}")
    return model


def apply_lora(
    model: AutoModelForCausalLM, config: Config
) -> tuple[AutoModelForCausalLM, LoraConfig | None]:
    """Wrap the model with LoRA adapters if enabled in config.

    Calls ``prepare_model_for_kbit_training`` when 4-bit quantization is
    active.

    Args:
        model: The base model to wrap.
        config: Pipeline configuration with LoRA and quantization settings.

    Returns:
        Tuple of ``(model, peft_config)`` where ``peft_config`` is ``None``
        if LoRA is disabled.
    """
    peft_config = None
    if config.USE_LORA:
        if config.USE_4BIT:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        print(
            f"LoRA config: r={config.LORA_R}, "
            f"alpha={config.LORA_ALPHA}, "
            f"dropout={config.LORA_DROPOUT}"
        )
    else:
        print("Full fine-tuning mode (no LoRA)")

    return model, peft_config
