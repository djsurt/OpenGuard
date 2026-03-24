"""Stage 1: Supervised Fine-Tuning (SFT) trainer setup and execution.

The paper trains for 1 epoch over 80k samples (50/50 DynaBench + Safety).
See Paper Section 3.4.
"""

from __future__ import annotations

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from src.config import Config


def build_sft_trainer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    peft_config: LoraConfig | None,
    config: Config,
) -> SFTTrainer:
    """Build and return an SFT trainer with the given configuration.

    Args:
        model: The base (or LoRA-wrapped) model to train.
        tokenizer: Tokenizer for the model.
        dataset: Chat-formatted SFT training dataset with a ``"text"`` column.
        peft_config: PEFT/LoRA configuration, or ``None`` for full fine-tune.
        config: Pipeline configuration.

    Returns:
        A configured :class:`SFTTrainer` ready to call ``.train()``.
    """
    n_gpus = max(torch.cuda.device_count(), 1)
    per_device_bs = max(1, config.SFT_BATCH_SIZE // (config.SFT_GRAD_ACCUM * n_gpus))

    sft_config = SFTConfig(
        output_dir=config.OUTPUT_DIR_SFT,
        num_train_epochs=config.SFT_EPOCHS,
        per_device_train_batch_size=1,  # Force 1
        gradient_accumulation_steps=config.SFT_GRAD_ACCUM * per_device_bs,  # Compensate
        gradient_checkpointing=True,
        learning_rate=config.SFT_LR,
        lr_scheduler_type="cosine",
        warmup_steps=int(
            0.03
            * (
                len(dataset)
                / (per_device_bs * config.SFT_GRAD_ACCUM * n_gpus)
                * config.SFT_EPOCHS
            )
        ),
        max_grad_norm=1.0,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=(not torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to="wandb",
        run_name="dynaguard-sft",
        dataset_text_field="text",
        packing=True,
        seed=42,
        max_length=min(config.SFT_MAX_SEQ_LEN, 1024),  # Cap at 1024
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("SFT Trainer ready.")
    print(f"  Effective batch size : {per_device_bs * config.SFT_GRAD_ACCUM * n_gpus}")
    print(f"  Num training samples : {len(dataset):,}")
    print(f"  Epochs               : {config.SFT_EPOCHS}")

    return trainer


def run_sft(
    trainer: SFTTrainer,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> object:
    """Run SFT training, save the model and tokenizer.

    Args:
        trainer: A configured SFT trainer.
        tokenizer: Tokenizer to save alongside the model.
        config: Pipeline configuration with ``OUTPUT_DIR_SFT``.

    Returns:
        The training result object.
    """
    sft_result = trainer.train()

    trainer.save_model(config.OUTPUT_DIR_SFT)
    tokenizer.save_pretrained(config.OUTPUT_DIR_SFT)

    print(f"\nSFT complete. Saved to {config.OUTPUT_DIR_SFT}")
    print(f"Training loss: {sft_result.training_loss:.4f}")

    return sft_result
