"""Stage 2: GRPO (Group Relative Policy Optimization) trainer and merge.

Paper Section 3.5, Eq. 3: Group Relative Policy Optimization.
"""

from __future__ import annotations

import shutil

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from src.config import Config
from src.reward import dynaguard_reward_fn


def load_sft_checkpoint(
    config: Config,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the SFT checkpoint model and tokenizer for GRPO.

    Handles both LoRA (via ``AutoPeftModelForCausalLM``) and full fine-tune
    checkpoints.

    Args:
        config: Pipeline configuration with ``OUTPUT_DIR_SFT`` and
            ``USE_LORA``.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    if config.USE_LORA:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            config.OUTPUT_DIR_SFT,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.OUTPUT_DIR_SFT,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR_SFT)
    return model, tokenizer


def build_grpo_trainer(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    config: Config,
) -> GRPOTrainer:
    """Build and return a GRPO trainer.

    Note: ``GRPO_BATCH_SIZE`` is overridden to 60 for divisibility by
    ``GRPO_NUM_ROLLOUTS`` (12).

    Args:
        model: The SFT-trained model.
        tokenizer: Tokenizer for the model.
        dataset: Chat-formatted GRPO dataset with ``"prompt"`` and
            ``"ground_truth_label"`` columns.
        config: Pipeline configuration.

    Returns:
        A configured :class:`GRPOTrainer` ready to call ``.train()``.
    """
    # Adjusted for divisibility by GRPO_NUM_ROLLOUTS (12)
    grpo_batch_size = 60

    n_gpus = max(torch.cuda.device_count(), 1)
    grpo_per_device_bs = max(1, grpo_batch_size // (config.GRPO_GRAD_ACCUM * n_gpus))

    grpo_config = GRPOConfig(
        output_dir=config.OUTPUT_DIR_GRPO,
        num_train_epochs=config.GRPO_EPOCHS,
        per_device_train_batch_size=grpo_per_device_bs,
        gradient_accumulation_steps=config.GRPO_GRAD_ACCUM,
        learning_rate=config.GRPO_LR,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        weight_decay=0.01,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="wandb",
        run_name="dynaguard-grpo",
        # GRPO-specific
        num_generations=config.GRPO_NUM_ROLLOUTS,
        max_completion_length=config.GRPO_MAX_NEW_TOKENS,
        # Generation params (paper: temp=1.0, top_p=1.0)
        temperature=1.0,
        seed=42,
        generation_batch_size=grpo_batch_size,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=dynaguard_reward_fn,
        processing_class=tokenizer,
    )

    print("GRPO Trainer ready.")
    print(f"  Prompts        : {len(dataset):,}")
    print(f"  Rollouts/prompt: {config.GRPO_NUM_ROLLOUTS}")

    return trainer


def run_grpo(
    trainer: GRPOTrainer,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> object:
    """Run GRPO training and save the checkpoint.

    Args:
        trainer: A configured GRPO trainer.
        tokenizer: Tokenizer to save alongside the model.
        config: Pipeline configuration with ``OUTPUT_DIR_GRPO``.

    Returns:
        The training result object.
    """
    grpo_result = trainer.train()

    trainer.save_model(config.OUTPUT_DIR_GRPO)
    tokenizer.save_pretrained(config.OUTPUT_DIR_GRPO)

    print(f"\nGRPO complete. Saved to {config.OUTPUT_DIR_GRPO}")
    return grpo_result


def merge_lora(config: Config) -> None:
    """Merge LoRA adapter weights into the base model.

    If LoRA is not enabled, copies the GRPO checkpoint directory to the
    merged output directory instead.

    Args:
        config: Pipeline configuration with output directories and
            ``USE_LORA``.
    """
    if config.USE_LORA:
        from peft import AutoPeftModelForCausalLM

        print("Merging LoRA weights into base model...")
        merged_model = AutoPeftModelForCausalLM.from_pretrained(
            config.OUTPUT_DIR_GRPO,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(config.OUTPUT_DIR_MERGED)

        # Save tokenizer alongside merged model
        tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR_GRPO)
        tokenizer.save_pretrained(config.OUTPUT_DIR_MERGED)
        print(f"Merged model saved to {config.OUTPUT_DIR_MERGED}")
    else:
        shutil.copytree(config.OUTPUT_DIR_GRPO, config.OUTPUT_DIR_MERGED, dirs_exist_ok=True)
        print(f"No LoRA to merge. Final model copied to {config.OUTPUT_DIR_MERGED}")

    print("\nReady for evaluation!")
