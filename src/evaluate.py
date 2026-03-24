"""Evaluation loop and metrics for DynaGuard models.

Evaluates a trained model on the DynaBench test set in both CoT (reasoning)
and Non-CoT (fast inference) modes.
"""

from __future__ import annotations

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.data import SYSTEM_PROMPT
from src.reward import extract_answer


def evaluate_dynaguard(
    model_path: str,
    test_dataset: Dataset,
    config: Config,
    use_cot: bool = True,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
) -> dict[str, float]:
    """Evaluate a DynaGuard model on the DynaBench test set.

    Args:
        model_path: Path to the model checkpoint.
        test_dataset: HF dataset with ``formatted_input`` and ``label`` columns.
        config: Pipeline configuration with ``SFT_MAX_SEQ_LEN``.
        use_cot: If ``True``, prepend ``<think>`` for reasoning mode;
            if ``False``, prepend ``<answer>`` for fast-inference mode.
        max_new_tokens: Maximum tokens to generate per sample.
        batch_size: Evaluation batch size.

    Returns:
        Dict with F1, precision, recall, accuracy, and sample counts.
    """
    eval_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    eval_model.eval()

    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    predictions: list[str] = []
    labels: list[str] = []
    mode_prefix = "<think>\n" if use_cot else "<answer>\n"

    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))
        prompts = []
        for ex in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["formatted_input"]},
            ]
            prompt = eval_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt += mode_prefix
            prompts.append(prompt)

        inputs = eval_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.SFT_MAX_SEQ_LEN,
        ).to(eval_model.device)

        with torch.no_grad():
            outputs = eval_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_k=300,
                do_sample=True,
            )

        for j, (output, ex) in enumerate(zip(outputs, batch)):
            gen_text = eval_tokenizer.decode(
                output[inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            pred = extract_answer(mode_prefix + gen_text)
            labels.append(ex["label"].strip().upper())
            predictions.append(pred)

        if (i // batch_size) % 20 == 0:
            print(f"  Evaluated {min(i + batch_size, len(test_dataset))}/{len(test_dataset)}...")

    # Filter out UNKNOWN predictions for metrics
    valid = [(p, l) for p, l in zip(predictions, labels) if p != "UNKNOWN"]
    if not valid:
        print("ERROR: No valid predictions found!")
        return {}

    v_preds, v_labels = zip(*valid)
    pos_label = "FAIL"  # FAIL = positive class (violation detected)

    metrics = {
        "f1": f1_score(v_labels, v_preds, pos_label=pos_label) * 100,
        "precision": precision_score(v_labels, v_preds, pos_label=pos_label) * 100,
        "recall": recall_score(v_labels, v_preds, pos_label=pos_label) * 100,
        "accuracy": accuracy_score(v_labels, v_preds) * 100,
        "n_valid": len(valid),
        "n_unknown": len(predictions) - len(valid),
    }

    mode_name = "CoT" if use_cot else "Non-CoT"
    print(f"\n{'=' * 50}")
    print(f"  DynaBench Evaluation ({mode_name} mode)")
    print(f"{'=' * 50}")
    print(f"  F1 Score    : {metrics['f1']:.1f}%")
    print(f"  Precision   : {metrics['precision']:.1f}%")
    print(f"  Recall      : {metrics['recall']:.1f}%")
    print(f"  Accuracy    : {metrics['accuracy']:.1f}%")
    print(f"  Valid/Total : {metrics['n_valid']}/{len(predictions)}")
    print(f"{'=' * 50}")

    return metrics
