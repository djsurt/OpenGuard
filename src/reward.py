"""Reward function and answer extraction for GRPO training.

The reward function scores model completions against ground-truth labels,
giving bonuses for well-formed XML output.
"""

from __future__ import annotations

import re


def extract_answer(text: str) -> str:
    """Extract PASS/FAIL from ``<answer>...</answer>`` tags.

    Args:
        text: Model-generated text that may contain answer tags.

    Returns:
        ``"PASS"``, ``"FAIL"``, or ``"UNKNOWN"`` if no verdict is found.
    """
    match = re.search(r"<answer>\s*(PASS|FAIL)\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: look for bare PASS/FAIL
    match = re.search(r"\b(PASS|FAIL)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def dynaguard_reward_fn(
    completions: list, ground_truth_label: list[str], **kwargs
) -> list[float]:
    """GRPO reward function scoring completions against ground truth.

    Returns:
        A list of reward scores:
        - ``+1.0`` (+ bonuses) if predicted label matches ground truth
        - ``-1.0`` if predicted label does not match
        - ``0.0`` if answer format is malformed (no PASS/FAIL found)

    Also gives small bonuses for well-formed output (proper XML tags).
    """
    rewards: list[float] = []
    for completion, gt_label in zip(completions, ground_truth_label):
        # Extract text from completion
        if isinstance(completion, list):  # Chat format
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        predicted = extract_answer(text)
        gt = gt_label.strip().upper()

        if predicted == "UNKNOWN":
            rewards.append(0.0)
        elif predicted == gt:
            # Correct classification
            reward = 1.0
            # Bonus for well-formed output with proper tags
            if "<answer>" in text and "</answer>" in text:
                reward += 0.1
            if "<think>" in text or "<explanation>" in text:
                reward += 0.1
            rewards.append(reward)
        else:
            # Wrong classification
            rewards.append(-1.0)

    return rewards
