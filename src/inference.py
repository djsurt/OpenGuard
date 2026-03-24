"""Interactive single-sample inference for DynaGuard.

Provides a convenience function to run the trained DynaGuard model on
a custom policy + transcript pair.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.data import SYSTEM_PROMPT


def run_dynaguard(
    policy: str,
    transcript: str,
    config: Config,
    use_cot: bool = False,
    max_new_tokens: int = 512,
) -> str:
    """Run DynaGuard on a single policy + transcript pair.

    Args:
        policy: Numbered list of rules for the guardian model.
        transcript: Agent output transcript to evaluate.
        config: Pipeline configuration with ``OUTPUT_DIR_MERGED``.
        use_cot: If ``True``, use chain-of-thought reasoning mode.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The model's generated response text.
    """
    model_path = config.OUTPUT_DIR_MERGED

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    mdl.eval()

    formatted_input = (
        f"<rules>\n{policy}\n</rules>\n\n"
        f"<transcript>\n{transcript}\n</transcript>"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_input},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += "<think>\n" if use_cot else "<answer>\n"

    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        output = mdl.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.6, do_sample=True
        )

    response = tok.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return response
