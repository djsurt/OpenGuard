# CLAUDE.md — DynaGuard Training Pipeline

## Project Overview
This project implements the **DynaGuard** training pipeline from:
> *DynaGuard: A Dynamic Guardian Model With User-Defined Policies* (Hoover et al.)

The pipeline fine-tunes a causal LM (Qwen, Llama, Mistral, Gemma, etc.) in two stages:
1. **Stage 1 — SFT**: Supervised fine-tuning on an 80k-sample DynaBench + Safety mix
2. **Stage 2 — GRPO**: Reinforcement learning with a compliance-based reward

---

## Repository Structure

```
dynaguard/
├── CLAUDE.md                  ← This file
├── notebook.ipynb             ← Thin Colab runner (imports + calls .py modules)
├── src/
│   ├── __init__.py
│   ├── config.py              ← All hyperparameters and paths
│   ├── data.py                ← Dataset loading, mixing, formatting
│   ├── model.py               ← Model + tokenizer loading, LoRA setup
│   ├── sft.py                 ← Stage 1: SFT trainer setup and run
│   ├── grpo.py                ← Stage 2: GRPO trainer setup and run
│   ├── reward.py              ← Reward function and answer extraction
│   ├── evaluate.py            ← Evaluation loop and metrics
│   └── inference.py           ← Interactive inference helper
├── requirements.txt
└── .gitignore
```

---

## Module Responsibilities

### `src/config.py`
- Single source of truth for **all** hyperparameters
- Exports a `Config` dataclass or plain constants
- Change `BASE_MODEL_ID` here to swap the backbone model

### `src/data.py`
- `load_raw_datasets()` — loads DynaBench train/test and Safety mix from HF Hub
- `build_sft_dataset(config)` — creates the 50/50 80k training mix
- `build_grpo_dataset(config)` — creates the 11k GRPO prompt set
- `format_for_chat(example, tokenizer)` — full conversation format for SFT
- `format_prompt_only(example, tokenizer)` — prompt-only format for GRPO

### `src/model.py`
- `load_tokenizer(config)` — loads tokenizer and sets pad token
- `load_base_model(config)` — loads model with optional 4-bit quantization
- `apply_lora(model, config)` — wraps model with LoRA via PEFT

### `src/reward.py`
- `extract_answer(text)` — extracts PASS/FAIL from model output
- `dynaguard_reward_fn(completions, ground_truth_label, **kwargs)` — GRPO reward

### `src/sft.py`
- `build_sft_trainer(model, tokenizer, dataset, config)` — configures SFTTrainer
- `run_sft(trainer, config)` — runs training and saves checkpoint

### `src/grpo.py`
- `load_sft_checkpoint(config)` — loads the SFT model for GRPO
- `build_grpo_trainer(model, tokenizer, dataset, config)` — configures GRPOTrainer
- `run_grpo(trainer, config)` — runs GRPO and saves checkpoint
- `merge_lora(config)` — merges LoRA weights into the base model

### `src/evaluate.py`
- `evaluate_dynaguard(model_path, test_dataset, config, use_cot)` — full eval loop
- Returns a dict with F1, precision, recall, accuracy

### `src/inference.py`
- `run_dynaguard(policy, transcript, config, use_cot)` — single-sample inference

---

## Coding Conventions

- **Python 3.10+**
- Type hints on all public function signatures
- Docstrings on every public function (Google style)
- No global mutable state — pass `config` explicitly
- All paths come from `config.py`, never hardcoded
- All HF Hub calls use `trust_remote_code=True` only when necessary

---

## Running the Pipeline

### In Colab (`notebook.ipynb`)
The notebook is a thin runner — it only:
1. Installs dependencies
2. Mounts Google Drive
3. Imports from `src/`
4. Calls the pipeline functions in order

```python
# All logic lives in src/ — the notebook just orchestrates
from src.config import Config
from src.data import load_raw_datasets, build_sft_dataset, build_grpo_dataset
from src.model import load_tokenizer, load_base_model, apply_lora
from src.sft import build_sft_trainer, run_sft
from src.grpo import load_sft_checkpoint, build_grpo_trainer, run_grpo, merge_lora
from src.evaluate import evaluate_dynaguard
```

### Locally
```bash
pip install -r requirements.txt
python -c "
from src.config import Config
from src import data, model, sft, grpo, evaluate
cfg = Config()
# ... run stages
"
```

---

## Git Workflow

- **Branch per collaborator / experiment**: `alice/data-pipeline`, `bob/grpo-tuning`
- **Never commit notebook outputs** — `nbstripout` is installed as a pre-commit hook
- **Logic changes** go in `src/*.py` files, not in the notebook
- **PRs** merge feature branches into `main` after review
- **Tag stable checkpoints**: `git tag -a v1.0 -m "SFT baseline, 87% F1"`

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `WANDB_PROJECT` | W&B project name (default: `dynaguard-training`) |
| `WANDB_DISABLED` | Set to `"true"` to disable W&B logging |
| `HF_TOKEN` | Hugging Face auth token (use Colab Secrets, never hardcode) |

---

## Key References

- Paper: *DynaGuard: A Dynamic Guardian Model With User-Defined Policies* (Hoover et al.)
- Dataset: `tomg-group-umd/DynaBench` on Hugging Face Hub
- Hyperparameters: Paper Tables 15 & 16
