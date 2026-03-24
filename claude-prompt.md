# Claude Code Prompt — Refactor DynaGuard Notebook into `.py` Modules

## Context

I have a Jupyter notebook (`notebook.ipynb`) that implements the full DynaGuard training
pipeline (SFT + GRPO fine-tuning of a causal LM). I want to refactor it so that all logic
lives in Python source files under `src/`, and the notebook becomes a thin orchestration
runner that only imports and calls those modules.

The repository structure and module responsibilities are described in `CLAUDE.md`.
Read `CLAUDE.md` first before making any changes.

---

## Task

Refactor `notebook.ipynb` into the following files. Create each file from scratch — do not
copy-paste cell code verbatim; clean it up as you go.

### Files to create

**`src/__init__.py`** — empty

**`src/config.py`**
- Define a `Config` dataclass (use `@dataclass`) containing every hyperparameter and path
  currently set as bare globals in Cell 3 of the notebook
- Include: `BASE_MODEL_ID`, all SFT/GRPO hyperparameters, LoRA settings, quantization flag,
  output dirs, W&B project name
- Add type hints and a short inline comment on each field explaining its role
- Export a `default_config = Config()` instance at module level

**`src/data.py`**
- `load_raw_datasets(config: Config)` — loads DynaBenchTrain, DynaBenchSafetyMix, DynaBench test
- `build_sft_dataset(config: Config)` — applies `homogenize_policy_feature`, creates 50/50 mix,
  shuffles; returns the concatenated dataset
- `build_grpo_dataset(config: Config)` — same logic but 11k GRPO subset
- `format_for_chat(example, tokenizer, system_prompt: str)` — full conversation format for SFT
- `format_prompt_only(example, tokenizer, system_prompt: str)` — prompt-only for GRPO
- `apply_formatting(sft_dataset, grpo_dataset, tokenizer, system_prompt)` — maps both formatters
  and returns `(sft_formatted, grpo_formatted)`
- Define `SYSTEM_PROMPT` as a module-level constant (move it out of Cell 12)

**`src/model.py`**
- `load_tokenizer(config: Config)` — loads tokenizer, sets pad token, returns tokenizer
- `load_base_model(config: Config)` — loads model with optional 4-bit BnB quantization,
  enables gradient checkpointing, returns model
- `apply_lora(model, config: Config)` — calls `prepare_model_for_kbit_training` if needed,
  creates `LoraConfig`, returns `(model, peft_config)`

**`src/reward.py`**
- `extract_answer(text: str) -> str` — exact logic from Cell 23
- `dynaguard_reward_fn(completions, ground_truth_label, **kwargs) -> list[float]` — exact
  logic from Cell 23

**`src/sft.py`**
- `build_sft_trainer(model, tokenizer, dataset, peft_config, config: Config)` — builds and
  returns `SFTTrainer` with `SFTConfig` as configured in Cell 20
- `run_sft(trainer, tokenizer, config: Config)` — runs training, saves model + tokenizer,
  prints training loss, returns `train_result`

**`src/grpo.py`**
- `load_sft_checkpoint(config: Config)` — loads model + tokenizer from `config.OUTPUT_DIR_SFT`
  (handles both LoRA and full fine-tune)
- `build_grpo_trainer(model, tokenizer, dataset, config: Config)` — builds `GRPOTrainer` with
  `GRPOConfig` as in Cell 24; note the `GRPO_BATCH_SIZE = 60` override
- `run_grpo(trainer, tokenizer, config: Config)` — runs GRPO, saves checkpoint
- `merge_lora(config: Config)` — merges LoRA adapter if `config.USE_LORA`; otherwise copies
  GRPO dir to merged dir

**`src/evaluate.py`**
- `evaluate_dynaguard(model_path, test_dataset, config: Config, use_cot: bool = True)` —
  exact logic from Cell 29 (batch eval loop, metrics dict)

**`src/inference.py`**
- `run_dynaguard(policy: str, transcript: str, config: Config, use_cot: bool = False) -> str`
  — exact logic from Cell 32 but accepting explicit args

**`requirements.txt`**
```
transformers>=4.45.0
trl>=0.12.0
peft>=0.13.0
accelerate
bitsandbytes
datasets
huggingface_hub
safetensors
scikit-learn
wandb
nbstripout
```

**`.gitignore`**
```
*.h5
*.bin
*.safetensors
__pycache__/
.ipynb_checkpoints/
dynaguard-sft/
dynaguard-grpo/
dynaguard-merged/
wandb/
```

---

### Rewrite `notebook.ipynb`

Replace the existing cells with the following structure. Keep markdown section headers.
Remove all logic from code cells — they should only import and call `src/` functions.

```
Cell 0  [markdown] — Title + pipeline overview (keep existing)
Cell 1  [code]     — pip install + nbstripout --install
Cell 2  [code]     — Mount Google Drive + add src/ to sys.path
Cell 3  [markdown] — "1. Configuration"
Cell 4  [code]     — from src.config import Config; cfg = Config()
                     (user edits Config fields here if needed)
Cell 5  [markdown] — "2. Load & Prepare Datasets"
Cell 6  [code]     — from src.data import ...; datasets = load_raw_datasets(cfg)
Cell 7  [code]     — sft_ds = build_sft_dataset(cfg); grpo_ds = build_grpo_dataset(cfg)
Cell 8  [markdown] — "3. Tokenizer & Chat Formatting"
Cell 9  [code]     — from src.model import load_tokenizer; tok = load_tokenizer(cfg)
Cell 10 [code]     — from src.data import apply_formatting, SYSTEM_PROMPT
                     sft_fmt, grpo_fmt = apply_formatting(sft_ds, grpo_ds, tok, SYSTEM_PROMPT)
Cell 11 [markdown] — "4. Load Base Model"
Cell 12 [code]     — from src.model import load_base_model, apply_lora
                     model = load_base_model(cfg)
                     model, peft_config = apply_lora(model, cfg)
Cell 13 [markdown] — "5. Stage 1 — SFT"
Cell 14 [code]     — gc.collect(); torch.cuda.empty_cache()
Cell 15 [code]     — from src.sft import build_sft_trainer, run_sft
                     trainer = build_sft_trainer(model, tok, sft_fmt, peft_config, cfg)
                     run_sft(trainer, tok, cfg)
Cell 16 [markdown] — "6. Stage 2 — GRPO"
Cell 17 [code]     — from src.grpo import load_sft_checkpoint, build_grpo_trainer, run_grpo
                     grpo_model, grpo_tok = load_sft_checkpoint(cfg)
                     grpo_trainer = build_grpo_trainer(grpo_model, grpo_tok, grpo_fmt, cfg)
                     run_grpo(grpo_trainer, grpo_tok, cfg)
Cell 18 [markdown] — "7. Merge & Save"
Cell 19 [code]     — from src.grpo import merge_lora; merge_lora(cfg)
Cell 20 [markdown] — "8. Evaluation"
Cell 21 [code]     — from src.evaluate import evaluate_dynaguard
                     metrics_cot  = evaluate_dynaguard(cfg.OUTPUT_DIR_MERGED, dataset_test, cfg, use_cot=True)
                     metrics_fast = evaluate_dynaguard(cfg.OUTPUT_DIR_MERGED, dataset_test, cfg, use_cot=False)
Cell 22 [markdown] — "9. Interactive Inference"
Cell 23 [code]     — from src.inference import run_dynaguard; run_dynaguard(..., cfg)
Cell 24 [markdown] — "10. (Optional) Push to Hub"
Cell 25 [code]     — push-to-hub code block (keep commented out)
```

---

## Constraints & Quality Requirements

1. **No logic in the notebook** — if a cell contains more than ~5 lines of non-import code,
   that logic belongs in `src/`.
2. **Pass `config` explicitly** — no global variables accessed across module boundaries.
3. **Type hints** on every public function signature.
4. **Docstrings** (Google style) on every public function.
5. **Do not break the pipeline** — the refactored code must be functionally equivalent to
   the original notebook.
6. **Preserve comments** from the original notebook that explain paper references
   (e.g. "Paper Section 3.4", "Table 15/16").
7. After creating all files, run a quick syntax check:
   ```bash
   python -m py_compile src/config.py src/data.py src/model.py \
       src/reward.py src/sft.py src/grpo.py src/evaluate.py src/inference.py
   ```
   Fix any errors before finishing.
