# ElementarySchoolForLLM

> A native **Apple Silicon** app that plans a training curriculum with an orchestrator AI, has two teacher AIs generate data that the orchestrator cross-checks, and then fine-tunes a local LLM (the student) on your Mac.

Author: **HolicJin** · License: **Apache-2.0** · [한국어](README.ko.md) · [English](README.md) · [日本語](README.ja.md)

## API terms of use

The orchestrator and teachers can use **remote APIs** (any AI service reached by URL + API key + model name).
**Using API outputs to train other models may be prohibited or restricted by some providers' terms.**
Check each provider's usage policy before use — the app asks you to confirm whenever a remote API is configured.
Fine-tuned models also inherit the **license of the student base model** (for example, community licenses that apply to some open-weight models).

This project is **not affiliated with, endorsed by, or sponsored by** Apple, Ollama, Hugging Face, Meta, Google, OpenAI, Anthropic, LM Studio, or any other third-party named in the docs. Product and company names are used only to describe compatibility or technical examples and remain the trademarks of their respective owners.

## How it works

```
[User] ⇄ Orchestrator (clarifies the training plan by Q&A)
              │ 01_learning_plan.md, 03_algorithm.md (draft)
              ▼
         Orchestrator → 02_data_spec.json (categories, counts, difficulty)
              ▼
         Hardware gate (before Teacher 1) ── if insufficient: error dialog + cancel all work
              ▼                              ┌─ student model download (parallel)
         Teacher 1: question + answer 1 ──queue──▶ Teacher 2: question only → answer 2 ──queue──▶ Orchestrator: blind A/B judge
              │                              └─ pre-training baseline answers (when memory allows)
              ▼
         Learning-rate sweep → student training (LoRA / QLoRA / DoRA / rsLoRA / PiSSA / full FT / DPO·ORPO)
              ▼
         Before/after evaluation → Results tab · test chat · Ollama export
```

1. **Plan** — chat with the orchestrator; it asks questions and writes `01_learning_plan.md` and a draft `03_algorithm.md`.
2. **Spec** — the orchestrator writes `02_data_spec.json` (categories, counts, difficulty).
3. **Hardware gate** — before teacher 1 starts, required memory/disk is computed per engine and mode. If the Mac can't run and train the student, an error dialog is shown and **everything is cancelled**.
4. **Generate (parallel, streaming)** — teacher 1 writes question + answer 1; teacher 2 sees **only the question** and writes answer 2; the orchestrator compares both **blind** (random order, authors hidden, re-judged with swapped order when close). Contradicting answers are regenerated or dropped. Winner/loser pairs are saved for DPO/ORPO. The student model downloads in parallel.
5. **Train** — short learning-rate sweep, then LoRA / QLoRA / DoRA / rsLoRA / PiSSA / full fine-tuning / DPO·ORPO with MLX-LM (default), PyTorch+PEFT, or mlx-lm-lora.
6. **Evaluate** — before/after comparison by the orchestrator, test chat, export to Ollama.

Use the strongest model you can for the **orchestrator** (it is the quality judge): a local 14B+ model or a remote API. Tiny (~3B) judges tend to accept wrong answers.

Defaults follow recent research ("LoRA Without Regret", 2025; "Learning Rate Matters: Vanilla LoRA May Suffice", 2026): LoRA on all linear layers, LoRA LR ≈ 10× full-FT LR as a starting point refined by a sweep, small batches, rank sized to the data.

## Languages

- UI: **한국어 · English · 日本語** (follows the macOS language; change it under Settings → *언어 / Language / 言語*, applied on restart).
- All system prompts sent to the AIs are in **English**, which small/low-end LLMs follow better. Training data is generated in the language chosen in the plan. See [`app/prompts/`](app/prompts/) (English-only by design).

## Usage flow

1. **Settings** — enter URL, API key, and model name for the orchestrator and both teachers (API format defaults to auto-detect). Prefer different models for Teacher 1 and Teacher 2. Choose the student model (Hugging Face ID or local folder) and training engine/mode.
2. **Plan** — create a project, answer the orchestrator’s questions, then press **Confirm plan**.
3. **Pipeline** — press **Start**. Watch stage progress (with ETA) and logs; pause, cancel, or resume from generated data.
4. **Results** — quality stats, before/after comparison, student test chat, Ollama export.
5. **Prompts** — view or edit role prompts (English system prompts).

## Data folder

Everything that is not the program — plans, intermediate data, training results, downloaded student models — lives in a separate data folder (default `~/Documents/ElementarySchoolForLLM`, changeable in Settings). Each project is split into `1_plan`, `2_generated`, `3_dataset`, `4_training`, `5_reports`; the Results tab’s **Clean up…** menu moves a stage and everything after it to the Trash (recoverable) so it can be rebuilt. The confirmed plan is always written in English for the teachers and judge, with a summary in your language appended.

```
ElementarySchoolForLLM/
  models/                         downloaded student models
  projects/<project>/
    project.json                  progress state
    1_plan/       00_plan_chat.json · 01_learning_plan.md · 02_data_spec.json · 03_algorithm.md
    2_generated/  05_teacher1.jsonl · 06_teacher2.jsonl · 07_quality_report.json · 07a_judgements.jsonl · 07b_preference_pairs.jsonl
    3_dataset/    08_train.jsonl · 08_valid.jsonl · train_data/
    4_training/   adapters/ · 09_training_log.txt · fused_model/
    5_reports/    04_hardware_report.json · 09b_eval_results.json · 10_provenance.md · pipeline_log.txt
```

## Troubleshooting HTTP 404

Use the base URL without the endpoint (for example `https://your-server/v1`, not `.../chat/completions`), keep the API format on *Auto-detect*, and check the model name with **Model list**. The app auto-cleans pasted URLs, auto-corrects the API format when the host makes it obvious, switches to `/responses` for Responses-only models, and shows the exact requested URL in 404 errors.

## Requirements

- Apple Silicon Mac (M1+), macOS 13.5+ — **Intel Macs are not supported**
- Internet on first launch. Nothing else: no Homebrew, Xcode, or admin rights.

## Install & run

```bash
git clone https://github.com/Holic-Jin/ElementarySchoolForLLM.git
```

Double-click `start.command`. On first run it installs into `~/.ElementarySchoolForLLM`: **uv** (SHA256-verified), **Python 3.12** + packages (PySide6, MLX, mlx-lm…), **Ollama** (code-signature verified, into `~/Applications`), and an app launcher `~/Applications/ElementarySchoolForLLM.app`.

If macOS blocks it, right-click `start.command` → **Open**.

## Training engines

| Mode | MLX-LM (default) | PyTorch + PEFT (MPS) | mlx-lm-lora (community / experimental) |
|---|---|---|---|
| LoRA / rsLoRA | ✓ | ✓ | LoRA ✓ |
| QLoRA (4bit) | ✓ | ✗ (bitsandbytes unsupported on Mac) | ✓ |
| DoRA | ✓ | ✓ | ✓ |
| PiSSA | ✗ | ✓ | ✗ |
| Full fine-tuning | ✓ | ✓ | ✓ |
| DPO / ORPO | ✗ | ✗ | ✓ |

Only MLX-LM is installed by default; other engines install when first selected in Settings (PyTorch is about 2 GB).

## Local LLM backends

Connect with Chat Completions–style base URLs: Ollama (`:11434/v1`, auto-installed), LM Studio (`:1234/v1`, user-installed), `mlx_lm.server` (`:8080/v1`, included), llama.cpp server (`:8081/v1`, install from Settings).

## Development

```bash
~/.ElementarySchoolForLLM/venv/bin/python -m unittest discover -s tests -t .
~/.ElementarySchoolForLLM/venv/bin/python -m tests.e2e_run --mode lora   # mock LLM + tiny real student
```

## Known limitations

- Ollama export (`ollama create`) and the llama.cpp install button were not exercised end-to-end during development.
- Remote APIs were verified with recorded/mock response formats, not with live production keys in CI.
- Small (~3B) orchestrator models judge leniently; a 14B+ local model or a remote API is recommended.

## Security & privacy

API keys and Hugging Face tokens live in the macOS Keychain (never in config files; masked in logs). Generated data and models are git-ignored. Remote code (`*.py`) is never downloaded for student models.

## Documentation in other languages

| Document | English | 한국어 | 日本語 |
|---|---|---|---|
| README | [README.md](README.md) | [README.ko.md](README.ko.md) | [README.ja.md](README.ja.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) | [CHANGELOG.ko.md](CHANGELOG.ko.md) | [CHANGELOG.ja.md](CHANGELOG.ja.md) |
| Third-party notices | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | [THIRD_PARTY_NOTICES.ko.md](THIRD_PARTY_NOTICES.ko.md) | [THIRD_PARTY_NOTICES.ja.md](THIRD_PARTY_NOTICES.ja.md) |

## Disclaimer

Provided "as is". You are responsible for the quality of generated data and models and for complying with AI providers' terms and model licenses.

## Trademarks

Apple, Mac, macOS, and Apple Silicon are trademarks of Apple Inc. Ollama, Hugging Face, PyTorch, Meta, Google, OpenAI, Anthropic, LM Studio, and other product names are trademarks of their respective owners. Use in this repository is descriptive only and does not imply endorsement.
