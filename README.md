# ElementarySchoolForLLM

> A native **Apple Silicon** app that plans a training curriculum with an orchestrator AI, has two teacher AIs generate data that the orchestrator cross-checks, and then fine-tunes a local LLM (the student) on your Mac.

Author: **HolicJin** · License: **Apache-2.0** · [한국어](README.ko.md)

## ⚠️ API terms of use

The orchestrator and teachers can use **remote APIs** (any AI service reached by URL + API key + model name).
**Using API outputs to train other models may be prohibited or restricted by some providers' terms.**
Check each provider's usage policy before use — the app asks you to confirm whenever a remote API is configured.
Fine-tuned models also inherit the **license of the student base model** (e.g. Llama/Gemma community licenses).

## How it works

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
- All system prompts sent to the AIs are in **English**, which small/low-end LLMs follow better. Training data is generated in the language chosen in the plan.

## Data folder

Everything that is not the program — plans, intermediate data, training results, downloaded student models — lives in a separate data folder (default `~/Documents/ElementarySchoolForLLM`, changeable in Settings). Each project is split into `1_plan`, `2_generated`, `3_dataset`, `4_training`, `5_reports`; the Results tab's **Clean up…** menu moves a stage and everything after it to the Trash (recoverable) so it can be rebuilt. The confirmed plan is always written in English for the teachers and judge, with a summary in your language appended.

## Troubleshooting HTTP 404

Use the base URL without the endpoint (e.g. `https://your-server/v1`, not `.../chat/completions`), keep the API format on *Auto-detect*, and check the model name with [Model list]. The app auto-cleans pasted URLs, auto-corrects the API format when the host makes it obvious, switches to `/responses` for Responses-only models, and shows the exact requested URL in 404 errors.

## Requirements

- Apple Silicon Mac (M1+), macOS 13.5+ — **Intel Macs are not supported**
- Internet on first launch. Nothing else: no Homebrew, Xcode, or admin rights.

## Install & run

```bash
git clone https://github.com/Holic-Jin/ElementarySchoolForLLM.git
```

Double-click `start.command`. On first run it installs into `~/.ElementarySchoolForLLM`: **uv** (SHA256-verified), **Python 3.12** + packages (PySide6, MLX, mlx-lm…), **Ollama** (code-signature verified, into `~/Applications`), and an app launcher `~/Applications/ElementarySchoolForLLM.app`.

If macOS blocks it, right-click `start.command` → **Open**.

## Development

```bash
~/.ElementarySchoolForLLM/venv/bin/python -m unittest discover -s tests -t .
~/.ElementarySchoolForLLM/venv/bin/python -m tests.e2e_run --mode lora   # mock LLM + tiny real student
```

## Security

API keys and HF tokens live in the macOS Keychain (never in config files; masked in logs). Generated data and models are git-ignored. Remote code (`*.py`) is never downloaded for student models.

## Disclaimer

Provided "as is". You are responsible for the quality of generated data and models and for complying with AI providers' terms and model licenses.
