# Changelog

[한국어](CHANGELOG.ko.md) · [English](CHANGELOG.md) · [日本語](CHANGELOG.ja.md)

## 0.1.0 — 2026-10-11 (first release)

Author: HolicJin · License: Apache-2.0

### Features
- Native PySide6 app for Apple Silicon Macs (macOS 13.5+); one-click first-run setup (`start.command`) with no Homebrew/Xcode/admin rights (uv + Python 3.12 + packages + Ollama, SHA256/code-signature verified).
- Roles: orchestrator, Teacher 1 (question + answer 1), Teacher 2 (independent answer 2 from the question only), local student LLM.
- Plan interview with the orchestrator; the confirmed plan is always English for teachers/judge, with a summary in the user's language.
- Blind A/B judging (random order, hidden authors, swap re-check), contradiction handling, regeneration, top-up rounds, preference pairs for DPO/ORPO.
- Hardware gate before Teacher 1: per engine/mode memory and disk estimate; on failure an error dialog cancels everything.
- Training engines: MLX-LM (default), PyTorch + PEFT (MPS), mlx-lm-lora (experimental). Modes: LoRA, QLoRA, DoRA, rsLoRA, PiSSA, full fine-tuning, DPO, ORPO. Research-based defaults with an automatic learning-rate sweep.
- Providers by URL + API key + model in Chat Completions, Messages, or generateContent format, auto-detected from the URL. Pasted URLs are normalized, API format auto-corrected, Responses-only models handled; terms-of-use confirmation for remote APIs.
- Streaming responses with activity lights (working / stalled or looping / done); repeated output is detected and retried.
- Fast two-step connection test with live elapsed time.
- Progress per stage with ETA plus a filterable log; pause / cancel / resume from checkpoints.
- Results: quality stats, before/after comparison, student test chat, Ollama export, provenance record.
- UI in Korean, English, and Japanese; all system prompts in English.
- Data stored outside the program (`~/Documents/ElementarySchoolForLLM`), split by stage, with a Clean-up menu that moves stages to the Trash.

### Known limitations
- Ollama export (`ollama create`) and the llama.cpp install button were not exercised end-to-end during development.
- Remote providers were verified with recorded/mock responses, not with live API keys.
- Small (~3B) orchestrator models judge leniently; a 14B+ local model or a remote API is recommended.

### Documentation
- README, changelog, and third-party notices are available in Korean, English, and Japanese.
- Third-party names are used descriptively only; see the Trademarks section in the README. This project is not affiliated with those vendors.
