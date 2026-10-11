# Third-Party Notices

ElementarySchoolForLLM (Copyright 2026 HolicJin, Apache-2.0) does **not** bundle third-party code in this repository.
The components below are downloaded from their official sources on first run (or when the user selects an optional engine)
and are used under their own licenses.

| Component | License | How it is used |
|---|---|---|
| [PySide6 / Qt for Python](https://pypi.org/project/PySide6/) | LGPL-3.0 (or GPL-2.0/3.0) | Installed from PyPI wheels, dynamically linked, not redistributed. Users may replace it. |
| [MLX](https://github.com/ml-explore/mlx) | MIT | Default training/inference engine |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | Default training/inference engine |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model download |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause | HTTP client |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | Hardware/memory detection |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | Training config files |
| [keyring](https://github.com/jaraco/keyring) | MIT | macOS Keychain access |
| [uv](https://github.com/astral-sh/uv) | MIT or Apache-2.0 | Python/package installer (bootstrap) |
| [PyTorch](https://github.com/pytorch/pytorch) *(optional)* | BSD-3-Clause | Optional PyTorch engine |
| [transformers](https://github.com/huggingface/transformers) *(optional)* | Apache-2.0 | Optional PyTorch engine |
| [PEFT](https://github.com/huggingface/peft) *(optional)* | Apache-2.0 | Optional PyTorch engine |
| [accelerate](https://github.com/huggingface/accelerate), [safetensors](https://github.com/huggingface/safetensors) *(optional)* | Apache-2.0 | Optional PyTorch engine |
| [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) *(optional)* | Apache-2.0 | Optional community engine |
| [Ollama](https://github.com/ollama/ollama) | MIT | Local LLM backend, downloaded from ollama.com |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) *(optional)* | MIT | Optional local LLM backend, downloaded from GitHub releases |

LM Studio is proprietary software; it is never downloaded or installed by this app — users install it themselves if they want to use it.

Models (student base models, teacher/orchestrator models) are **not** part of this software. Each model has its own license and terms, which the user must follow.
