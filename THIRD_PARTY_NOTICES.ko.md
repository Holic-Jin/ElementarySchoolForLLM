# 제3자 고지

[한국어](THIRD_PARTY_NOTICES.ko.md) · [English](THIRD_PARTY_NOTICES.md) · [日本語](THIRD_PARTY_NOTICES.ja.md)

ElementarySchoolForLLM(Copyright 2026 HolicJin, Apache-2.0)은 이 저장소에 **제3자 코드를 포함하지 않습니다.**
아래 구성 요소는 첫 실행 시(또는 선택 엔진을 고를 때) 공식 출처에서 내려받아 각 라이선스에 따라 사용됩니다.

이 프로젝트는 아래 구성 요소·상표의 소유자와 **제휴·보증·후원 관계가 없습니다.**
명칭은 사용자가 설치하거나 연결할 수 있는 소프트웨어를 식별하기 위해서만 사용합니다.

| 구성 요소 | 라이선스 | 사용 방식 |
|---|---|---|
| [PySide6 / Qt for Python](https://pypi.org/project/PySide6/) | LGPL-3.0 (또는 GPL-2.0/3.0) | PyPI 휠에서 설치, 동적 링크, 재배포하지 않음. 사용자가 교체 가능. |
| [MLX](https://github.com/ml-explore/mlx) | MIT | 기본 학습/추론 엔진 |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | 기본 학습/추론 엔진 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | 모델 다운로드 |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause | HTTP 클라이언트 |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | 하드웨어/메모리 감지 |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | 학습 설정 파일 |
| [keyring](https://github.com/jaraco/keyring) | MIT | macOS Keychain 접근 |
| [uv](https://github.com/astral-sh/uv) | MIT 또는 Apache-2.0 | Python/패키지 설치(부트스트랩) |
| [PyTorch](https://github.com/pytorch/pytorch) *(선택)* | BSD-3-Clause | 선택 PyTorch 엔진 |
| [transformers](https://github.com/huggingface/transformers) *(선택)* | Apache-2.0 | 선택 PyTorch 엔진 |
| [PEFT](https://github.com/huggingface/peft) *(선택)* | Apache-2.0 | 선택 PyTorch 엔진 |
| [accelerate](https://github.com/huggingface/accelerate), [safetensors](https://github.com/huggingface/safetensors) *(선택)* | Apache-2.0 | 선택 PyTorch 엔진 |
| [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) *(선택)* | Apache-2.0 | 선택 커뮤니티 엔진 |
| [Ollama](https://github.com/ollama/ollama) | MIT | 로컬 LLM 백엔드, ollama.com에서 다운로드 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) *(선택)* | MIT | 선택 로컬 LLM 백엔드, GitHub 릴리스에서 다운로드 |

LM Studio는 독점 소프트웨어이며, 이 앱이 내려받거나 설치하지 않습니다. 사용하려면 사용자가 직접 설치해야 합니다.

모델(학생 베이스 모델, 교사/오케스트레이터 모델)은 이 소프트웨어의 일부가 **아닙니다.** 각 모델의 라이선스와 약관은 사용자가 지켜야 합니다.

문서에 나오는 Apple, Mac, macOS 및 기타 제3자 명칭은 각 소유자의 상표이며, 호환성 설명 목적으로만 사용됩니다.
