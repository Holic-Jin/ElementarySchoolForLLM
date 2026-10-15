# 第三者通知

[한국어](THIRD_PARTY_NOTICES.ko.md) · [English](THIRD_PARTY_NOTICES.md) · [日本語](THIRD_PARTY_NOTICES.ja.md)

ElementarySchoolForLLM（Copyright 2026 HolicJin, Apache-2.0）は、このリポジトリに **第三者コードを同梱しません。**
下記コンポーネントは初回実行時（または任意エンジン選択時）に公式ソースから取得され、各ライセンスの下で使用されます。

本プロジェクトは、下記コンポーネントや商標の所有者と **提携・保証・後援関係はありません。**
名称は、ユーザーがインストールまたは接続し得るソフトウェアを識別するためだけに用います。

| コンポーネント | ライセンス | 用途 |
|---|---|---|
| [PySide6 / Qt for Python](https://pypi.org/project/PySide6/) | LGPL-3.0（または GPL-2.0/3.0） | PyPI ホイールからインストール、動的リンク、再配布なし。ユーザーが置換可能。 |
| [MLX](https://github.com/ml-explore/mlx) | MIT | 既定の学習/推論エンジン |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | 既定の学習/推論エンジン |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | モデルダウンロード |
| [httpx](https://github.com/encode/httpx) | BSD-3-Clause | HTTP クライアント |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | ハードウェア/メモリ検出 |
| [PyYAML](https://github.com/yaml/pyyaml) | MIT | 学習設定ファイル |
| [keyring](https://github.com/jaraco/keyring) | MIT | macOS キーチェーンアクセス |
| [uv](https://github.com/astral-sh/uv) | MIT または Apache-2.0 | Python/パッケージインストーラ（ブートストラップ） |
| [PyTorch](https://github.com/pytorch/pytorch) *(任意)* | BSD-3-Clause | 任意の PyTorch エンジン |
| [transformers](https://github.com/huggingface/transformers) *(任意)* | Apache-2.0 | 任意の PyTorch エンジン |
| [PEFT](https://github.com/huggingface/peft) *(任意)* | Apache-2.0 | 任意の PyTorch エンジン |
| [accelerate](https://github.com/huggingface/accelerate), [safetensors](https://github.com/huggingface/safetensors) *(任意)* | Apache-2.0 | 任意の PyTorch エンジン |
| [mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) *(任意)* | Apache-2.0 | 任意のコミュニティエンジン |
| [Ollama](https://github.com/ollama/ollama) | MIT | ローカルLLMバックエンド（ollama.com から取得） |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) *(任意)* | MIT | 任意のローカルLLMバックエンド（GitHub Releases から取得） |

LM Studio はプロプライエタリソフトウェアであり、本アプリはダウンロードもインストールも行いません。利用する場合はユーザーが自らインストールします。

モデル（生徒ベースモデル、教師/オーケストレータモデル）は本ソフトウェアの一部では **ありません。** 各モデルのライセンスと規約は利用者が遵守する必要があります。

文書に登場する Apple、Mac、macOS その他の第三者名称は各所有者の商標であり、互換性の説明目的でのみ使用します。
