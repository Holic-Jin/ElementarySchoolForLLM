# ElementarySchoolForLLM

> オーケストレータと対話しながら学習計画を立て、2人の教師AIが作ったデータをオーケストレータが照合し、Mac上のローカルLLM（生徒）をファインチューニングする **Apple Silicon 専用ネイティブアプリ**

作者: **HolicJin** · ライセンス: **Apache-2.0** · [한국어](README.ko.md) · [English](README.md) · [日本語](README.ja.md)

## API利用規約

オーケストレータと教師は **リモートAPI**（URL + APIキー + モデル名で接続するAIサービス）を利用できます。
**APIの出力を他モデルの学習に使うことは、一部プロバイダの利用規約で禁止または制限されている場合があります。**
利用前に各プロバイダのポリシーを確認してください。アプリはリモートAPIを設定するたびに確認ダイアログを表示します。
ファインチューニング後のモデルは、**生徒ベースモデルのライセンス**（一部のオープンウェイトモデルのコミュニティライセンスなど）にも従います。

本プロジェクトは、Apple、Ollama、Hugging Face、Meta、Google、OpenAI、Anthropic、LM Studio など文書に登場する **第三者との提携・保証・後援関係はありません。** 製品名・会社名は互換性や技術例の説明にのみ用いられ、各商標権者に帰属します。

## しくみ

```
[ユーザー] ⇄ オーケストレータ（Q&Aで学習計画を整理）
              │ 01_learning_plan.md, 03_algorithm.md（草案）
              ▼
         オーケストレータ → 02_data_spec.json（カテゴリ・件数・難易度）
              ▼
         ハード要件チェック（教師1の前）── 不足ならエラー表示 + 全作業キャンセル
              ▼                              ┌─ 生徒モデルダウンロード（並列）
         教師1: 質問 + 回答1 ──キュー──▶ 教師2: 質問だけ見て回答2 ──キュー──▶ オーケストレータ: ブラインドA/B判定
              │                              └─ 学習前ベースライン回答（メモリ余裕時）
              ▼
         学習率スイープ → 生徒学習（LoRA / QLoRA / DoRA / rsLoRA / PiSSA / フルFT / DPO·ORPO）
              ▼
         学習前後の評価 → 結果タブ · テストチャット · Ollama書き出し
```

1. **計画** — オーケストレータと対話し、`01_learning_plan.md` と草案 `03_algorithm.md` を作成。
2. **仕様** — オーケストレータが `02_data_spec.json`（カテゴリ、件数、難易度）を作成。
3. **ハード要件ゲート** — 教師1の開始前にエンジン/モード別のメモリ・ディスク必要量を計算。実行・学習が不可能ならエラーを出し **すべてキャンセル**。
4. **生成（並列・ストリーミング）** — 教師1が質問+回答1、教師2は **質問のみ** で回答2、オーケストレータは **ブラインド** 比較（順序ランダム、著者非表示、僅差なら順序入替で再判定）。矛盾する回答は再生成または除外。採用/不採用ペアは DPO/ORPO 用に保存。生徒モデルは並列ダウンロード。
5. **学習** — 短い学習率スイープの後、MLX-LM（既定）/ PyTorch+PEFT / mlx-lm-lora で LoRA / QLoRA / DoRA / rsLoRA / PiSSA / フルFT / DPO·ORPO。
6. **評価** — オーケストレータによる学習前後比較、テストチャット、Ollama書き出し。

**オーケストレータは品質判定役**なので、可能な限り大きなモデル（ローカル14B以上またはリモートAPI）を推奨します。約3B級は誤答を通しやすいです。

既定値は近年の研究（"LoRA Without Regret", 2025; "Learning Rate Matters: Vanilla LoRA May Suffice", 2026）に沿います。全線形層へLoRA、LoRA学習率 ≈ フルFT × 10を出発点にスイープ、小さなバッチ、データ量に応じたrank。

## 言語

- UI: **한국어 · English · 日本語**（macOS言語に追従。設定 → *언어 / Language / 言語* で変更、再起動後に反映）。
- AIへ送るシステムプロンプトは **すべて英語** です（小型LLMが英語指示に従いやすいため）。学習データ自体は計画で選んだ言語で生成されます。詳細は [`app/prompts/`](app/prompts/)（設計上英語専用）。

## 使い方の流れ

1. **設定** — オーケストレータ・教師1・教師2のURL、APIキー、モデル名を入力（API形式は自動検出が既定）。教師1と教師2は **別モデル** を推奨。生徒モデル（Hugging Face IDまたはローカルフォルダ）と学習エンジン/方式を選択。
2. **学習計画** — プロジェクトを作り、質問に答えて **[計画を確定]**。
3. **パイプライン** — **[開始]**。段階別進捗（残り時間）とログを確認。一時停止・キャンセル・再開可能。
4. **結果** — 品質統計、学習前後比較、生徒テストチャット、Ollama書き出し。
5. **プロンプト** — 役割別プロンプトの確認・編集（英語のシステムプロンプト）。

## データフォルダ

プログラム本体以外（計画、中間データ、学習結果、ダウンロードした生徒モデル）は別データフォルダに保存します（既定 `~/Documents/ElementarySchoolForLLM`、設定で変更可）。各プロジェクトは `1_plan` / `2_generated` / `3_dataset` / `4_training` / `5_reports` に分かれ、結果タブの **クリーンアップ…** は段階以降をゴミ箱へ移して再構築できます。確定計画は教師・判定用に常に英語で書かれ、ユーザー言語の要約が付きます。

```
ElementarySchoolForLLM/
  models/                         ダウンロードした生徒モデル
  projects/<project>/
    project.json                  進行状態
    1_plan/       00_plan_chat.json · 01_learning_plan.md · 02_data_spec.json · 03_algorithm.md
    2_generated/  05_teacher1.jsonl · 06_teacher2.jsonl · 07_quality_report.json · 07a_judgements.jsonl · 07b_preference_pairs.jsonl
    3_dataset/    08_train.jsonl · 08_valid.jsonl · train_data/
    4_training/   adapters/ · 09_training_log.txt · fused_model/
    5_reports/    04_hardware_report.json · 09b_eval_results.json · 10_provenance.md · pipeline_log.txt
```

## HTTP 404 のトラブルシューティング

エンドポイントを除いたベースURLを使ってください（例: `https://your-server/v1`。`.../chat/completions` は不可）。API形式は *自動検出* のまま、モデル名は **モデル一覧** で確認。貼り付けURLの整形、ホストからの形式推定、Responses専用モデルへの切替、404時の実リクエストURL表示に対応します。

## 必要環境

- Apple Silicon Mac（M1以降）、macOS 13.5+ — **Intel Mac は非対応**
- 初回起動時のみインターネット。Homebrew / Xcode / 管理者権限は不要。

## インストールと実行

```bash
git clone https://github.com/Holic-Jin/ElementarySchoolForLLM.git
```

`start.command` をダブルクリック。初回は `~/.ElementarySchoolForLLM` に **uv**（SHA256検証）、**Python 3.12** + パッケージ（PySide6, MLX, mlx-lm など）、**Ollama**（コード署名検証後 `~/Applications`）、ランチャ `~/Applications/ElementarySchoolForLLM.app` を入れます。

macOSにブロックされたら `start.command` を右クリック → **開く**。

## 学習エンジン

| モード | MLX-LM（既定） | PyTorch + PEFT (MPS) | mlx-lm-lora（コミュニティ / 実験的） |
|---|---|---|---|
| LoRA / rsLoRA | ✓ | ✓ | LoRA ✓ |
| QLoRA (4bit) | ✓ | ✗（Macで bitsandbytes 非対応） | ✓ |
| DoRA | ✓ | ✓ | ✓ |
| PiSSA | ✗ | ✓ | ✗ |
| フルファインチューニング | ✓ | ✓ | ✓ |
| DPO / ORPO | ✗ | ✗ | ✓ |

既定で入るのは MLX-LM のみ。他エンジンは設定で初回選択時にインストール（PyTorch は約2GB）。

## ローカルLLMバックエンド

Chat Completions形式のベースURLで接続: Ollama（`:11434/v1`、自動インストール）、LM Studio（`:1234/v1`、ユーザーが別途インストール）、`mlx_lm.server`（`:8080/v1`、同梱）、llama.cpp server（`:8081/v1`、設定からインストール）。

## 開発

```bash
~/.ElementarySchoolForLLM/venv/bin/python -m unittest discover -s tests -t .
~/.ElementarySchoolForLLM/venv/bin/python -m tests.e2e_run --mode lora   # モックLLM + 小さな実生徒モデル
```

## 既知の制限

- Ollama書き出し（`ollama create`）と llama.cpp インストールボタンは、開発中にエンドツーエンドで未検証です。
- リモートAPIは記録/モック応答形式で確認しており、CIで本番キーは使いません。
- 約3B級オーケストレータは採点が甘いです。14B以上のローカルモデルかリモートAPIを推奨します。

## セキュリティとプライバシー

APIキーと Hugging Face トークンは macOS キーチェーンに保存（設定ファイルには書かず、ログではマスク）。生成データとモデルは gitignore。生徒モデル用にリモートコード（`*.py`）はダウンロードしません。

## 他言語の文書

| 文書 | English | 한국어 | 日本語 |
|---|---|---|---|
| README | [README.md](README.md) | [README.ko.md](README.ko.md) | [README.ja.md](README.ja.md) |
| 変更履歴 | [CHANGELOG.md](CHANGELOG.md) | [CHANGELOG.ko.md](CHANGELOG.ko.md) | [CHANGELOG.ja.md](CHANGELOG.ja.md) |
| 第三者通知 | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | [THIRD_PARTY_NOTICES.ko.md](THIRD_PARTY_NOTICES.ko.md) | [THIRD_PARTY_NOTICES.ja.md](THIRD_PARTY_NOTICES.ja.md) |

## 免責

本ソフトウェアは「現状有姿」で提供されます。生成データと学習モデルの品質、ならびに各AIプロバイダの規約とモデルライセンスの遵守は利用者の責任です。

## 商標

Apple、Mac、macOS、Apple Silicon は Apple Inc. の商標です。Ollama、Hugging Face、PyTorch、Meta、Google、OpenAI、Anthropic、LM Studio その他の製品名は各所有者の商標であり、本リポジトリでの使用は説明目的のみで、保証を意味しません。
