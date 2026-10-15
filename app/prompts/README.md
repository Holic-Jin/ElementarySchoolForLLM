# System prompts (English only)

[한국어](#한국어) · [English](#english) · [日本語](#日本語)

These `.md` files are **runtime system prompts** loaded by the app (`load_prompt`).
They intentionally stay in **English only**. Do not add `*.ko.md` / `*.ja.md` variants for these filenames — the UI already supports Korean, English, and Japanese, and training data is generated in the language chosen in the plan.

User-facing documentation (README, changelog, third-party notices) is translated separately at the repository root.

## English

### Why English only
Small and low-end LLMs follow English instructions more reliably. Role prompts (orchestrator, teachers, judge) are therefore kept in English regardless of the UI language.

### GitHub / content policy notes
- Prompts describe a lawful fine-tuning workflow for a local student model.
- They do not request secrets, credential theft, malware, or other disallowed activity.
- Third-party product names that appear elsewhere in the project are trademarks of their owners; these prompt files do not claim affiliation.

### Files
| File | Role |
|---|---|
| `orchestrator_interview.md` | Plan interview with the user |
| `orchestrator_finalize.md` | Finalize the plan as English JSON |
| `orchestrator_translate.md` | Translate plan values to English |
| `orchestrator_spec.md` | Write the data specification JSON |
| `orchestrator_judge.md` | Blind A/B quality judge |
| `orchestrator_compare.md` | Before/after student answer comparison |
| `teacher1_generate.md` | Teacher 1: create question + answer |
| `teacher1_answer.md` | Teacher 1: write model answer |
| `teacher2_answer.md` | Teacher 2: independent answer from the question only |

---

## 한국어

### 영어 전용인 이유
소형·저사양 LLM은 영어 지시를 더 안정적으로 따릅니다. UI 언어와 관계없이 역할 프롬프트(오케스트레이터·교사·심사)는 영어로 유지합니다.

### GitHub / 콘텐츠 정책
- 프롬프트는 로컬 학생 모델을 합법적으로 파인튜닝하는 흐름만 다룹니다.
- 비밀정보 탈취, 악성코드, 기타 금지 활동을 요청하지 않습니다.
- 프로젝트 다른 문서의 제3자 제품명은 각 소유자 상표이며, 이 프롬프트는 제휴를 주장하지 않습니다.

### 파일
위 표와 동일합니다. 파일명을 바꾼 `*.ko.md` 변형을 추가하지 마세요. 앱이 이름으로 로드합니다.

---

## 日本語

### 英語専用の理由
小型・低スペックLLMは英語の指示に安定して従います。UI言語に関係なく、役割プロンプト（オーケストレータ・教師・判定）は英語のままにします。

### GitHub / コンテンツポリシー
- プロンプトはローカル生徒モデルの合法的なファインチューニング手順のみを扱います。
- 秘密情報の窃取、マルウェア、その他の禁止行為を求めません。
- 他ドキュメントに出る第三者製品名は各所有者の商標であり、これらのプロンプトは提携を主張しません。

### ファイル
上表と同じです。アプリがファイル名で読み込むため、`*.ja.md` などの別名バリエーションは追加しないでください。
