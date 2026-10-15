# ElementarySchoolForLLM

> 오케스트레이터와 대화로 학습방안을 세우고, 두 교사 AI가 만든 데이터를 오케스트레이터가 비교 평가한 뒤, 내 Mac의 로컬 LLM(학생)을 학습시키는 **Apple Silicon 전용 네이티브 앱**

제작: **HolicJin** · 라이선스: **Apache-2.0** · [한국어](README.ko.md) · [English](README.md) · [日本語](README.ja.md)

## API 이용약관

오케스트레이터·교사1·교사2에 **원격 API**(주소 + API 키 + 모델명 방식의 AI 서비스)를 쓸 수 있습니다.
**API로 생성한 출력을 다른 모델 학습에 사용하는 것은 일부 제공자의 이용약관에서 금지되거나 제한될 수 있습니다.**
사용 전에 반드시 각 제공자의 이용정책을 확인하세요. 앱은 원격 API를 쓸 때 확인 창을 띄우며, 확인 책임은 사용자에게 있습니다.
또한 학습한 모델은 **학생 베이스 모델의 라이선스**(예: 일부 오픈웨이트 모델의 커뮤니티 라이선스)를 따릅니다.

이 프로젝트는 Apple, Ollama, Hugging Face, Meta, Google, OpenAI, Anthropic, LM Studio 등 문서에 언급된 **제3자와 제휴·보증·후원 관계가 없습니다.** 제품·회사명은 호환성 설명이나 기술 예시에만 사용되며, 각 상표권자에게 귀속됩니다.

## 동작 방식

```
[사용자] ⇄ 오케스트레이터 (질문·답변으로 학습방안 정리)
              │ 01_learning_plan.md, 03_algorithm.md(초안)
              ▼
         오케스트레이터 → 02_data_spec.json (카테고리·개수·난이도)
              ▼
         사양 검사 (교사1 시작 전) ── 부족하면 에러창 + 모든 작업 취소
              ▼                      ┌─ 학생 모델 다운로드 (병렬)
         교사1: 질문 + 답변1 ──큐──▶ 교사2: 질문만 보고 답변2 ──큐──▶ 오케스트레이터: 블라인드 A/B 비교 평가
              │                      └─ 학생 학습 전 기준 응답 (메모리 여유 시 병렬)
              ▼
         학습률 탐색 → 학생 학습 (LoRA / QLoRA / DoRA / rsLoRA / PiSSA / 전체 파인튜닝 / DPO·ORPO)
              ▼
         학습 전후 비교 평가 → 결과 탭 · 테스트 채팅 · Ollama 내보내기
```

- **두 교사의 독립 답변 교차 검증**: 교사2에게는 교사1의 답을 보여주지 않습니다. 오케스트레이터는 답변 순서를 무작위로 섞고 작성자를 숨긴 채 평가하며, 점수 차가 작으면 순서를 바꿔 다시 평가합니다. 두 답이 서로 모순되면 정답이 불확실한 문항으로 보고 다시 만들거나 제외합니다.
- **선호도 데이터 부산물**: 채택된 답과 탈락한 답의 쌍을 `07b_preference_pairs.jsonl`에 저장해 DPO·ORPO 학습에 쓸 수 있습니다.
- **연구 기반 자동 설정**: "LoRA Without Regret"(2025), "Learning Rate Matters"(2026) 등을 근거로 모든 선형층에 LoRA, LoRA 학습률 ≈ 전체 FT × 10을 출발점으로 짧은 학습률 탐색, 작은 배치, 데이터량에 맞는 rank를 자동 선택합니다. 근거는 `03_algorithm.md`에 기록됩니다.

**오케스트레이터는 품질 심사관**이므로 가능한 한 큰 모델(로컬 14B 이상 또는 원격 API)을 권장합니다. 3B급 소형 모델은 틀린 답도 통과시키는 경향이 있습니다.

## 화면 언어 · 프롬프트 언어

- 화면은 **한국어 · English · 日本語**를 지원합니다. 처음에는 macOS 시스템 언어를 따르고, 설정 탭의 **언어 / Language / 言語**에서 바꿀 수 있습니다(앱을 다시 시작하면 적용).
- AI에게 보내는 시스템 프롬프트는 **모두 영어**입니다. 소형·저사양 LLM은 영어 지시를 더 잘 따르기 때문입니다. 학습 데이터 자체는 학습방안에서 정한 언어로 생성됩니다. 자세한 설명은 [`app/prompts/`](app/prompts/)를 보세요(설계상 영어 전용).

## 사용 순서

1. **설정** — 오케스트레이터·교사1·교사2의 주소, API 키, 모델명을 입력합니다(API 형식은 기본값 '자동 감지'). 교사1과 교사2는 **서로 다른 모델**을 권장합니다. 학생 모델(Hugging Face ID 또는 로컬 폴더)과 학습 엔진·방식을 고릅니다.
2. **학습방안** — 새 프로젝트를 만들고 오케스트레이터의 질문에 답합니다. 오른쪽에 정리 내용이 실시간으로 보입니다. **[방안 확정]**을 누릅니다.
3. **파이프라인** — **[시작]**. 전체·단계별 진행률(남은 시간)과 로그가 함께 표시됩니다. 일시정지·취소할 수 있고, 다시 시작하면 생성된 데이터부터 이어서 진행합니다.
4. **결과** — 품질 통계, 학습 전후 비교, 학생 테스트 채팅, Ollama 내보내기.
5. **프롬프트** — 역할별 프롬프트를 확인·수정할 수 있습니다(영어 시스템 프롬프트).

## 요구 사항

- Apple Silicon Mac (M1 이상), macOS 13.5 이상 — **Intel Mac은 지원하지 않습니다**
- 첫 실행 시 인터넷 연결 (그 외 의존성 없음: Homebrew·Xcode·관리자 권한 불필요)

## 설치 · 실행

```bash
git clone https://github.com/Holic-Jin/ElementarySchoolForLLM.git
```

Finder에서 `start.command`를 더블클릭하세요. 첫 실행 때 사용자 폴더(`~/.ElementarySchoolForLLM`)에 다음을 설치합니다.

1. **uv** (Python 관리자 단일 바이너리, SHA256 검증)
2. **Python 3.12**와 가상환경, 필수 패키지 (PySide6, MLX, mlx-lm 등)
3. **Ollama** (로컬 LLM 실행용, 코드 서명 검증 후 `~/Applications`에 설치)
4. 앱 런처 `~/Applications/ElementarySchoolForLLM.app` — 이후에는 이걸로 실행

> macOS가 "확인되지 않은 개발자" 경고로 막으면 `start.command`를 우클릭 → **열기**를 선택하세요.

## 문제 해결: API 연결 시 HTTP 404

- 주소에는 **엔드포인트를 뺀 기본 주소**를 넣으세요. 예: `https://your-server/v1` (O) / `.../chat/completions` (X). 앱이 자동으로 정리하지만, 특이한 주소는 직접 확인이 필요합니다.
- **API 형식**은 기본값인 '자동 감지'를 쓰세요. 주소로 형식을 알아내고, 알 수 없으면 가장 널리 쓰이는 Chat Completions 형식으로 연결합니다.
- **모델명**이 정확한지 [모델 목록] 버튼으로 확인하세요. 404 오류 메시지에 실제로 요청한 주소가 표시됩니다.
- `/responses`만 지원하는 모델은 자동으로 전환해 호출합니다.

## 학습 엔진

| 모드 | MLX-LM (기본·권장) | PyTorch + PEFT (MPS) | mlx-lm-lora (커뮤니티·실험적) |
|---|---|---|---|
| LoRA / rsLoRA | ✓ | ✓ | LoRA ✓ |
| QLoRA (4bit) | ✓ | ✗ (Mac에서 bitsandbytes 미지원) | ✓ |
| DoRA | ✓ | ✓ | ✓ |
| PiSSA | ✗ | ✓ | ✗ |
| 전체 파인튜닝 | ✓ | ✓ | ✓ |
| DPO / ORPO | ✗ | ✗ | ✓ |

MLX-LM만 기본으로 설치되고, 다른 엔진은 설정 탭에서 처음 선택할 때 설치합니다(PyTorch는 약 2GB).

## 로컬 LLM 백엔드

모두 Chat Completions 형식 주소로 연결합니다: Ollama(`:11434/v1`, 자동 설치), LM Studio(`:1234/v1`, 직접 설치), mlx_lm.server(`:8080/v1`, 기본 포함), llama.cpp server(`:8081/v1`, 설정에서 설치).

## 데이터 폴더 (프로그램과 분리)

학습방안부터 중간 데이터, 학습 결과, 내려받은 학생 모델까지 **프로그램 폴더 밖** 데이터 폴더에 저장합니다. 기본 위치는 `~/Documents/ElementarySchoolForLLM`이고, 설정 탭에서 바꿀 수 있습니다.

```
ElementarySchoolForLLM/
  models/                         내려받은 학생 모델
  projects/<프로젝트>/
    project.json                  진행 상태
    1_plan/       00_plan_chat.json · 01_learning_plan.md(영어 확정안 + 사용자 언어 요약) · 02_data_spec.json · 03_algorithm.md
    2_generated/  05_teacher1.jsonl · 06_teacher2.jsonl · 07_quality_report.json · 07a_judgements.jsonl · 07b_preference_pairs.jsonl
    3_dataset/    08_train.jsonl · 08_valid.jsonl · train_data/
    4_training/   adapters/ · 09_training_log.txt · fused_model/
    5_reports/    04_hardware_report.json · 09b_eval_results.json · 10_provenance.md · pipeline_log.txt
```

**깨진 데이터 정리:** 결과 탭의 **[정리…]** 메뉴에서 *데이터 명세부터 / 생성 데이터부터 / 학습만* 다시 만들거나 프로젝트 전체를 지울 수 있습니다. 파일은 영구 삭제하지 않고 **휴지통**으로 보내므로 Finder에서 되돌릴 수 있습니다.

**영어 학습방안:** 확정된 학습방안은 항상 영어로 만들어 교사·심사에게 전달합니다(영어가 아니면 자동 번역). 학습 데이터 자체는 방안에서 정한 언어로 생성됩니다.

## 개발

```bash
# 단위 테스트
~/.ElementarySchoolForLLM/venv/bin/python -m unittest discover -s tests -t .
# 실제 AI 없이 전체 파이프라인 검증 (가짜 LLM 서버 + 소형 학생 모델 학습)
~/.ElementarySchoolForLLM/venv/bin/python -m tests.e2e_run --mode lora
```

## 알려진 제한

- Ollama 내보내기(`ollama create`)와 llama.cpp 설치 버튼은 개발 중 끝까지 실행해 보지 못했습니다.
- 원격 API는 응답 형식(Chat Completions·Messages·generateContent)을 흉내 낸 테스트로 확인했습니다.
- 3B급 소형 오케스트레이터는 채점이 관대합니다. 14B 이상 로컬 모델이나 원격 API를 권장합니다.

## 보안 · 개인정보

- API 키와 Hugging Face 토큰은 macOS Keychain에 저장되며 설정 파일·로그에 남지 않습니다(로그에서는 가려 표시).
- 학습 데이터와 모델은 저장소에 포함되지 않습니다(`.gitignore`).
- 학생 모델은 원격 코드(`*.py`)를 내려받지 않습니다.

## 다른 언어 문서

| 문서 | English | 한국어 | 日本語 |
|---|---|---|---|
| README | [README.md](README.md) | [README.ko.md](README.ko.md) | [README.ja.md](README.ja.md) |
| 변경 기록 | [CHANGELOG.md](CHANGELOG.md) | [CHANGELOG.ko.md](CHANGELOG.ko.md) | [CHANGELOG.ja.md](CHANGELOG.ja.md) |
| 제3자 고지 | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | [THIRD_PARTY_NOTICES.ko.md](THIRD_PARTY_NOTICES.ko.md) | [THIRD_PARTY_NOTICES.ja.md](THIRD_PARTY_NOTICES.ja.md) |

## 면책

이 소프트웨어는 "있는 그대로" 제공됩니다. 생성된 데이터와 학습된 모델의 품질, 그리고 각 AI 제공자 약관과 모델 라이선스를 지키는 책임은 사용자에게 있습니다.

## 상표

Apple, Mac, macOS, Apple Silicon은 Apple Inc.의 상표입니다. Ollama, Hugging Face, PyTorch, Meta, Google, OpenAI, Anthropic, LM Studio 및 기타 제품명은 각 소유자의 상표이며, 이 저장소에서의 사용은 설명 목적일 뿐 보증을 의미하지 않습니다.
