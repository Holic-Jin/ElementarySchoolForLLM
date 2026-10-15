# 변경 기록

[한국어](CHANGELOG.ko.md) · [English](CHANGELOG.md) · [日本語](CHANGELOG.ja.md)

## 0.1.0 — 2026-10-11 (첫 공개)

제작: HolicJin · 라이선스: Apache-2.0

### 기능
- Apple Silicon Mac(macOS 13.5+)용 PySide6 네이티브 앱. `start.command` 한 번으로 첫 설치(Homebrew·Xcode·관리자 권한 불필요; uv + Python 3.12 + 패키지 + Ollama, SHA256/코드 서명 검증).
- 역할: 오케스트레이터, 교사1(질문 + 답변1), 교사2(질문만 보고 독립 답변2), 로컬 학생 LLM.
- 오케스트레이터와 학습방안 인터뷰. 확정 방안은 교사·심사용으로 항상 영어이며, 사용자 언어 요약이 붙습니다.
- 블라인드 A/B 평가(순서 무작위, 작성자 비공개, 근소 시 순서 교체 재평가), 모순 처리, 재생성, 보충 라운드, DPO/ORPO용 선호도 쌍.
- 교사1 전 하드웨어 게이트: 엔진·모드별 메모리·디스크 추정. 실패 시 에러 대화상자로 전체 취소.
- 학습 엔진: MLX-LM(기본), PyTorch + PEFT(MPS), mlx-lm-lora(실험적). 모드: LoRA, QLoRA, DoRA, rsLoRA, PiSSA, 전체 파인튜닝, DPO, ORPO. 연구 기반 기본값과 자동 학습률 탐색.
- URL + API 키 + 모델명으로 Chat Completions / Messages / generateContent 형식 연결(URL에서 자동 감지). 붙여넣은 URL 정규화, API 형식 자동 보정, Responses 전용 모델 처리, 원격 API 이용약관 확인.
- 스트리밍 응답과 활동 표시등(작업 중 / 정체·루프 / 완료). 반복 출력 감지 후 재시도.
- 경과 시간이 보이는 빠른 2단계 연결 테스트.
- 단계별 진행률·ETA·필터 가능한 로그. 일시정지 / 취소 / 체크포인트에서 재개.
- 결과: 품질 통계, 학습 전후 비교, 학생 테스트 채팅, Ollama 내보내기, 출처(provenance) 기록.
- UI 한국어·영어·일본어. 시스템 프롬프트는 모두 영어.
- 데이터는 프로그램 밖(`~/Documents/ElementarySchoolForLLM`)에 단계별로 저장. 정리 메뉴는 해당 단계를 휴지통으로 이동.

### 알려진 제한
- Ollama 내보내기(`ollama create`)와 llama.cpp 설치 버튼은 개발 중 끝까지 실행해 보지 못했습니다.
- 원격 제공자는 기록/모의 응답으로 확인했으며, 실제 API 키로는 검증하지 않았습니다.
- 약 3B급 오케스트레이터는 채점이 관대합니다. 14B 이상 로컬 모델 또는 원격 API를 권장합니다.

### 문서
- README, 변경 기록, 제3자 고지를 한국어·영어·일본어로 제공합니다.
- 제3자 명칭은 설명 목적으로만 사용합니다. README의 상표 절을 참고하세요. 해당 업체와 제휴 관계가 없습니다.
