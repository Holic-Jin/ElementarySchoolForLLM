"""10_provenance.md: 사용한 모델·제공자·라이선스·데이터 출처를 자동 기록(사용자 본인 확인용)."""
from __future__ import annotations

from ..i18n import T

import datetime as _dt
from pathlib import Path

from .config import ROLE_NAMES, ROLES, AppConfig
from .providers import is_local_url, tos_url_for


def hf_license(model_id: str, token: str = "") -> str:
    if Path(model_id).expanduser().is_dir():
        return T('로컬 모델 (라이선스 직접 확인 필요)')
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, token=token or None)
        lic = getattr(info, "card_data", None) and info.card_data.get("license")
        if not lic:
            lic = next((t.split(":", 1)[1] for t in (info.tags or []) if t.startswith("license:")), "")
        return lic or T('표기 없음 (모델 페이지에서 직접 확인)')
    except Exception:
        return T('조회 실패 (모델 페이지에서 직접 확인)')


def write_provenance(path: Path, cfg: AppConfig, plan: dict, stats: dict, train_cfg: dict, engine: str) -> None:
    remote = [r for r in ROLES if not is_local_url(cfg.roles[r].base_url)]
    lines = [T('# 출처 기록 — {get}', get=plan.get('title', '')), "",
             T('- 생성 시각: {now:%Y-%m-%d %H:%M}', now=_dt.datetime.now()),
             T('- 생성 프로그램: ElementarySchoolForLLM (HolicJin, Apache-2.0)'), ""]
    if remote:
        lines += [T('> **주의:** 원격 API로 생성한 데이터가 포함되어 있습니다. 이 데이터나 학습한 모델을 공개·배포하기 전에 해당 제공자의 이용약관(출력물로 다른 모델을 학습하는 것에 대한 제한)을 반드시 확인하세요.'), ""]
    lines += [T('## 사용한 AI'), T('| 역할 | 제공자 주소 | 모델 | 구분 | 이용약관 |'), "|---|---|---|---|---|"]
    for r in ROLES:
        rc = cfg.roles[r]
        local = is_local_url(rc.base_url)
        lines.append(f"| {ROLE_NAMES[r]} | {rc.base_url} | {rc.model} | {T('로컬') if local else T('원격 API')} | "
                     f"{'-' if local else (tos_url_for(rc.base_url) or T('직접 확인'))} |")
    lic = hf_license(cfg.student.model_id, cfg.student.hf_token)
    lines += ["", T('## 학생 모델'), T('- 베이스 모델: {model_id}', model_id=cfg.student.model_id), T('- 베이스 모델 라이선스: {lic}', lic=lic),
              T('- 파인튜닝한 모델은 베이스 모델의 라이선스를 따릅니다(예: Llama·Gemma 계열은 자체 약관의 표기·사용 제한).'),
              T('- 학습 엔진: {engine}', engine=engine), T('- 학습 설정: {train_cfg}', train_cfg=train_cfg), "", T('## 데이터')]
    lines += [f"- {k}: {v}" for k, v in stats.items()]
    lines += ["", T('## 품질 관리'), T('- 교사1이 질문+답변1, 교사2가 질문만 보고 독립 답변2 생성'),
              T('- 오케스트레이터가 답변 순서를 무작위로 섞고 작성자를 숨긴 채 비교 평가(블라인드 A/B)'),
              T('- 두 답변이 서로 모순되면 정답 불확실로 보고 재생성 또는 제외'), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
