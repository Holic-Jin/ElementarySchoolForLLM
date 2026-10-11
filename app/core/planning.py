"""학습방안 단계: 오케스트레이터와의 Q&A, 방안 확정 → 01_learning_plan.md + 03_algorithm.md 초안."""
from __future__ import annotations

from ..i18n import T

import asyncio

from . import algorithm, hardware
from .config import AppConfig
from .jsonutil import read_json, write_json
from .project import F_ALGO, F_ALGO_DRAFT, F_CHAT, F_PLAN, Project
from .providers import make_provider, provider_config_from_role
from .roles import Orchestrator, split_summary


def load_chat(project: Project) -> list[dict[str, str]]:
    return read_json(project.path(F_CHAT), []) or []


def save_chat(project: Project, history: list[dict[str, str]]) -> None:
    write_json(project.path(F_CHAT), history)


async def interview_turn(cfg: AppConfig, history: list[dict[str, str]], activity=None) -> tuple[str, str]:
    """history의 마지막은 사용자 메시지. 오케스트레이터 응답(본문, 요약)을 반환."""
    async with make_provider(provider_config_from_role(cfg.roles["orchestrator"]), activity=activity) as p:
        text = await Orchestrator(p).interview(history, cfg.student.model_id)
    return split_summary(text)


def plan_markdown(plan: dict) -> str:
    """확정 학습방안 문서. 교사에게 전달되는 본문은 영어이고, 맨 아래에 사용자 언어 요약을 붙인다."""
    crit = "\n".join(f"- {c}" for c in plan.get("evaluation_criteria", []))
    head = [f"# {plan.get('title', 'Training plan')}", "",
            "> This English plan is what the teachers and the judge receive.", "",
            "| Item | Content |", "|---|---|",
            f"| Goal | {plan.get('goal', '')} |",
            f"| Goal type | {'Knowledge injection' if plan.get('goal_type') == 'knowledge' else 'Format/task adaptation'} |",
            f"| Domain/scope | {plan.get('domain', '')} |",
            f"| Data language | {plan.get('language', '')} |",
            f"| Answer style | {plan.get('answer_style', '')} |",
            f"| Target items | {plan.get('target_samples', '')} |",
            f"| Student model comment | {plan.get('student_model_comment', '')} |", "",
            "## Evaluation criteria", crit or "- (none)", ""]
    body = plan.get("plan_markdown", "").strip()
    out = "\n".join(head) + ("\n" + body + "\n" if body else "")
    summary = (plan.get("user_summary_markdown") or "").strip()
    if summary:
        out += "\n---\n\n## " + T("요약 (사용자 언어)") + "\n\n" + summary + "\n"
    return out


def _student_overrides(cfg: AppConfig) -> dict:
    s = cfg.student
    return {"mode": s.mode, "rank": s.rank, "learning_rate": s.learning_rate, "epochs": s.epochs,
            "batch_size": s.batch_size, "max_seq_len": s.max_seq_len, "lr_sweep": s.lr_sweep}


async def finalize(cfg: AppConfig, project: Project, history: list[dict[str, str]], activity=None) -> dict:
    async with make_provider(provider_config_from_role(cfg.roles["orchestrator"]), activity=activity) as p:
        plan = await Orchestrator(p).finalize_plan(history, cfg.student.model_id)
    project.path(F_PLAN).write_text(plan_markdown(plan), encoding="utf-8")
    project.update_state(plan=plan, status="planned", student_model=cfg.student.model_id)

    # 03_algorithm.md 초안 (모델 config만 조회, 가중치는 받지 않음)
    model = None
    try:
        mcfg = await asyncio.to_thread(hardware.load_model_config, cfg.student.model_id, cfg.student.hf_token)
        model = hardware.model_info_from_config(cfg.student.model_id, mcfg)
    except Exception:
        model = None
    engine = cfg.student.engine if cfg.student.engine != "auto" else "mlx"
    rec = algorithm.recommend(plan["goal_type"], plan["target_samples"], 350, model, None, engine,
                              _student_overrides(cfg))
    if model is None:
        rec.rationale.append(T('학생 모델 정보를 가져오지 못해 기본값으로 작성(확정본에서 갱신)'))
    draft = algorithm.render_markdown(T('초안 — 방안 확정 시점'), plan, rec)
    project.path(F_ALGO_DRAFT).write_text(draft, encoding="utf-8")
    project.path(F_ALGO).write_text(algorithm.algorithm_document(plan, draft), encoding="utf-8")
    return plan
