"""역할별 시스템 프롬프트 & 호출 래퍼 (오케스트레이터, 교사1, 교사2)."""
from __future__ import annotations

from ..i18n import T

import json
import re
from typing import Any

from . import paths
from .providers import LLMError, Provider

PROMPT_FILES = {
    "orchestrator_interview": T('오케스트레이터 — 학습방안 질문'),
    "orchestrator_finalize": T('오케스트레이터 — 학습방안 확정'),
    "orchestrator_spec": T('오케스트레이터 — 학습데이터 명세'),
    "orchestrator_judge": T('오케스트레이터 — 답변1·2 비교 평가'),
    "orchestrator_compare": T('오케스트레이터 — 학습 전후 비교'),
    "teacher1_generate": T('교사1 — 질문+답변1 생성'),
    "teacher1_answer": T('교사1 — 답변 재생성'),
    "teacher2_answer": T('교사2 — 독립 답변 생성'),
    "orchestrator_translate": T("오케스트레이터 — 학습방안 영어 변환"),
}

# 교사·심사에 전달되는 학습방안 필드 (반드시 영어)
PLAN_TEXT_FIELDS = ("title", "goal", "domain", "answer_style", "student_model_comment", "evaluation_criteria",
                    "plan_markdown")


def non_english_ratio(text: str) -> float:
    """문자 중 라틴 문자가 아닌 글자(한글·가나·한자 등)의 비율."""
    import unicodedata

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    other = sum(1 for c in letters if ord(c) > 0x24F and not unicodedata.name(c, "").startswith("LATIN"))
    return other / len(letters)


def plan_is_english(plan: dict, threshold: float = 0.1) -> bool:
    parts = []
    for k in PLAN_TEXT_FIELDS:
        v = plan.get(k)
        parts += v if isinstance(v, list) else [v]
    return non_english_ratio(" ".join(str(x) for x in parts if x)) <= threshold

_SUMMARY_RE = re.compile(r"<plan_summary>(.*?)(?:</plan_summary>|$)", re.DOTALL)


def load_prompt(name: str) -> str:
    """사용자 수정본(~/.ElementarySchoolForLLM/prompts)이 있으면 우선 사용."""
    user = paths.USER_PROMPTS_DIR / f"{name}.md"
    if user.exists():
        return user.read_text(encoding="utf-8")
    return (paths.DEFAULT_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(template: str, **kw: Any) -> str:
    for k, v in kw.items():
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        template = template.replace("{{" + k + "}}", v)
    return template


def split_summary(text: str) -> tuple[str, str]:
    """오케스트레이터 응답에서 <plan_summary> 블록을 분리 → (본문, 요약)."""
    m = _SUMMARY_RE.search(text)
    if not m:
        return text.strip(), ""
    body = (text[: m.start()] + text[m.end():]).strip()
    return body, m.group(1).strip()


def _difficulty_plan(n: int, dist: dict[str, float]) -> dict[str, int]:
    keys = ["easy", "medium", "hard"]
    total = sum(max(0.0, float(dist.get(k, 0))) for k in keys) or 1.0
    raw = {k: n * max(0.0, float(dist.get(k, 0))) / total for k in keys}
    counts = {k: int(raw[k]) for k in keys}
    rest = n - sum(counts.values())
    for k in sorted(keys, key=lambda k: raw[k] - counts[k], reverse=True)[:rest]:
        counts[k] += 1
    return counts


class Orchestrator:
    def __init__(self, provider: Provider):
        self.p = provider

    async def interview(self, history: list[dict[str, str]], student_model: str) -> str:
        system = render(load_prompt("orchestrator_interview"), student_model=student_model)
        return await self.p.chat([{"role": "system", "content": system}] + history)

    async def finalize_plan(self, history: list[dict[str, str]], student_model: str) -> dict:
        system = render(load_prompt("orchestrator_interview"), student_model=student_model)
        msgs = [{"role": "system", "content": system}] + history + [
            {"role": "user", "content": load_prompt("orchestrator_finalize")}]
        plan = await self.p.chat_json(msgs, temperature=0.3, max_tokens=4096)
        if not isinstance(plan, dict):
            raise LLMError(T('학습방안 형식이 올바르지 않습니다.'))
        for k in [k for k, v in plan.items() if _is_placeholder(v, PLAN_PLACEHOLDERS)]:
            plan[k] = ""
        plan["evaluation_criteria"] = [c for c in plan.get("evaluation_criteria", []) or []
                                       if not _is_placeholder(c, PLAN_PLACEHOLDERS)]
        if not plan.get("title"):
            plan["title"] = T('학습 프로젝트')
        plan["goal_type"] = "knowledge" if str(plan.get("goal_type", "")).lower().startswith("know") else "adaptation"
        try:
            plan["target_samples"] = max(10, int(plan.get("target_samples", 200)))
        except (TypeError, ValueError):
            plan["target_samples"] = 200
        plan.setdefault("evaluation_criteria", [])
        plan.setdefault("plan_markdown", "")
        plan.setdefault("user_summary_markdown", "")
        return await self.ensure_english(plan)

    async def ensure_english(self, plan: dict) -> dict:
        """교사에게 전달할 학습방안이 영어가 아니면 영어로 번역한다(최대 2회 시도)."""
        for _ in range(2):
            if plan_is_english(plan):
                break
            payload = {k: plan.get(k) for k in PLAN_TEXT_FIELDS}
            prompt = render(load_prompt("orchestrator_translate"), plan=json.dumps(payload, ensure_ascii=False, indent=2))
            try:
                tr = await self.p.chat_json([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=4096)
            except LLMError:
                continue
            if isinstance(tr, dict):
                for k in PLAN_TEXT_FIELDS:
                    if tr.get(k):
                        plan[k] = tr[k]
        if not plan_is_english(plan):
            raise LLMError(T("학습방안을 영어로 만들지 못했습니다. 오케스트레이터 모델을 바꾸거나 다시 시도하세요."))
        plan["english"] = True
        return plan

    async def make_data_spec(self, plan: dict) -> dict:
        prompt = render(load_prompt("orchestrator_spec"),
                        plan=plan.get("plan_markdown") or json.dumps(plan, ensure_ascii=False),
                        target_samples=plan.get("target_samples", 200))
        last_err = ""
        for _ in range(3):
            msgs = [{"role": "user", "content": prompt + (f"\n\nYour previous output was invalid: {last_err}" if last_err else "")}]
            spec = await self.p.chat_json(msgs, temperature=0.3, max_tokens=4096)
            try:
                return validate_spec(spec, plan)
            except ValueError as e:
                last_err = str(e)
        try:  # 마지막 시도: 예시 문구는 비우고 구조만 맞으면 사용
            return validate_spec(spec, plan, strict=False)
        except ValueError:
            raise LLMError(T('데이터 명세 검증 실패: {last_err}', last_err=last_err))

    async def judge_pair(self, plan: dict, spec: dict, item: dict, answer_a: str, answer_b: str) -> dict:
        prompt = render(load_prompt("orchestrator_judge"), goal=plan.get("goal", ""), category=item.get("category", ""),
                        difficulty=item.get("difficulty", ""), answer_format=spec.get("answer_format", ""),
                        criteria="; ".join(plan.get("evaluation_criteria", [])) or "correct and complete answers",
                        question=item["question"], answer_a=answer_a, answer_b=answer_b)
        msgs = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            res = await self.p.chat_json(msgs, temperature=0.0, max_tokens=1024)
            if not isinstance(res, dict) or "A" not in res or "B" not in res:
                raise LLMError(T('평가 결과 형식이 올바르지 않습니다.'))
            if not _all_zero(res):
                return res
            # 예시의 자리표시자를 베껴 모든 점수를 0으로 낸 경우 한 번 더 요청
            msgs = msgs + [{"role": "assistant", "content": json.dumps(res, ensure_ascii=False)},
                           {"role": "user", "content": "All scores are 0. Re-evaluate with real scores from 1 to 10 and output only the JSON."}]
        return res

    async def compare_before_after(self, question: str, reference: str, ans_x: str, ans_y: str) -> dict:
        prompt = render(load_prompt("orchestrator_compare"), question=question, reference=reference,
                        answer_x=ans_x, answer_y=ans_y)
        return await self.p.chat_json([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)


class Teacher1:
    def __init__(self, provider: Provider):
        self.p = provider

    async def generate(self, plan: dict, spec: dict, category: dict, n: int, avoid: list[str]) -> list[dict]:
        plan_counts = _difficulty_plan(n, category.get("difficulty", {}))
        diff_text = ", ".join(f"{v} {k}" for k, v in plan_counts.items() if v)
        prompt = render(load_prompt("teacher1_generate"), goal=plan.get("goal", ""), language=spec.get("language", ""),
                        category=category["name"], category_description=category.get("description", ""),
                        answer_format=spec.get("answer_format", ""), example_question=category.get("example_question", ""),
                        n=n, difficulty_plan=diff_text,
                        avoid=" / ".join(avoid[-30:]) if avoid else "(none)")
        res = await self.p.chat_json([{"role": "user", "content": prompt}], max_tokens=max(2048, n * 700))
        items = res.get("items", res) if isinstance(res, dict) else res
        out = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and str(it.get("question", "")).strip() and str(it.get("answer", "")).strip():
                d = str(it.get("difficulty", "medium")).lower()
                out.append({"question": str(it["question"]).strip(), "answer": str(it["answer"]).strip(),
                            "difficulty": d if d in ("easy", "medium", "hard") else "medium"})
        return out

    async def answer(self, plan: dict, spec: dict, question: str) -> str:
        prompt = render(load_prompt("teacher1_answer"), goal=plan.get("goal", ""), language=spec.get("language", ""),
                        answer_format=spec.get("answer_format", ""), question=question)
        return (await self.p.chat([{"role": "user", "content": prompt}])).strip()


class Teacher2:
    def __init__(self, provider: Provider):
        self.p = provider

    async def answer(self, plan: dict, spec: dict, question: str) -> str:
        """교사2에게는 질문만 전달한다(교사1의 답변1은 절대 포함하지 않음)."""
        prompt = render(load_prompt("teacher2_answer"), goal=plan.get("goal", ""), language=spec.get("language", ""),
                        answer_format=spec.get("answer_format", ""), question=question)
        return (await self.p.chat([{"role": "user", "content": prompt}])).strip()


# 프롬프트의 <...> 자리표시자를 값으로 그대로 베낀 경우를 잡아낸다 (소형 모델에서 자주 발생)
def _is_placeholder(v: Any, _table: object = None) -> bool:
    return isinstance(v, str) and v.strip().startswith("<") and v.strip().endswith(">")


SPEC_PLACEHOLDERS = PLAN_PLACEHOLDERS = None


def _all_zero(res: dict) -> bool:
    nums = []
    for key in ("A", "B", "question"):
        d = res.get(key)
        if isinstance(d, dict):
            nums += [v for v in d.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return bool(nums) and all(v == 0 for v in nums)


def validate_spec(spec: Any, plan: dict, strict: bool = True) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("The specification must be a JSON object.")
    if strict:
        copied = [k for k in ("system_prompt", "answer_format") if _is_placeholder(spec.get(k), SPEC_PLACEHOLDERS)]
        copied += [f"categories[{i}]" for i, c in enumerate(spec.get("categories") or [])
                   if isinstance(c, dict) and any(_is_placeholder(c.get(k), SPEC_PLACEHOLDERS)
                                                  for k in ("name", "description", "example_question"))]
        if copied:
            raise ValueError(f"You copied the <...> placeholders as values ({', '.join(copied)}). Fill them with real content for this plan.")
    cats = spec.get("categories")
    if not isinstance(cats, list) or not cats:
        raise ValueError("categories is empty.")
    clean = []
    seen = set()
    for c in cats:
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            continue
        name = str(c["name"]).strip()
        if name in seen:
            continue
        seen.add(name)
        try:
            count = max(1, int(c.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        diff = c.get("difficulty") if isinstance(c.get("difficulty"), dict) else {}
        clean.append({"name": name, "description": str(c.get("description", "")), "count": count,
                      "difficulty": {k: float(diff.get(k, 0) or 0) for k in ("easy", "medium", "hard")} if diff
                      else {"easy": 0.3, "medium": 0.5, "hard": 0.2},
                      "example_question": str(c.get("example_question", ""))})
    if not clean:
        raise ValueError("No valid categories.")
    # 목표 문항 수에 맞게 비례 조정
    target = int(plan.get("target_samples", sum(c["count"] for c in clean)))
    total = sum(c["count"] for c in clean)
    if total != target and total > 0 and target >= len(clean):
        # 최대 나머지 방식으로 비례 배분해 합계를 목표와 정확히 맞춘다
        raw = [c["count"] * target / total for c in clean]
        counts = [max(1, int(r)) for r in raw]
        order = sorted(range(len(clean)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        k = 0
        while sum(counts) < target:
            counts[order[k % len(order)]] += 1
            k += 1
        while sum(counts) > target:
            i = max(range(len(counts)), key=lambda j: counts[j])
            counts[i] -= 1
        for c, n in zip(clean, counts):
            c["count"] = n
    try:
        avg = int(spec.get("avg_answer_tokens", 250))
    except (TypeError, ValueError):
        avg = 250
    sp = spec.get("system_prompt", "")
    af = spec.get("answer_format", "")
    return {"language": str(spec.get("language", plan.get("language", ""))),
            "system_prompt": "" if _is_placeholder(sp, SPEC_PLACEHOLDERS) else str(sp),
            "answer_format": str(plan.get("answer_style", "")) if (not af or _is_placeholder(af, SPEC_PLACEHOLDERS)) else str(af),
            "avg_answer_tokens": max(16, avg),
            "categories": clean}
