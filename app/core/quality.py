"""품질 평가: 중복 제거, 길이 필터, 답변1·2 블라인드 비교 판정 규칙."""
from __future__ import annotations

from ..i18n import T

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, field

from .roles import Orchestrator

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    t = normalize(text).replace(" ", "")
    if len(t) <= n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Deduper:
    """정규화 해시(완전 중복) + 문자 3-gram Jaccard 유사도(유사 중복)."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._hashes: set[str] = set()
        self._grams: list[set[str]] = []

    def add(self, text: str) -> bool:
        """새 문장이면 등록하고 True, 중복이면 False."""
        h = text_hash(text)
        if h in self._hashes:
            return False
        g = char_ngrams(text)
        for other in self._grams:
            if jaccard(g, other) >= self.threshold:
                return False
        self._hashes.add(h)
        self._grams.append(g)
        return True


def approx_tokens(text: str) -> int:
    """토크나이저 없이 쓰는 근사치: ASCII 4글자≈1토큰, 한글 등 비ASCII 1글자≈0.8토큰."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return int(ascii_chars / 4 + (len(text) - ascii_chars) * 0.8) + 1


def length_ok(question: str, answer: str, max_tokens: int = 4096) -> bool:
    if len(question.strip()) < 4 or len(answer.strip()) < 2:
        return False
    return approx_tokens(question) + approx_tokens(answer) <= max_tokens


@dataclass
class Verdict:
    status: str  # accepted | regenerate | rejected | rejected_question | uncertain
    chosen: str | None = None  # "1" | "2"
    score_1: float = 0.0
    score_2: float = 0.0
    question_score: float = 0.0
    contradiction: bool = False
    clear_winner: bool = False  # 선호도 쌍으로 쓸 만큼 승패가 분명한가
    reason: str = ""
    raw: list[dict] = field(default_factory=list)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0 <= f <= 10 else None


def _score(d: dict) -> float:
    """종합 점수. 모델이 overall을 빠뜨리거나 범위를 벗어나면 세부 항목 평균으로 대신한다."""
    if not isinstance(d, dict):
        return 0.0
    o = _num(d.get("overall"))
    if o is not None and o > 0:
        return o
    parts = [x for x in (_num(d.get(k)) for k in ("accuracy", "completeness", "format", "safety", "clarity", "relevance"))
             if x is not None]
    return sum(parts) / len(parts) if parts else 0.0


def decide(q_score: float, difficulty_ok: bool, s1: float, s2: float, contradiction: bool,
           threshold: float, can_regenerate: bool, swap_flipped: bool = False) -> Verdict:
    """판정 규칙
    ① 한쪽만 기준 통과 → 그 답변 채택
    ② 둘 다 통과 → 높은 점수 채택 (순서 교체 평가에서 승자가 뒤집혔으면 승패 불분명 → 선호도 쌍 미저장)
    ③ 둘 다 통과했지만 서로 모순 → 정답 불확실: 재생성 가능하면 재생성, 아니면 제외
    ④ 둘 다 미달 → 재생성(최대 N회), 그래도 미달이면 탈락
    ⑤ 질문 자체 불량(점수 미달 또는 난이도 부적합이 심함) → 질문째 폐기
    """
    v = Verdict(status="rejected", score_1=s1, score_2=s2, question_score=q_score, contradiction=contradiction)
    q_threshold = max(1.0, threshold - 2.0)
    if q_score < q_threshold:
        v.status, v.reason = "rejected_question", T('질문 점수 미달({q_score:.1f})', q_score=q_score)
        return v
    pass1, pass2 = s1 >= threshold, s2 >= threshold
    if pass1 and pass2:
        if contradiction:
            v.status = "regenerate" if can_regenerate else "uncertain"
            v.reason = T('두 답변이 서로 모순 — 정답 불확실')
            return v
        v.status = "accepted"
        v.chosen = "1" if s1 >= s2 else "2"
        v.clear_winner = (abs(s1 - s2) >= 1.0) and not swap_flipped
        v.reason = T('두 답변 모두 통과, 높은 점수 채택') + (T(' (승패 불분명)') if swap_flipped else "")
    elif pass1 or pass2:
        v.status = "accepted"
        v.chosen = "1" if pass1 else "2"
        v.clear_winner = True
        v.reason = T('한쪽 답변만 통과')
    else:
        v.status = "regenerate" if can_regenerate else "rejected"
        v.reason = T('두 답변 모두 기준 미달({s1:.1f}/{s2:.1f})', s1=s1, s2=s2)
    if v.status == "accepted" and not difficulty_ok:
        v.reason += T(' · 난이도 부적합 표시')
    return v


async def judge_pair(orch: Orchestrator, plan: dict, spec: dict, item: dict, answer_1: str, answer_2: str,
                     threshold: float, can_regenerate: bool, rng: random.Random, swap_margin: float = 1.0) -> Verdict:
    """블라인드 A/B 비교. 순서를 무작위로 섞고, 점수 차가 작으면 순서를 바꿔 재평가한다."""
    first_is_1 = rng.random() < 0.5
    a, b = (answer_1, answer_2) if first_is_1 else (answer_2, answer_1)
    r1 = await orch.judge_pair(plan, spec, item, a, b)
    sa, sb = _score(r1.get("A", {})), _score(r1.get("B", {}))
    s1, s2 = (sa, sb) if first_is_1 else (sb, sa)
    raws = [r1]
    q = r1.get("question", {}) or {}
    q_score = _score(q)
    diff_ok = bool(q.get("difficulty_ok", True))
    contradiction = bool(r1.get("contradiction", False))
    flipped = False
    if abs(s1 - s2) < swap_margin:
        r2 = await orch.judge_pair(plan, spec, item, b, a)  # 순서 교체
        raws.append(r2)
        tb, ta = _score(r2.get("A", {})), _score(r2.get("B", {}))  # r2의 A는 원래 b
        t1, t2 = (ta, tb) if first_is_1 else (tb, ta)
        if (s1 - s2) * (t1 - t2) < 0:
            flipped = True
        s1, s2 = (s1 + t1) / 2, (s2 + t2) / 2
        q_score = (q_score + _score(r2.get("question", {}) or {})) / 2
        contradiction = contradiction or bool(r2.get("contradiction", False))
    v = decide(q_score, diff_ok, s1, s2, contradiction, threshold, can_regenerate, flipped)
    v.raw = raws
    return v
