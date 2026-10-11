"""LLM 응답에서 JSON 추출 + JSONL 입출력."""
from __future__ import annotations

from ..i18n import T

import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """추론 모델이 출력하는 <think>…</think> 블록 제거."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # 여는 태그 없이 닫는 태그만 있는 경우
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """응답 문자열에서 첫 번째 JSON 객체/배열을 찾아 파싱. 실패 시 ValueError."""
    text = strip_thinking(text)
    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                end = _matching_bracket(cand, start, opener, closer)
                if end == -1:
                    break
                try:
                    return json.loads(cand[start : end + 1])
                except json.JSONDecodeError:
                    start = cand.find(opener, start + 1)
    raise ValueError(T('응답에서 JSON을 찾을 수 없습니다: ') + text[:200])


def _matching_bracket(s: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


_jsonl_lock = threading.Lock()


def append_jsonl(path: Path, obj: dict) -> None:
    with _jsonl_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with _jsonl_lock, open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 중단으로 잘린 마지막 줄 등은 무시
    return rows


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
