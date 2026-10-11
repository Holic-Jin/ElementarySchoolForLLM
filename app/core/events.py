"""파이프라인 → UI 보고 인터페이스. core는 Qt에 의존하지 않고 이 프로토콜로만 보고한다."""
from __future__ import annotations

from ..i18n import T

import datetime as _dt
import threading
from pathlib import Path
from typing import Protocol

# (id, 표시 이름, 전체 진행률 가중치)
STAGES: list[tuple[str, str, int]] = [
    ("spec", T('데이터 명세'), 4),
    ("hwcheck", T('사양 검사'), 2),
    ("teacher1", T('교사1: 질문+답변1'), 18),
    ("teacher2", T('교사2: 독립 답변2'), 18),
    ("judge", T('오케스트레이터 비교 평가'), 16),
    ("download", T('학생 모델 준비'), 8),
    ("baseline", T('학습 전 기준 응답'), 4),
    ("lr_sweep", T('학습률 탐색'), 6),
    ("train", T('학생 학습'), 18),
    ("eval", T('학습 후 평가'), 6),
]
STAGE_NAMES = {sid: name for sid, name, _ in STAGES}


class Reporter(Protocol):
    def log(self, level: str, msg: str) -> None: ...
    def stage(self, stage_id: str, done: int, total: int, info: str = "") -> None: ...


class FileLogReporter:
    """다른 Reporter를 감싸 로그를 파일에도 기록한다(스레드 안전)."""

    def __init__(self, inner: Reporter | None, path: Path):
        self.inner = inner
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, msg: str) -> None:
        line = f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {msg}\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
        if self.inner:
            self.inner.log(level, msg)

    def stage(self, stage_id: str, done: int, total: int, info: str = "") -> None:
        if self.inner:
            self.inner.stage(stage_id, done, total, info)


class PrintReporter:
    def log(self, level: str, msg: str) -> None:
        print(f"[{level}] {msg}")

    def stage(self, stage_id: str, done: int, total: int, info: str = "") -> None:
        print(f"<{stage_id}> {done}/{total} {info}")


def mask_secret(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, s[:3] + "…" + s[-2:])
    return text
