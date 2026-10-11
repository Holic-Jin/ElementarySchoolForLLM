"""작업 신호등: 대기 중인 작업이 실제로 진행 중인지(주황), 멈췄거나 헛도는지(빨강), 끝났는지(초록)를 판단한다.

- 신호(beat): LLM 토큰·추론 토큰, 서브프로세스 출력 줄, CPU 사용, 다운로드 바이트 증가 등
- 일정 시간 신호가 없으면 '멈춤', 같은 문장을 반복 출력하면 '헛돎'으로 본다.
- core는 타임스탬프만 갱신하고, UI가 주기적으로 evaluate()를 호출해 색을 정한다(스레드 안전).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..i18n import T

GRAY, ORANGE, RED, GREEN = "gray", "orange", "red", "green"

FIRST_SIGNAL_GRACE = 90.0  # 첫 신호(첫 토큰·모델 로딩)를 기다리는 시간
STALL_AFTER = 30.0  # 신호가 시작된 뒤 이 시간 동안 새 신호가 없으면 멈춤


@dataclass
class LightState:
    color: str
    text: str


class Activity:
    def __init__(self, key: str, label: str, first_grace: float = FIRST_SIGNAL_GRACE, stall_after: float = STALL_AFTER):
        self.key = key
        self.label = label
        self.first_grace = first_grace
        self.stall_after = stall_after
        self._lock = threading.Lock()
        self.inflight = 0
        self.started = 0.0
        self.last_beat = 0.0
        self.got_signal = False
        self.tokens = 0
        self.reasoning = 0
        self.detail = ""
        self.looping = False
        self.error = ""
        self.finished_at = 0.0
        self.completed = 0
        self.duration = 0.0

    # ------------------------------------------------------------------ core 쪽 API
    def begin(self, detail: str = "") -> None:
        with self._lock:
            now = time.monotonic()
            if self.inflight == 0:
                self.started = now
                self.last_beat = now
                self.got_signal = False
                self.tokens = self.reasoning = 0
                self.looping = False
                self.error = ""
            self.inflight += 1
            if detail:
                self.detail = detail

    def beat(self, tokens: int = 0, reasoning: int = 0, detail: str | None = None) -> None:
        with self._lock:
            self.last_beat = time.monotonic()
            self.got_signal = True
            self.tokens += tokens
            self.reasoning += reasoning
            if detail is not None:
                self.detail = detail

    def flag_loop(self) -> None:
        with self._lock:
            self.looping = True

    def end(self, ok: bool = True, error: str = "") -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            if not ok:
                self.error = error or T("오류")
            if self.inflight == 0:
                self.finished_at = time.monotonic()
                self.duration = self.finished_at - self.started
                if ok:
                    self.completed += 1
                    self.looping = False

    def reset(self) -> None:
        with self._lock:
            self.inflight = 0
            self.completed = 0
            self.error = ""
            self.looping = False
            self.detail = ""
            self.finished_at = 0.0

    # ------------------------------------------------------------------ UI 쪽 API
    def evaluate(self, now: float | None = None) -> LightState:
        with self._lock:
            now = time.monotonic() if now is None else now
            if self.inflight == 0:
                if self.error:
                    return LightState(RED, T("오류: {e}", e=self.error[:80]))
                if self.completed > 1:
                    return LightState(GREEN, T("완료 · {n}회 처리 (마지막 {sec:.1f}초)", n=self.completed, sec=self.duration))
                if self.completed:
                    return LightState(GREEN, T("완료 ({sec:.1f}초)", sec=self.duration))
                return LightState(GRAY, T("대기"))
            elapsed = now - self.started
            quiet = now - self.last_beat
            if self.looping:
                return LightState(RED, T("반복 출력 감지 — 모델이 헛돌고 있습니다"))
            if not self.got_signal:
                if quiet > self.first_grace:
                    return LightState(RED, T("응답 없음 {sec:.0f}초 — 멈췄거나 서버가 매우 느립니다", sec=quiet))
                return LightState(ORANGE, T("시작 대기 중 {sec:.0f}초 (모델 로딩·프롬프트 처리)", sec=elapsed))
            if quiet > self.stall_after:
                return LightState(RED, T("{sec:.0f}초째 신호 없음 — 멈춘 것 같습니다", sec=quiet))
            parts = []
            if self.reasoning and not self.tokens:
                parts.append(T("생각(추론) 중 · 추론 토큰 {n}", n=self.reasoning))
            elif self.tokens:
                rate = self.tokens / max(0.1, elapsed)
                parts.append(T("응답 생성 중 · 토큰 {n} ({rate:.0f} tok/s)", n=self.tokens, rate=rate))
            if self.detail:
                parts.append(self.detail)
            if not parts:
                parts.append(T("진행 중"))
            if self.inflight > 1:
                parts.append(T("동시 작업 {n}개", n=self.inflight))
            return LightState(ORANGE, " · ".join(parts) + T(" · {sec:.0f}초", sec=elapsed))


def is_looping(text: str, repeats: int = 6, min_period: int = 6, max_period: int = 200) -> bool:
    """출력 끝부분에 같은 조각이 repeats번 이상 연속 반복되면 헛도는 것으로 본다."""
    tail = text[-(max_period * repeats):]
    for p in range(min_period, max_period + 1):
        if len(tail) < p * repeats:
            break
        chunk = tail[-p:]
        if not chunk.strip():
            continue
        if tail[-p * repeats:] == chunk * repeats:
            return True
    return False


class ActivityHub:
    """이름으로 Activity를 찾거나 만든다. UI는 같은 허브를 공유해 신호등을 그린다."""

    def __init__(self) -> None:
        self._acts: dict[str, Activity] = {}
        self._lock = threading.Lock()

    def get(self, key: str, label: str = "", **kw) -> Activity:
        with self._lock:
            if key not in self._acts:
                self._acts[key] = Activity(key, label or key, **kw)
            elif label:
                self._acts[key].label = label
            return self._acts[key]


HUB = ActivityHub()


class CpuWatcher:
    """프로세스(와 자식)의 CPU 사용률을 주기적으로 재서, 일하는 중이면 신호를 보낸다."""

    def __init__(self, pid: int, activity: Activity, threshold: float = 3.0, interval: float = 2.0):
        self.pid, self.activity, self.threshold, self.interval = pid, activity, threshold, interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "CpuWatcher":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            import psutil

            root = psutil.Process(self.pid)
            procs: dict[int, psutil.Process] = {}
            while not self._stop.is_set() and root.is_running():
                total = 0.0
                for p in [root] + root.children(recursive=True):
                    proc = procs.setdefault(p.pid, p)
                    try:
                        total += proc.cpu_percent(None)
                    except psutil.Error:
                        pass
                if total >= self.threshold:
                    self.activity.beat(detail=T("CPU {cpu:.0f}%", cpu=total))
                self._stop.wait(self.interval)
        except Exception:
            return
