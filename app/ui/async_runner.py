"""Qt ↔ asyncio 연결: 백그라운드 스레드에서 코루틴을 실행하고 결과를 메인 스레드 시그널로 전달."""
from __future__ import annotations

import asyncio
import threading
import traceback
from typing import Any, Awaitable, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

_active: set["_Task"] = set()


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, factory: Callable[[], Awaitable[Any]]):
        super().__init__()
        self.setAutoDelete(False)
        self.factory = factory
        self.signals = _Signals()

    @Slot()
    def run(self) -> None:
        try:
            res = asyncio.run(self.factory())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._emit(self.signals.failed, str(e) or e.__class__.__name__)
        else:
            self._emit(self.signals.done, res)

    @staticmethod
    def _emit(sig, value) -> None:
        try:
            sig.emit(value)
        except RuntimeError:  # 앱 종료 중 수신 객체가 이미 삭제된 경우
            pass


def run_async(factory: Callable[[], Awaitable[Any]], on_done: Callable[[Any], None] | None = None,
              on_fail: Callable[[str], None] | None = None) -> None:
    t = _Task(factory)
    if on_done:
        t.signals.done.connect(on_done)
    if on_fail:
        t.signals.failed.connect(on_fail)
    t.signals.done.connect(lambda *_: _active.discard(t))
    t.signals.failed.connect(lambda *_: _active.discard(t))
    _active.add(t)
    QThreadPool.globalInstance().start(t)


class QtReporter(QObject):
    """core.events.Reporter 구현: 어느 스레드에서 호출해도 메인 스레드로 시그널 전달."""

    log_signal = Signal(str, str)
    stage_signal = Signal(str, int, int, str)

    def log(self, level: str, msg: str) -> None:
        self.log_signal.emit(level, msg)

    def stage(self, stage_id: str, done: int, total: int, info: str = "") -> None:
        self.stage_signal.emit(stage_id, int(done), int(total), info)


class AskBridge(QObject):
    """파이프라인 스레드에서 사용자에게 예/아니오를 묻는다."""

    ask = Signal(str, str, object)

    async def __call__(self, title: str, message: str) -> bool:
        holder = {"event": threading.Event(), "answer": False}
        self.ask.emit(title, message, holder)
        await asyncio.to_thread(holder["event"].wait)
        return bool(holder["answer"])
