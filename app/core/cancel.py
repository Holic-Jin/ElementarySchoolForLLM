"""취소/일시정지 토큰. 진행 중인 HTTP 요청, 서브프로세스, 다운로드를 한 번에 중단한다."""
from __future__ import annotations

from ..i18n import T

import asyncio
import os
import signal
import threading


class Cancelled(Exception):
    """사용자 취소 또는 사양 부족 등으로 전체 작업이 취소됨."""


class CancelToken:
    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._pids: set[int] = set()
        self._lock = threading.Lock()
        self.reason = ""

    # --- 취소
    def cancel(self, reason: str = T('사용자 취소')) -> None:
        self.reason = reason
        self._cancel.set()
        self._resume.set()
        with self._lock:
            pids = list(self._pids)
        for pid in pids:
            kill_process_group(pid)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled(self.reason or T('취소됨'))

    # --- 일시정지
    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    async def checkpoint(self) -> None:
        """비동기 작업 사이사이에 호출: 취소 확인 + 일시정지 대기."""
        self.check()
        while not self._resume.is_set():
            await asyncio.sleep(0.25)
        self.check()

    # --- 서브프로세스 등록
    def register_pid(self, pid: int) -> None:
        with self._lock:
            self._pids.add(pid)
        if self.cancelled:
            kill_process_group(pid)

    def unregister_pid(self, pid: int) -> None:
        with self._lock:
            self._pids.discard(pid)


def kill_process_group(pid: int) -> None:
    """start_new_session=True 로 띄운 프로세스와 그 자식들을 종료."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                return
        if sig == signal.SIGTERM:
            # 잠시 기다린 뒤 아직 살아 있으면 SIGKILL
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                except OSError:
                    return
                threading.Event().wait(0.1)
