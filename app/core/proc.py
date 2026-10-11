"""서브프로세스 실행 도우미: 줄 단위 로그 스트리밍 + 취소 토큰 연동 + 잠자기 방지."""
from __future__ import annotations

from ..i18n import T

import asyncio
import os
import subprocess
from typing import Callable

from .cancel import Cancelled, CancelToken, kill_process_group
from .events import Reporter


async def run_streaming(cmd: list[str], reporter: Reporter | None, cancel: CancelToken | None,
                        on_line: Callable[[str], None] | None = None, cwd: str | None = None,
                        env: dict[str, str] | None = None, log_prefix: str = "",
                        on_start: Callable[[int], None] | None = None, activity=None) -> int:
    """명령을 실행하고 stdout/stderr를 한 줄씩 로그로 보낸다. 취소되면 프로세스 그룹을 종료하고 Cancelled."""
    full_env = os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"
    full_env.setdefault("NO_COLOR", "1")  # 색상 코드 없이 출력(로그 가독성)
    if env:
        full_env.update(env)
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                cwd=cwd, env=full_env, start_new_session=True)
    if cancel:
        cancel.register_pid(proc.pid)
    if on_start:
        on_start(proc.pid)
    watcher = None
    if activity is not None:
        from .activity import CpuWatcher

        activity.begin()
        watcher = CpuWatcher(proc.pid, activity).start()
    code = -1
    try:
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # tqdm 진행 표시(\r)도 줄로 취급
            parts = buf.replace(b"\r", b"\n").split(b"\n")
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                if reporter:
                    reporter.log("DEBUG", f"{log_prefix}{line}")
                if activity is not None:
                    activity.beat(detail=line[-60:])
                if on_line:
                    on_line(line)
            if cancel and cancel.cancelled:
                kill_process_group(proc.pid)
        if buf.strip():
            line = buf.decode("utf-8", "replace").rstrip()
            if reporter:
                reporter.log("DEBUG", f"{log_prefix}{line}")
            if on_line:
                on_line(line)
        code = await proc.wait()
    except asyncio.CancelledError:
        kill_process_group(proc.pid)
        raise
    finally:
        if cancel:
            cancel.unregister_pid(proc.pid)
        if watcher:
            watcher.stop()
        if activity is not None:
            activity.end(code == 0, "" if code == 0 else f"exit {code}")
    if cancel and cancel.cancelled:
        raise Cancelled(cancel.reason or T('취소됨'))
    return code


async def run_capture(cmd: list[str], timeout: float = 60.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        kill_process_group(proc.pid)
        return -1, "timeout"
    return proc.returncode or 0, out.decode("utf-8", "replace")


def run_capture_sync(cmd: list[str], timeout: float = 60.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (subprocess.TimeoutExpired, OSError) as e:
        return -1, str(e)


class KeepAwake:
    """작업 중 Mac 잠자기 방지 (caffeinate -i -w <pid>)."""

    def __init__(self) -> None:
        self._p: subprocess.Popen | None = None

    def __enter__(self):
        try:
            self._p = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self._p = None
        return self

    def __exit__(self, *exc):
        if self._p and self._p.poll() is None:
            self._p.terminate()
