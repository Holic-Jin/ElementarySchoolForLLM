"""로컬 LLM 백엔드 관리: Ollama 실행/모델 받기/내리기, llama.cpp 설치·실행, 학생 모델 서버."""
from __future__ import annotations

from ..i18n import T

import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import zipfile
from pathlib import Path
from typing import Callable

import httpx

from . import paths
from .cancel import CancelToken, kill_process_group
from .events import Reporter

OLLAMA_URL = "http://localhost:11434"
OLLAMA_CANDIDATES = [
    Path.home() / "Applications/Ollama.app/Contents/Resources/ollama",
    Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    Path("/usr/local/bin/ollama"),
    Path("/opt/homebrew/bin/ollama"),
]


def ollama_root(base_url: str) -> str:
    """'http://localhost:11434/v1' → 'http://localhost:11434'"""
    b = base_url.rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


def is_ollama_url(base_url: str) -> bool:
    return ":11434" in base_url


def ollama_binary() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for c in OLLAMA_CANDIDATES:
        if c.exists():
            return str(c)
    return None


async def ollama_alive(root: str = OLLAMA_URL) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{root}/api/version")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


async def ensure_ollama(root: str = OLLAMA_URL, reporter: Reporter | None = None) -> bool:
    """Ollama 서버가 꺼져 있으면 백그라운드로 실행한다."""
    if await ollama_alive(root):
        return True
    binary = ollama_binary()
    if not binary:
        if reporter:
            reporter.log("ERROR", T('Ollama가 설치되어 있지 않습니다. 설정 탭에서 설치하거나 https://ollama.com 에서 설치하세요.'))
        return False
    log = open(paths.LOG_DIR / "ollama.log", "a")
    subprocess.Popen([binary, "serve"], stdout=log, stderr=log, start_new_session=True)
    for _ in range(40):
        await asyncio.sleep(0.5)
        if await ollama_alive(root):
            if reporter:
                reporter.log("INFO", T('Ollama 서버를 시작했습니다.'))
            return True
    return False


async def ollama_models(root: str = OLLAMA_URL) -> dict[str, int]:
    """설치된 모델 이름 → 파일 크기(byte)."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{root}/api/tags")
        r.raise_for_status()
        return {m["name"]: int(m.get("size", 0)) for m in r.json().get("models", [])}


def ollama_has(models: dict[str, int], name: str) -> bool:
    if name in models:
        return True
    return ":" not in name and f"{name}:latest" in models


async def ollama_pull(name: str, on_progress: Callable[[int, int, str], None], root: str = OLLAMA_URL,
                      cancel: CancelToken | None = None, activity=None) -> None:
    if activity is not None:
        activity.begin()
    try:
        await _ollama_pull(name, on_progress, root, cancel, activity)
    except BaseException as e:
        if activity is not None:
            activity.end(False, str(e)[:80])
        raise
    if activity is not None:
        activity.end(True)


async def _ollama_pull(name, on_progress, root, cancel, activity) -> None:
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("POST", f"{root}/api/pull", json={"model": name, "stream": True}) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if cancel and cancel.cancelled:
                    raise asyncio.CancelledError()
                if not line.strip():
                    continue
                d = json.loads(line)
                if activity is not None:
                    activity.beat(detail=d.get("status", ""))
                if d.get("error"):
                    raise RuntimeError(d["error"])
                on_progress(int(d.get("completed", 0)), int(d.get("total", 0)), d.get("status", ""))


async def ollama_unload(name: str, root: str = OLLAMA_URL) -> None:
    """keep_alive=0 요청으로 모델을 메모리에서 내린다(학습 전에 메모리 확보)."""
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            await c.post(f"{root}/api/generate", json={"model": name, "keep_alive": 0})
    except httpx.HTTPError:
        pass


async def ollama_create(name: str, modelfile: Path, reporter: Reporter, cancel: CancelToken) -> None:
    from .proc import run_streaming

    binary = ollama_binary()
    if not binary:
        raise RuntimeError(T('Ollama가 설치되어 있지 않습니다.'))
    await ensure_ollama(reporter=reporter)
    code = await run_streaming([binary, "create", name, "-f", str(modelfile)], reporter, cancel, log_prefix="[ollama] ")
    if code != 0:
        raise RuntimeError(T('Ollama 모델 등록 실패 (이 모델 구조는 Ollama가 직접 가져오지 못할 수 있습니다)'))


async def install_ollama(reporter: Reporter) -> None:
    """부트스트랩에서 실패했을 때 설정 화면에서 다시 설치."""
    from .proc import run_capture

    dest = Path.home() / "Applications"
    dest.mkdir(exist_ok=True)
    tmp = paths.TOOLS_DIR / "Ollama-darwin.zip"
    reporter.log("INFO", T('Ollama 다운로드 중…'))
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as c:
        async with c.stream("GET", "https://ollama.com/download/Ollama-darwin.zip") as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)
    unzip_dir = paths.TOOLS_DIR / "ollama_unzip"
    shutil.rmtree(unzip_dir, ignore_errors=True)
    code, out = await run_capture(["/usr/bin/ditto", "-x", "-k", str(tmp), str(unzip_dir)])
    if code != 0:
        raise RuntimeError(T('압축 해제 실패: ') + out)
    code, out = await run_capture(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(unzip_dir / "Ollama.app")])
    if code != 0:
        raise RuntimeError(T('Ollama 코드 서명 검증 실패 — 설치하지 않습니다.'))
    shutil.rmtree(dest / "Ollama.app", ignore_errors=True)
    shutil.move(str(unzip_dir / "Ollama.app"), str(dest / "Ollama.app"))
    tmp.unlink(missing_ok=True)
    reporter.log("INFO", T('Ollama 설치 완료 (코드 서명 검증됨)'))


# ----------------------------------------------------------------------------- llama.cpp
LLAMA_DIR = paths.TOOLS_DIR / "llama.cpp"


def llama_server_binary() -> str | None:
    for p in LLAMA_DIR.rglob("llama-server"):
        if os.access(p, os.X_OK):
            return str(p)
    return shutil.which("llama-server")


async def install_llama_cpp(reporter: Reporter) -> str:
    """GitHub 최신 릴리스에서 macOS arm64 빌드를 받아 SHA256(digest가 제공되면) 검증 후 설치."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        r = await c.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                        headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        rel = r.json()
        asset = next((a for a in rel.get("assets", []) if "macos-arm64" in a["name"] and a["name"].endswith(".zip")), None)
        if not asset:
            raise RuntimeError(T('macOS arm64용 llama.cpp 빌드를 찾을 수 없습니다.'))
        reporter.log("INFO", T('llama.cpp {get} 다운로드 중: {name}', get=rel.get('tag_name'), name=asset['name']))
        LLAMA_DIR.mkdir(parents=True, exist_ok=True)
        zpath = LLAMA_DIR / asset["name"]
        h = hashlib.sha256()
        async with c.stream("GET", asset["browser_download_url"]) as resp:
            resp.raise_for_status()
            with open(zpath, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    h.update(chunk)
                    f.write(chunk)
    digest = (asset.get("digest") or "").removeprefix("sha256:")
    if digest and digest != h.hexdigest():
        zpath.unlink(missing_ok=True)
        raise RuntimeError(T('llama.cpp SHA256 검증 실패'))
    reporter.log("INFO", T('SHA256 검증됨') if digest else T('릴리스에 체크섬 정보가 없어 검증을 생략했습니다.'))
    with zipfile.ZipFile(zpath) as z:
        z.extractall(LLAMA_DIR)
    zpath.unlink(missing_ok=True)
    for p in LLAMA_DIR.rglob("*"):
        if p.is_file() and p.name.startswith("llama-"):
            p.chmod(0o755)
    subprocess.run(["/usr/bin/xattr", "-dr", "com.apple.quarantine", str(LLAMA_DIR)], capture_output=True)
    binary = llama_server_binary()
    if not binary:
        raise RuntimeError(T('llama-server 실행 파일을 찾을 수 없습니다.'))
    reporter.log("INFO", T('llama.cpp 설치 완료: {binary}', binary=binary))
    return binary


def start_llama_server(gguf: str, port: int = 8081) -> subprocess.Popen:
    binary = llama_server_binary()
    if not binary:
        raise RuntimeError(T('llama.cpp가 설치되어 있지 않습니다.'))
    log = open(paths.LOG_DIR / "llama-server.log", "a")
    return subprocess.Popen([binary, "-m", gguf, "--host", "127.0.0.1", "--port", str(port)],
                            stdout=log, stderr=log, start_new_session=True)


# ----------------------------------------------------------------------------- 학생 모델 서버
def free_port(preferred: int = 8090) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StudentServer:
    """학습 엔진의 serve_command로 학생 모델을 OpenAI 호환 서버로 띄운다(async with 사용)."""

    def __init__(self, cmd_builder: Callable[[int], list[str]], reporter: Reporter | None = None,
                 cancel: CancelToken | None = None, timeout: float = 600, model_name: str = "default_model",
                 activity=None):
        self.cmd_builder = cmd_builder
        self.model_name = model_name
        self.activity = activity
        self.reporter = reporter
        self.cancel = cancel
        self.timeout = timeout
        self.port = 0
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def start(self) -> str:
        self.port = free_port()
        cmd = self.cmd_builder(self.port)
        log = open(paths.LOG_DIR / "student-server.log", "a")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
        if self.cancel:
            self.cancel.register_pid(self.proc.pid)
        watcher = None
        if self.activity is not None:
            from .activity import CpuWatcher

            self.activity.begin(T("모델 로딩"))
            watcher = CpuWatcher(self.proc.pid, self.activity).start()
        try:
            name = await self._wait_ready()
        except BaseException as e:
            if self.activity is not None:
                self.activity.end(False, str(e)[:80])
            raise
        finally:
            if watcher:
                watcher.stop()
        if self.activity is not None:
            self.activity.end(True)
        return name

    async def _wait_ready(self) -> str:
        waited = 0.0
        async with httpx.AsyncClient(timeout=3) as c:
            while waited < self.timeout:
                if self.cancel:
                    self.cancel.check()
                if self.proc.poll() is not None:
                    raise RuntimeError(T('학생 모델 서버가 시작 직후 종료되었습니다. logs/student-server.log를 확인하세요.'))
                try:
                    r = await c.get(f"{self.base_url}/models")
                    if r.status_code == 200:
                        return self.model_name
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
                waited += 1.0
        raise RuntimeError(T('학생 모델 서버 시작 시간 초과'))

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            if self.cancel:
                self.cancel.unregister_pid(self.proc.pid)
            kill_process_group(self.proc.pid)
        self.proc = None

    async def __aenter__(self) -> "StudentServer":
        self.model_name = await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        self.stop()
