"""학습 엔진 공통 인터페이스, 진행률 파싱, 학생 모델 다운로드."""
from __future__ import annotations

from ...i18n import T

import asyncio
import importlib.util
import json
import re
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import paths
from ..algorithm import TrainConfig
from ..cancel import CancelToken
from ..events import Reporter
from ..proc import run_capture, run_streaming

# mlx-lm 출력 예: "Iter 10: Train loss 2.345, Learning Rate 1.000e-04, It/sec 1.2, Tokens/sec 300.1, Trained Tokens 1234, Peak mem 3.456 GB"
#                "Iter 20: Val loss 2.100, Val took 3.2s"
_ITER_RE = re.compile(r"Iter\s+(\d+):\s+(Train|Val)\s+loss\s+([0-9.]+|nan|inf)", re.IGNORECASE)
_TPS_RE = re.compile(r"Tokens/sec\s+([0-9.]+)")
_PEAK_RE = re.compile(r"Peak mem\s+([0-9.]+)\s*GB", re.IGNORECASE)
_LR_RE = re.compile(r"Learning Rate\s+([0-9.eE+-]+)")
# mlx-lm 0.30+ 표 형식 예: "    20    0.060 ▼      113      4.9k"  /  "    20    val 0.054    0.70s"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_NEW_VAL_RE = re.compile(r"^\s*(\d+)\s+val\s+([0-9.]+|nan|inf)\b", re.IGNORECASE)
_NEW_TRAIN_RE = re.compile(r"^\s*(\d+)\s+([0-9.]+|nan|inf)\s*[▲▼=]?\s+([0-9.]+)k?\s+[0-9.]+[kMG]?\s*$", re.IGNORECASE)


@dataclass
class TrainProgress:
    step: int = 0
    total: int = 0
    train_loss: float | None = None
    val_loss: float | None = None
    tokens_per_sec: float | None = None
    peak_mem_gb: float | None = None
    lr: float | None = None


@dataclass
class TrainResult:
    adapter_path: Path | None
    model_path: Path
    final_train_loss: float | None
    final_val_loss: float | None
    peak_mem_gb: float | None
    full_model: bool = False  # 전체 파인튜닝이면 adapter 대신 전체 가중치


def _float(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_train_line(line: str, prog: TrainProgress) -> bool:
    """학습 로그 한 줄을 파싱해 prog를 갱신. 의미 있는 줄이면 True. (구/신 mlx-lm 형식 모두 지원)"""
    line = _ANSI_RE.sub("", line)
    m = _ITER_RE.search(line)
    if not m:
        if (v := _NEW_VAL_RE.match(line)):
            prog.step = int(v.group(1))
            prog.val_loss = _float(v.group(2))
            return True
        if (t := _NEW_TRAIN_RE.match(line)):
            prog.step = int(t.group(1))
            prog.train_loss = _float(t.group(2))
            prog.tokens_per_sec = _float(t.group(3))
            return True
        return False
    prog.step = int(m.group(1))
    try:
        val = float(m.group(3))
    except ValueError:
        val = float("nan")
    if m.group(2).lower() == "train":
        prog.train_loss = val
    else:
        prog.val_loss = val
    if (t := _TPS_RE.search(line)):
        prog.tokens_per_sec = float(t.group(1))
    if (p := _PEAK_RE.search(line)):
        prog.peak_mem_gb = max(prog.peak_mem_gb or 0.0, float(p.group(1)))
    if (lr := _LR_RE.search(line)):
        try:
            prog.lr = float(lr.group(1))
        except ValueError:
            pass
    return True


class TrainerEngine(ABC):
    id: str = ""
    name: str = ""
    description: str = ""
    requirements_file: str | None = None  # 추가 설치가 필요한 엔진만
    import_name: str = ""
    experimental: bool = False
    serve_model_name: str = "default_model"
    activity = None  # 학습 중 작업 신호등 (파이프라인이 설정)  # 학생 서버에 요청할 때 쓰는 모델 이름 (mlx_lm.server 규칙)

    def is_installed(self) -> bool:
        return importlib.util.find_spec(self.import_name) is not None

    def install_command(self) -> list[str]:
        req = str(paths.REPO_ROOT / (self.requirements_file or "requirements.txt"))
        uv = paths.uv_binary()
        if uv:
            return [uv, "pip", "install", "--python", sys.executable, "-r", req]
        return [sys.executable, "-m", "pip", "install", "-r", req]

    async def install(self, reporter: Reporter, cancel: CancelToken | None = None, activity=None) -> None:
        if not paths.uv_binary():
            # uv venv에는 pip이 없을 수 있음
            await run_streaming([sys.executable, "-m", "ensurepip", "--upgrade"], reporter, cancel)
        code = await run_streaming(self.install_command(), reporter, cancel, log_prefix=T('[설치] '), activity=activity)
        if code != 0:
            raise RuntimeError(T('{name} 설치 실패 (종료 코드 {code})', name=self.name, code=code))
        importlib.invalidate_caches()

    @abstractmethod
    async def supported_modes(self) -> list[str]: ...

    async def prepare_model(self, model_path: Path, cfg: TrainConfig, workdir: Path, reporter: Reporter,
                            cancel: CancelToken) -> Path:
        """학습에 쓸 모델 경로 반환(QLoRA면 양자화 등). 기본은 그대로."""
        return model_path

    @abstractmethod
    async def train(self, cfg: TrainConfig, model_path: Path, data_dir: Path, out_dir: Path, reporter: Reporter,
                    cancel: CancelToken, on_progress: Callable[[TrainProgress], None]) -> TrainResult: ...

    @abstractmethod
    def serve_command(self, model_path: Path, adapter_path: Path | None, port: int) -> list[str]:
        """학생 모델을 OpenAI 호환 서버로 띄우는 명령."""

    async def _run_and_track(self, cmd: list[str], cfg: TrainConfig, reporter: Reporter, cancel: CancelToken,
                             on_progress: Callable[[TrainProgress], None]) -> TrainProgress:
        prog = TrainProgress(total=cfg.iters)
        reported_peak = {"v": False}

        def on_line(line: str) -> None:
            if parse_train_line(line, prog):
                if _PEAK_RE.search(line):
                    reported_peak["v"] = True
                on_progress(prog)

        def on_start(pid: int) -> None:
            asyncio.get_running_loop().create_task(_sample_memory(pid, prog, reported_peak))

        code = await run_streaming(cmd, reporter, cancel, on_line=on_line, log_prefix=f"[{self.id}] ",
                                   on_start=on_start, activity=self.activity)
        if code != 0:
            raise RuntimeError(T('학습 프로세스가 실패했습니다 (종료 코드 {code}). 로그를 확인하세요.', code=code))
        return prog


async def _sample_memory(pid: int, prog: TrainProgress, reported: dict) -> None:
    """학습 프로세스(와 자식)의 메모리 사용량을 1초마다 측정해 최대값을 기록 (엔진이 직접 보고하지 않을 때)."""
    try:
        import psutil

        root = psutil.Process(pid)
        while root.is_running() and root.status() != psutil.STATUS_ZOMBIE:
            if not reported["v"]:
                total = 0
                for p in [root] + root.children(recursive=True):
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        pass
                prog.peak_mem_gb = max(prog.peak_mem_gb or 0.0, total / 1024 ** 3)
            await asyncio.sleep(1.0)
    except Exception:
        return


# ----------------------------------------------------------------------------- 학생 모델 다운로드
def local_model_dir(model_id: str) -> Path:
    p = Path(model_id).expanduser()
    if p.is_dir():
        return p
    return paths.MODELS_DIR / model_id.replace("/", "__")


_DOWNLOAD_SCRIPT = r"""
import sys, json
from huggingface_hub import snapshot_download
a = json.loads(sys.argv[1])
snapshot_download(a["repo"], local_dir=a["dir"], token=a.get("token") or None,
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "tokenizer*", "*.jinja"])
print("DOWNLOAD_DONE")
"""


async def download_model(model_id: str, hf_token: str, reporter: Reporter, cancel: CancelToken,
                         on_progress: Callable[[int, int], None], activity=None) -> Path:
    """HF Hub에서 학생 모델을 받는다. 서브프로세스로 실행해 취소 시 즉시 중단하고 미완성 파일을 지운다.
    원격 코드(*.py)는 받지 않는다."""
    target = local_model_dir(model_id)
    if Path(model_id).expanduser().is_dir() or (target / ".download_complete").exists():
        if activity is not None:  # 이미 준비된 모델: 바로 완료 표시
            activity.begin(T("이미 준비됨"))
            activity.end(True)
        on_progress(1, 1)
        return target
    done_marker = target / ".download_complete"
    target.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        from huggingface_hub import HfApi

        info = await asyncio.to_thread(HfApi().model_info, model_id, files_metadata=True, token=hf_token or None)
        pats = (".json", ".safetensors", ".model", ".tiktoken", ".txt", ".jinja")
        total = sum((s.size or 0) for s in info.siblings or []
                    if s.rfilename.endswith(pats) or s.rfilename.startswith("tokenizer"))
    except Exception as e:  # 크기 정보 없이도 진행
        reporter.log("WARN", T('모델 크기 조회 실패(진행률 없이 다운로드): {e}', e=e))

    last = {"size": -1}

    async def poll() -> None:
        while True:
            await asyncio.sleep(1.0)
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if activity is not None and size > last["size"]:
                activity.beat(detail=f"{size / 1e9:.2f} GB")
            last["size"] = size
            on_progress(min(size, total) if total else 0, total or 0)

    if activity is not None:
        activity.begin()

    poller = asyncio.create_task(poll())
    args = json.dumps({"repo": model_id, "dir": str(target), "token": hf_token})
    try:
        code = await run_streaming([sys.executable, "-c", _DOWNLOAD_SCRIPT, args], None, cancel)
    except BaseException:
        poller.cancel()
        shutil.rmtree(target, ignore_errors=True)  # 미완성 파일 정리
        if activity is not None:
            activity.end(False, T("중단됨"))
        raise
    poller.cancel()
    if activity is not None:
        activity.end(code == 0, "" if code == 0 else T("다운로드 실패"))
    if code != 0:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(T('학생 모델 다운로드 실패: {model_id} (모델 ID, 인터넷, HF 토큰을 확인하세요)', model_id=model_id))
    done_marker.write_text("ok")
    on_progress(total or 1, total or 1)
    return target


async def help_text(cmd: list[str]) -> str:
    code, out = await run_capture(cmd + ["--help"], timeout=60)
    return out if code == 0 else ""
