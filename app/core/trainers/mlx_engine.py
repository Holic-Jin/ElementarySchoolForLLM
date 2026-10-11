"""MLX-LM 학습 엔진 (기본·권장). Apple 공식 MLX 기반으로 통합메모리에 최적화되어 있다."""
from __future__ import annotations

from ...i18n import T

import re
import sys
from pathlib import Path
from typing import Callable

import yaml

from ..algorithm import TrainConfig
from ..cancel import CancelToken
from ..events import Reporter
from ..proc import run_capture, run_streaming
from .base import TrainerEngine, TrainProgress, TrainResult, help_text

# 모델 계열별 "모든 선형층" 키. 모르는 계열은 mlx-lm 기본값 사용.
LLAMA_LIKE_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
PHI3_KEYS = ["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]
LLAMA_LIKE_TYPES = {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3", "gemma3_text", "olmo2",
                    "granite", "exaone", "internlm2", "cohere", "starcoder2", "smollm3"}

_cli_cache: dict[str, list[str]] = {}


async def mlx_cli(sub: str) -> list[str]:
    """설치된 mlx-lm 버전에 맞는 실행 방식을 찾는다: `python -m mlx_lm <sub>` 또는 `python -m mlx_lm.<sub>`."""
    if sub in _cli_cache:
        return _cli_cache[sub]
    for cand in ([sys.executable, "-m", "mlx_lm", sub], [sys.executable, "-m", f"mlx_lm.{sub}"]):
        code, out = await run_capture(cand + ["--help"], timeout=90)
        if code == 0 and "usage" in out.lower():
            _cli_cache[sub] = cand
            return cand
    raise RuntimeError(T("mlx-lm의 '{sub}' 명령을 찾을 수 없습니다. mlx-lm 설치를 확인하세요.", sub=sub))


def lora_keys_for(model_dir: Path) -> list[str] | None:
    try:
        import json

        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        mtype = str((cfg.get("text_config") or cfg).get("model_type", cfg.get("model_type", "")))
    except Exception:
        return None
    if mtype in LLAMA_LIKE_TYPES:
        return LLAMA_LIKE_KEYS
    if mtype in ("phi3", "phimoe"):
        return PHI3_KEYS
    return None


class MLXEngine(TrainerEngine):
    id = "mlx"
    name = T('MLX-LM (기본·권장)')
    description = T('MLX 기반. 통합메모리에 최적화되어 속도·메모리가 가장 유리합니다. LoRA·QLoRA·DoRA·rsLoRA·전체 파인튜닝.')
    import_name = "mlx_lm"

    def __init__(self) -> None:
        self._help: str | None = None

    async def _lora_help(self) -> str:
        if self._help is None:
            self._help = await help_text(await mlx_cli("lora"))
        return self._help

    async def supported_modes(self) -> list[str]:
        if not self.is_installed():
            return []
        h = await self._lora_help()
        modes = ["lora", "rslora", "qlora"]
        m = re.search(r"--fine-tune-type\s*\{([^}]*)\}", h)
        types = {t.strip() for t in m.group(1).split(",")} if m else ({"lora"} if "--fine-tune-type" not in h else set())
        if "--fine-tune-type" in h and not m:
            types = {"lora", "dora", "full"}  # 선택지 표시가 없는 버전
        if "dora" in types:
            modes.append("dora")
        if "full" in types:
            modes.append("full")
        return modes

    async def prepare_model(self, model_path: Path, cfg: TrainConfig, workdir: Path, reporter: Reporter,
                            cancel: CancelToken) -> Path:
        if cfg.mode != "qlora":
            return model_path
        qdir = model_path.parent / (model_path.name + "__mlx-4bit")
        if (qdir / "config.json").exists():
            return qdir
        reporter.log("INFO", T('QLoRA: 학생 모델을 4bit로 양자화합니다.'))
        cmd = await mlx_cli("convert") + ["--hf-path", str(model_path), "--mlx-path", str(qdir), "-q", "--q-bits", "4"]
        code = await run_streaming(cmd, reporter, cancel, log_prefix="[convert] ")
        if code != 0:
            raise RuntimeError(T('4bit 양자화 실패'))
        return qdir

    def build_yaml(self, cfg: TrainConfig, model_path: Path, data_dir: Path, adapter_dir: Path, help_txt: str) -> dict:
        y: dict = {
            "model": str(model_path),
            "train": True,
            "data": str(data_dir),
            "seed": cfg.seed,
            "num_layers": -1,
            "batch_size": cfg.batch_size,
            "iters": cfg.iters,
            "val_batches": -1,
            "learning_rate": float(cfg.learning_rate),
            "steps_per_report": cfg.steps_per_report,
            "steps_per_eval": cfg.steps_per_eval,
            "adapter_path": str(adapter_dir),
            "save_every": max(cfg.iters, 100),
            "max_seq_length": cfg.max_seq_len,
            "grad_checkpoint": cfg.grad_checkpoint,
        }
        if "--fine-tune-type" in help_txt:
            y["fine_tune_type"] = {"dora": "dora", "full": "full"}.get(cfg.mode, "lora")
        if "--mask-prompt" in help_txt:
            y["mask_prompt"] = True  # 답변 부분만 학습
        if cfg.mode != "full":
            lp = {"rank": cfg.rank, "scale": round(cfg.scale, 4), "dropout": cfg.dropout}
            keys = lora_keys_for(model_path)
            if keys:
                lp["keys"] = keys
            y["lora_parameters"] = lp
        return y

    async def train(self, cfg: TrainConfig, model_path: Path, data_dir: Path, out_dir: Path, reporter: Reporter,
                    cancel: CancelToken, on_progress: Callable[[TrainProgress], None]) -> TrainResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        help_txt = await self._lora_help()
        y = self.build_yaml(cfg, model_path, data_dir, out_dir, help_txt)
        cfg_path = out_dir / "train_config.yaml"
        cfg_path.write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False), encoding="utf-8")
        rank = "" if cfg.mode == "full" else f", rank {cfg.rank}"
        reporter.log("INFO", T('MLX-LM 학습 시작: {mode}{rank}, LR {learning_rate:.2e}, {iters} step', mode=cfg.mode, rank=rank, learning_rate=cfg.learning_rate, iters=cfg.iters))
        cmd = await mlx_cli("lora") + ["--config", str(cfg_path)]
        prog = await self._run_and_track(cmd, cfg, reporter, cancel, on_progress)
        return TrainResult(adapter_path=out_dir, model_path=model_path, final_train_loss=prog.train_loss,
                           final_val_loss=prog.val_loss, peak_mem_gb=prog.peak_mem_gb, full_model=False)

    def serve_command(self, model_path: Path, adapter_path: Path | None, port: int) -> list[str]:
        cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", str(model_path), "--port", str(port),
               "--host", "127.0.0.1"]
        if adapter_path and (adapter_path / "adapters.safetensors").exists():
            cmd += ["--adapter-path", str(adapter_path)]
        return cmd

    async def fuse(self, model_path: Path, adapter_path: Path, save_path: Path, reporter: Reporter,
                   cancel: CancelToken, dequantize: bool = False) -> None:
        cmd = await mlx_cli("fuse") + ["--model", str(model_path), "--adapter-path", str(adapter_path),
                                       "--save-path", str(save_path)]
        if dequantize:
            h = await help_text(await mlx_cli("fuse"))
            cmd.append("--dequantize" if "--dequantize" in h else "--de-quantize")
        code = await run_streaming(cmd, reporter, cancel, log_prefix="[fuse] ")
        if code != 0:
            raise RuntimeError(T('어댑터 병합(fuse) 실패'))
