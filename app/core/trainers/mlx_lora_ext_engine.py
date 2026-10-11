"""mlx-lm-lora 엔진 (커뮤니티 확장, 실험적, Apache-2.0).

MLX 기반으로 4/6/8bit 양자화 학습과 DPO·ORPO 같은 선호도 학습을 지원한다.
설치된 버전의 --help를 읽어 지원 옵션을 확인한 뒤 사용한다(버전마다 옵션이 바뀔 수 있음).
"""
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
from .base import TrainerEngine, TrainProgress, TrainResult, help_text
from .mlx_engine import lora_keys_for

CMD = [sys.executable, "-m", "mlx_lm_lora.train"]


class MLXLoraExtEngine(TrainerEngine):
    id = "mlxext"
    name = T('mlx-lm-lora (커뮤니티, 실험적)')
    description = (T('MLX 기반 확장 트레이너. 4/6/8bit 양자화 학습과 DPO·ORPO 선호도 학습을 지원합니다. 커뮤니티 프로젝트라 버전에 따라 동작이 달라질 수 있습니다.'))
    requirements_file = "requirements-mlxext.txt"
    import_name = "mlx_lm_lora"
    experimental = True

    def __init__(self) -> None:
        self._help: str | None = None

    async def _h(self) -> str:
        if self._help is None:
            self._help = await help_text(CMD)
        return self._help

    @staticmethod
    def _choices(h: str, flag: str) -> set[str]:
        m = re.search(re.escape(flag) + r"\s*\{([^}]*)\}", h)
        return {x.strip() for x in m.group(1).split(",")} if m else set()

    async def supported_modes(self) -> list[str]:
        if not self.is_installed():
            return []
        h = await self._h()
        types = self._choices(h, "--train-type") or {"lora"}
        modes_ = self._choices(h, "--train-mode")
        out = ["lora"]
        if "--load-in-4bits" in h:
            out.append("qlora")
        if "dora" in types:
            out.append("dora")
        if "full" in types:
            out.append("full")
        for m in ("dpo", "orpo"):
            if m in modes_:
                out.append(m)
        return out

    async def train(self, cfg: TrainConfig, model_path: Path, data_dir: Path, out_dir: Path, reporter: Reporter,
                    cancel: CancelToken, on_progress: Callable[[TrainProgress], None]) -> TrainResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        h = await self._h()
        if not h:
            raise RuntimeError(T('mlx-lm-lora 실행 파일을 찾을 수 없습니다. 엔진을 다시 설치하세요.'))
        y: dict = {"model": str(model_path), "train": True, "data": str(data_dir), "seed": cfg.seed, "num_layers": -1,
                   "batch_size": cfg.batch_size, "iters": cfg.iters, "val_batches": -1,
                   "learning_rate": float(cfg.learning_rate), "steps_per_report": cfg.steps_per_report,
                   "steps_per_eval": cfg.steps_per_eval, "adapter_path": str(out_dir),
                   "save_every": max(cfg.iters, 100), "max_seq_length": cfg.max_seq_len,
                   "grad_checkpoint": cfg.grad_checkpoint, "fuse": False}
        y["train_type"] = {"dora": "dora", "full": "full"}.get(cfg.mode, "lora")
        y["train_mode"] = cfg.mode if cfg.mode in ("dpo", "orpo") else "sft"
        if cfg.mode == "qlora":
            y["load_in_4bits"] = True
        if "--mask-prompt" in h and y["train_mode"] == "sft":
            y["mask_prompt"] = True
        if cfg.mode != "full":
            lp = {"rank": cfg.rank, "scale": round(cfg.scale, 4), "dropout": cfg.dropout}
            keys = lora_keys_for(model_path)
            if keys:
                lp["keys"] = keys
            y["lora_parameters"] = lp
        cfg_path = out_dir / "train_config.yaml"
        cfg_path.write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False), encoding="utf-8")
        reporter.log("INFO", T('mlx-lm-lora 학습 시작: {train_mode}/{train_type}, LR {learning_rate:.2e}', train_mode=y['train_mode'], train_type=y['train_type'], learning_rate=cfg.learning_rate))
        prog = await self._run_and_track(CMD + ["--config", str(cfg_path)], cfg, reporter, cancel, on_progress)
        return TrainResult(adapter_path=out_dir, model_path=model_path, final_train_loss=prog.train_loss,
                           final_val_loss=prog.val_loss, peak_mem_gb=prog.peak_mem_gb, full_model=False)

    def serve_command(self, model_path: Path, adapter_path: Path | None, port: int) -> list[str]:
        cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", str(model_path), "--port", str(port),
               "--host", "127.0.0.1"]
        if adapter_path and (adapter_path / "adapters.safetensors").exists():
            cmd += ["--adapter-path", str(adapter_path)]
        return cmd
