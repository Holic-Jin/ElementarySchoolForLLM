"""PyTorch + PEFT 학습 엔진 (Apple GPU / MPS). PEFT의 다양한 LoRA 변형(PiSSA 등)을 지원한다."""
from __future__ import annotations

from ...i18n import T

import json
import sys
from pathlib import Path
from typing import Callable

from ..algorithm import TrainConfig
from ..cancel import CancelToken
from ..events import Reporter
from .base import TrainerEngine, TrainProgress, TrainResult

HERE = Path(__file__).resolve().parent


class TorchEngine(TrainerEngine):
    id = "torch"
    name = "PyTorch + PEFT (MPS)"
    description = (T('transformers·PEFT 기반으로 LoRA 변형(PiSSA 등)을 가장 폭넓게 지원합니다. MLX보다 느리고 메모리를 더 쓰며, Mac에서는 4bit(QLoRA) 학습이 불가합니다. 설치 용량 약 2GB 추가.'))
    requirements_file = "requirements-torch.txt"
    import_name = "peft"

    def is_installed(self) -> bool:
        import importlib.util

        return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft"))

    async def supported_modes(self) -> list[str]:
        return ["lora", "rslora", "dora", "pissa", "full"] if self.is_installed() else []

    async def train(self, cfg: TrainConfig, model_path: Path, data_dir: Path, out_dir: Path, reporter: Reporter,
                    cancel: CancelToken, on_progress: Callable[[TrainProgress], None]) -> TrainResult:
        if cfg.mode == "qlora":
            raise RuntimeError(T('PyTorch 엔진은 Mac에서 QLoRA(4bit 학습)를 지원하지 않습니다. MLX-LM 엔진을 사용하세요.'))
        out_dir.mkdir(parents=True, exist_ok=True)
        conf = {"model": str(model_path), "data": str(data_dir), "out_dir": str(out_dir), "mode": cfg.mode,
                "rank": cfg.rank, "alpha": cfg.alpha, "dropout": cfg.dropout, "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size, "iters": cfg.iters, "max_seq_len": cfg.max_seq_len,
                "steps_per_report": cfg.steps_per_report, "steps_per_eval": cfg.steps_per_eval,
                "grad_checkpoint": cfg.grad_checkpoint, "seed": cfg.seed}
        cfg_path = out_dir / "train_config.json"
        cfg_path.write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
        reporter.log("INFO", T('PyTorch 학습 시작: {mode}, LR {learning_rate:.2e}, {iters} step', mode=cfg.mode, learning_rate=cfg.learning_rate, iters=cfg.iters))
        cmd = [sys.executable, str(HERE / "torch_train.py"), str(cfg_path)]
        prog = await self._run_and_track(cmd, cfg, reporter, cancel, on_progress)
        return TrainResult(adapter_path=out_dir, model_path=model_path, final_train_loss=prog.train_loss,
                           final_val_loss=prog.val_loss, peak_mem_gb=prog.peak_mem_gb, full_model=cfg.mode == "full")

    def serve_command(self, model_path: Path, adapter_path: Path | None, port: int) -> list[str]:
        cmd = [sys.executable, str(HERE / "torch_serve.py"), "--model", str(model_path), "--port", str(port)]
        if adapter_path and adapter_path.exists() and any(adapter_path.glob("*.safetensors")):
            cmd += ["--adapter-path", str(adapter_path)]
        return cmd
