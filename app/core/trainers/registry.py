"""학습 엔진 목록."""
from __future__ import annotations

from ...i18n import T

from .base import TrainerEngine
from .mlx_engine import MLXEngine
from .mlx_lora_ext_engine import MLXLoraExtEngine
from .torch_engine import TorchEngine

_ENGINES: dict[str, TrainerEngine] = {}

MODE_LABELS = {
    "auto": T('자동 추천'),
    "lora": T('LoRA (기본)'),
    "qlora": T('QLoRA (4bit 베이스)'),
    "dora": "DoRA",
    "rslora": T('rsLoRA (α/√r 스케일)'),
    "pissa": T('PiSSA 초기화'),
    "full": T('전체 파인튜닝'),
    "dpo": T('DPO (선호도 학습)'),
    "orpo": T('ORPO (선호도 학습)'),
}


def engines() -> dict[str, TrainerEngine]:
    if not _ENGINES:
        for e in (MLXEngine(), TorchEngine(), MLXLoraExtEngine()):
            _ENGINES[e.id] = e
    return _ENGINES


def get_engine(engine_id: str) -> TrainerEngine:
    return engines()[engine_id]


async def pick_auto_engine(mode: str, feasible_by_engine: dict[str, list[str]]) -> str:
    """'자동': 원하는 모드를 지원하고 설치되어 있으며 메모리가 가장 적게 드는 엔진(MLX 계열 우선)."""
    for eid in ("mlx", "mlxext", "torch"):
        eng = engines()[eid]
        if not eng.is_installed():
            continue
        modes = await eng.supported_modes()
        if (mode == "auto" or mode in modes) and (mode == "auto" or mode in feasible_by_engine.get(eid, [mode])):
            return eid
    return "mlx"
