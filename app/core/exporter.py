"""학습 결과 내보내기: 어댑터 병합(fuse) → Ollama 모델 등록."""
from __future__ import annotations

from ..i18n import T

import json
import sys
from pathlib import Path

from .cancel import CancelToken
from .events import Reporter
from .local_backends import ollama_create
from .proc import run_streaming

# Ollama가 safetensors 디렉터리에서 직접 가져올 수 있는 모델 구조 (2026 기준 공식 문서의 지원 목록 근사)
OLLAMA_IMPORTABLE = {"llama", "mistral", "mixtral", "gemma", "gemma2", "gemma3", "phi3", "qwen2", "qwen3"}

_TORCH_MERGE = r"""
import sys, json
from pathlib import Path
a = json.loads(sys.argv[1])
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
mj, mn = (int(x) for x in transformers.__version__.split(".")[:2])
kw = {"dtype": torch.bfloat16} if (mj, mn) >= (4, 56) else {"torch_dtype": torch.bfloat16}
m = AutoModelForCausalLM.from_pretrained(a["model"], **kw)
ad = Path(a["adapter"])
if (ad / "adapter_config.json").exists():
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, str(ad)).merge_and_unload()
else:
    m = AutoModelForCausalLM.from_pretrained(str(ad), **kw)
m.save_pretrained(a["out"], safe_serialization=True)
AutoTokenizer.from_pretrained(a["model"]).save_pretrained(a["out"])
print("MERGE_DONE")
"""


def model_type(model_dir: Path) -> str:
    try:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return str((cfg.get("text_config") or cfg).get("model_type", cfg.get("model_type", "")))
    except Exception:
        return ""


def ollama_supported(model_dir: Path) -> tuple[bool, str]:
    mt = model_type(model_dir)
    if mt in OLLAMA_IMPORTABLE:
        return True, ""
    return False, T("모델 구조 '{v}'는 Ollama 직접 가져오기 지원 목록에 없습니다. 결과 탭의 테스트 채팅(학생 서버)으로 사용하세요.", v=mt or T('알 수 없음'))


async def fuse(engine_id: str, model_path: Path, adapter_path: Path, out_dir: Path, reporter: Reporter,
               cancel: CancelToken) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if engine_id in ("mlx", "mlxext"):
        from .trainers.mlx_engine import MLXEngine

        quantized = "__mlx-4bit" in model_path.name
        await MLXEngine().fuse(model_path, adapter_path, out_dir, reporter, cancel, dequantize=quantized)
    else:
        args = json.dumps({"model": str(model_path), "adapter": str(adapter_path), "out": str(out_dir)})
        code = await run_streaming([sys.executable, "-c", _TORCH_MERGE, args], reporter, cancel, log_prefix="[merge] ")
        if code != 0:
            raise RuntimeError(T('PyTorch 어댑터 병합 실패'))
    return out_dir


async def export_to_ollama(engine_id: str, model_path: Path, adapter_path: Path, project_dir: Path, name: str,
                           system_prompt: str, reporter: Reporter, cancel: CancelToken) -> str:
    fused = await fuse(engine_id, model_path, adapter_path, project_dir / "fused_model", reporter, cancel)
    ok, why = ollama_supported(fused)
    if not ok:
        raise RuntimeError(why)
    mf = project_dir / "Modelfile"
    lines = [f"FROM {fused}"]
    if system_prompt:
        lines.append('SYSTEM """' + system_prompt.replace('"""', "'''") + '"""')
    mf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    await ollama_create(name, mf, reporter, cancel)
    reporter.log("INFO", T('Ollama 모델 등록 완료: {name}  (ollama run {name2})', name=name, name2=name))
    return name
