"""사양 측정 & 필요 사양 계산.

교사1 단계 시작 전에 호출되어, 학생 모델 추론·학습에 필요한 메모리/디스크를 학습 엔진·모드별로 추정한다.
추정 계수는 ENGINE_FACTORS 에 모아 두었고, 실제 학습 후 측정한 최대 메모리를 리포트에 함께 기록해 보정에 쓴다.
"""
from __future__ import annotations

from ..i18n import T

import json
import math
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

GB = 1024 ** 3

# 엔진별 메모리 계수(경험값). overhead_gb: 프레임워크 고정 비용, act_mult: 활성값 배수
ENGINE_FACTORS: dict[str, dict[str, float]] = {
    "mlx": {"overhead_gb": 0.6, "act_mult": 1.0, "full_bytes_per_param": 12.0},
    "mlxext": {"overhead_gb": 0.7, "act_mult": 1.0, "full_bytes_per_param": 12.0},
    "torch": {"overhead_gb": 1.2, "act_mult": 1.3, "full_bytes_per_param": 16.0},
}
ENGINE_MODES: dict[str, list[str]] = {
    "mlx": ["qlora", "lora", "rslora", "dora", "full"],
    "torch": ["lora", "rslora", "dora", "pissa", "full"],
    "mlxext": ["qlora", "lora", "dora", "full", "dpo", "orpo"],
}
MODE_ORDER_LIGHT_FIRST = ["qlora", "lora", "rslora", "pissa", "dora", "dpo", "orpo", "full"]
SAFETY = 1.2


@dataclass
class HardwareInfo:
    chip: str
    arch: str
    macos: str
    total_mem_gb: float
    gpu_mem_gb: float  # Metal이 권장하는 최대 작업 메모리
    disk_free_gb: float


@dataclass
class ModelInfo:
    model_id: str
    model_type: str
    params: int
    hidden: int
    layers: int
    vocab: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    tied_embeddings: bool
    quant_bits: int = 0  # 이미 양자화된 모델(mlx-community 등)이면 비트 수

    @property
    def params_b(self) -> float:
        return self.params / 1e9


@dataclass
class MemoryEstimate:
    engine: str
    mode: str
    weights_gb: float
    trainable_params: int
    optimizer_gb: float
    activations_gb: float
    total_gb: float


@dataclass
class HardwareReport:
    ok: bool
    hardware: HardwareInfo
    model: ModelInfo | None
    engine: str
    requested_mode: str
    batch_size: int
    seq_len: int
    available_gb: float
    reserved_for_local_llm_gb: float
    inference_gb: float
    estimates: list[MemoryEstimate] = field(default_factory=list)
    feasible_modes: list[str] = field(default_factory=list)
    other_engine_feasible: dict[str, list[str]] = field(default_factory=dict)
    disk_needed_gb: float = 0.0
    disk_ok: bool = True
    messages: list[str] = field(default_factory=list)
    measured_peak_gb: float | None = None  # 학습 후 실제 측정값

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- 하드웨어 측정
def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def detect(path_for_disk: Path | None = None) -> HardwareInfo:
    import psutil

    total = psutil.virtual_memory().total
    gpu = 0
    try:
        import mlx.core as mx

        info_fn = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info")
        info = info_fn()
        gpu = int(info.get("max_recommended_working_set_size", 0))
    except Exception:
        gpu = 0
    if not gpu:
        gpu = int(total * 0.70)  # macOS 기본 GPU 할당 한도 근사치
    disk = shutil.disk_usage(str(path_for_disk or Path.home())).free
    return HardwareInfo(chip=_sysctl("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon",
                        arch=platform.machine(), macos=platform.mac_ver()[0], total_mem_gb=round(total / GB, 2),
                        gpu_mem_gb=round(gpu / GB, 2), disk_free_gb=round(disk / GB, 2))


# ----------------------------------------------------------------------------- 모델 정보
def load_model_config(model_id: str, hf_token: str = "") -> dict:
    """로컬 경로면 config.json을 읽고, 아니면 HF Hub에서 config.json만 받는다(가중치 다운로드 전)."""
    p = Path(model_id).expanduser()
    if p.is_dir():
        return json.loads((p / "config.json").read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "config.json", token=hf_token or None)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_info_from_config(model_id: str, cfg: dict) -> ModelInfo:
    c = cfg.get("text_config") or cfg  # 멀티모달 모델의 텍스트 부분
    hidden = int(c.get("hidden_size") or c.get("n_embd") or c.get("d_model") or 0)
    layers = int(c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers") or 0)
    vocab = int(c.get("vocab_size") or cfg.get("vocab_size") or 32000)
    heads = int(c.get("num_attention_heads") or c.get("n_head") or max(1, hidden // 128))
    kv_heads = int(c.get("num_key_value_heads") or heads)
    head_dim = int(c.get("head_dim") or (hidden // heads if heads else 128))
    inter = int(c.get("intermediate_size") or c.get("ffn_dim") or c.get("n_inner") or 4 * hidden)
    tied = bool(c.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))
    experts = int(c.get("num_local_experts") or c.get("n_routed_experts") or c.get("num_experts") or 1)
    moe_inter = int(c.get("moe_intermediate_size") or inter)
    shared = int(c.get("n_shared_experts") or 0)
    if not hidden or not layers:
        raise ValueError(T('config.json에서 hidden_size/num_hidden_layers를 찾을 수 없습니다.'))
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    attn = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    if experts > 1:
        mlp = experts * 3 * hidden * moe_inter + shared * 3 * hidden * moe_inter + hidden * experts
    else:
        mlp = 3 * hidden * inter
    params = layers * (attn + mlp + 2 * hidden) + vocab * hidden + (0 if tied else vocab * hidden)
    qbits = 0
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if isinstance(q, dict):
        try:
            qbits = int(q.get("bits") or 0)
        except (TypeError, ValueError):
            qbits = 0
    return ModelInfo(model_id=model_id, model_type=str(c.get("model_type") or cfg.get("model_type", "")),
                     params=int(params), hidden=hidden, layers=layers, vocab=vocab, intermediate=inter, heads=heads,
                     kv_heads=kv_heads, head_dim=head_dim, tied_embeddings=tied, quant_bits=qbits)


def lora_params_per_rank(m: ModelInfo) -> int:
    """모든 선형층(q,k,v,o,gate,up,down)에 LoRA를 붙일 때 rank 1당 학습 파라미터 수."""
    q_dim = m.heads * m.head_dim
    kv_dim = m.kv_heads * m.head_dim
    per_layer = (m.hidden + q_dim) + 2 * (m.hidden + kv_dim) + (q_dim + m.hidden) + 3 * (m.hidden + m.intermediate)
    return per_layer * m.layers


# ----------------------------------------------------------------------------- 메모리 추정
def weight_bytes_per_param(engine: str, mode: str, m: ModelInfo) -> float:
    if mode == "qlora" or (m.quant_bits and engine != "torch"):
        bits = m.quant_bits or 4
        return bits / 8 + 0.0625  # 그룹 스케일/바이어스 포함 근사
    return 2.0  # bf16/fp16


def inference_memory_gb(m: ModelInfo, engine: str = "mlx", seq_len: int = 2048) -> float:
    w = m.params * weight_bytes_per_param(engine, "lora", m)
    kv = 2 * m.layers * seq_len * m.kv_heads * m.head_dim * 2
    return (w + kv) * SAFETY / GB + 0.5


def estimate_training_memory(m: ModelInfo, engine: str, mode: str, batch: int, seq: int, rank: int = 16) -> MemoryEstimate:
    f = ENGINE_FACTORS.get(engine, ENGINE_FACTORS["mlx"])
    bpp = weight_bytes_per_param(engine, mode, m)
    weights = m.params * bpp
    if mode == "full":
        trainable = m.params
        optimizer = m.params * (f["full_bytes_per_param"] - 2.0)  # 가중치 외 그래디언트+옵티마이저 상태
    else:
        trainable = lora_params_per_rank(m) * rank * (2 if mode == "dora" else 1)
        optimizer = trainable * (4 + 4 + 8)  # fp32 파라미터 + 그래디언트 + Adam 2상태
        if mode in ("dpo", "orpo"):
            weights *= 1.0  # 참조 모델은 어댑터를 끈 같은 모델을 사용
    # 활성값: 그래디언트 체크포인팅 기준 — 층 경계 저장(2*s*b*h*L) + 한 층 전체(34*s*b*h + 5*a*s^2*b) + 로짓(bf16 + fp32 log-softmax 근사)
    s, b, h, L, a = seq, batch, m.hidden, m.layers, m.heads
    act = 2 * s * b * h * L + (34 * s * b * h + 5 * a * s * s * b) + s * b * m.vocab * 6
    if mode in ("dpo", "orpo"):
        act *= 2  # chosen/rejected 두 시퀀스
    act *= f["act_mult"]
    total = (weights + optimizer + act) * SAFETY + f["overhead_gb"] * GB
    return MemoryEstimate(engine=engine, mode=mode, weights_gb=round(weights / GB, 2), trainable_params=int(trainable),
                          optimizer_gb=round(optimizer / GB, 2), activations_gb=round(act / GB, 2),
                          total_gb=round(total / GB, 2))


def disk_needed_gb(m: ModelInfo, data_tokens: int = 0) -> float:
    return round(m.params * 2 * 2.5 / GB + data_tokens * 8 / GB + 1.0, 2)


# ----------------------------------------------------------------------------- 판정
def check(hw: HardwareInfo, m: ModelInfo, engine: str, requested_mode: str, batch: int, seq: int, rank: int,
          reserved_local_gb: float = 0.0, data_tokens: int = 0) -> HardwareReport:
    """선택한 엔진 기준으로 학습 가능 여부를 판정한다.

    - requested_mode == "auto": 가능한 가장 가벼운 모드조차 불가하면 실패
    - 그 외: 요청한 모드가 불가하면 실패 (자동 전환하지 않음)
    학습 시점에는 로컬 교사/오케스트레이터 모델을 내리므로(reserved_local_gb는 데이터 생성 중 동시 구동 판단용)
    학습 가능 여부는 GPU 메모리 전체 기준으로 판정한다.
    """
    available = hw.gpu_mem_gb
    report = HardwareReport(ok=False, hardware=hw, model=m, engine=engine, requested_mode=requested_mode,
                            batch_size=batch, seq_len=seq, available_gb=available,
                            reserved_for_local_llm_gb=round(reserved_local_gb, 2),
                            inference_gb=round(inference_memory_gb(m, engine, seq), 2))
    for eng, modes in ENGINE_MODES.items():
        feas = []
        for mode in modes:
            est = estimate_training_memory(m, eng, mode, batch if mode != "full" else max(1, batch // 2), seq, rank)
            if eng == engine:
                report.estimates.append(est)
            if est.total_gb <= available:
                feas.append(mode)
        if eng == engine:
            report.feasible_modes = [x for x in MODE_ORDER_LIGHT_FIRST if x in feas]
        elif feas:
            report.other_engine_feasible[eng] = [x for x in MODE_ORDER_LIGHT_FIRST if x in feas]
    report.disk_needed_gb = disk_needed_gb(m, data_tokens)
    report.disk_ok = hw.disk_free_gb >= report.disk_needed_gb
    if requested_mode == "auto":
        mode_ok = bool(report.feasible_modes)
    else:
        mode_ok = requested_mode in report.feasible_modes
    if engine not in ENGINE_MODES:
        report.messages.append(T('알 수 없는 학습 엔진: {engine}', engine=engine))
        mode_ok = False
    elif requested_mode != "auto" and requested_mode not in ENGINE_MODES[engine]:
        report.messages.append(T("선택한 엔진({engine})은 '{requested_mode}' 모드를 지원하지 않습니다.", engine=engine, requested_mode=requested_mode))
    if not mode_ok:
        need = min((e.total_gb for e in report.estimates if requested_mode in ("auto", e.mode)), default=math.inf)
        report.messages.append(
            T('메모리 부족: 학습에 약 {need:.1f}GB가 필요하지만 GPU에 할당 가능한 메모리는 {available:.1f}GB입니다.', need=need, available=available))
        if report.feasible_modes:
            report.messages.append(T('이 엔진에서 가능한 모드: ') + ", ".join(report.feasible_modes))
        for eng, modes in report.other_engine_feasible.items():
            report.messages.append(T('{eng} 엔진으로는 가능: ', eng=eng) + ", ".join(modes))
    if not report.disk_ok:
        report.messages.append(T('디스크 부족: 약 {disk_needed_gb:.1f}GB 필요, 여유 {disk_free_gb:.1f}GB', disk_needed_gb=report.disk_needed_gb, disk_free_gb=hw.disk_free_gb))
    report.ok = mode_ok and report.disk_ok
    return report


def format_report_table(r: HardwareReport) -> str:
    lines = [T('칩: {chip}  /  macOS {macos}', chip=r.hardware.chip, macos=r.hardware.macos),
             T('통합 메모리: {total_mem_gb:.1f}GB  (GPU 할당 가능: {available_gb:.1f}GB)', total_mem_gb=r.hardware.total_mem_gb, available_gb=r.available_gb),
             T('디스크 여유: {disk_free_gb:.1f}GB  (필요: {disk_needed_gb:.1f}GB)', disk_free_gb=r.hardware.disk_free_gb, disk_needed_gb=r.disk_needed_gb)]
    if r.model:
        lines.append(T('학생 모델: {model_id}  (약 {params_b:.2f}B 파라미터, 추론 약 {inference_gb:.1f}GB)', model_id=r.model.model_id, params_b=r.model.params_b, inference_gb=r.inference_gb))
    lines.append(T('학습 엔진: {engine}  /  요청 모드: {requested_mode}  /  배치 {batch_size}, 길이 {seq_len}', engine=r.engine, requested_mode=r.requested_mode, batch_size=r.batch_size, seq_len=r.seq_len))
    lines.append("")
    lines.append(T('모드        필요 메모리   가능'))
    for e in r.estimates:
        mark = "O" if e.mode in r.feasible_modes else "X"
        lines.append(f"{e.mode:<10} {e.total_gb:>8.1f}GB     {mark}")
    if r.messages:
        lines.append("")
        lines.extend(r.messages)
    return "\n".join(lines)
