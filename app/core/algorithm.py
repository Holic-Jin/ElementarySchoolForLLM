"""학습 모드/rank/하이퍼파라미터 자동 추천 → 03_algorithm.md (초안 → 확정본).

근거 연구
- Thinking Machines, "LoRA Without Regret" (2025): 모든 선형층에 LoRA, LoRA LR ≈ 전체 FT LR × 10,
  작은 배치(<32), α=32·α/r 스케일, 데이터가 어댑터 용량을 넘으면 성능 하락
- Lee et al., "Learning Rate Matters: Vanilla LoRA May Suffice" (arXiv 2602.04998, 2026): LR을 튜닝하면 변형 간 차이 1~2%
- Hayou 등 후속 분석(2026): 10배 비율은 모델·데이터·초기화에 따라 달라짐 → 짧은 LR 탐색으로 확정
- Biderman et al., "LoRA Learns Less and Forgets Less" (2024): 지식 대량 주입은 전체 FT가 유리하나 망각이 큼
"""
from __future__ import annotations

from ..i18n import T

import datetime as _dt
import math
from dataclasses import asdict, dataclass, field
from typing import Callable

from .hardware import HardwareReport, ModelInfo, lora_params_per_rank

FULL_FT_BASE_LR = 1e-5
LORA_LR_MULT = 10.0
SHORT_RUN_LR_MULT = 15.0  # 100 step 미만 짧은 학습
RANKS = [8, 16, 32, 64, 128, 256]
BITS_PER_PARAM = 2.0  # LoRA 파라미터 1개가 저장할 수 있는 정보량(근사)
BITS_PER_TOKEN = 1.0  # SFT 데이터 정보량 상한 근사(첫 에폭 손실 ≈ 1 bit/token)
LORA_FAMILY = {"lora", "qlora", "dora", "rslora", "pissa", "dpo", "orpo"}

REFERENCES = [
    "Thinking Machines Lab, \"LoRA Without Regret\" (2025) — https://thinkingmachines.ai/blog/lora/",
    "\"Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning\" (arXiv 2602.04998, 2026)",
    "\"A Unified Study of LoRA Variants\" (arXiv 2601.22708, 2026)",
    "Biderman et al., \"LoRA Learns Less and Forgets Less\" (2024)",
    "Liu et al., \"DoRA: Weight-Decomposed Low-Rank Adaptation\" (ICML 2024)",
    "Kalajdzievski, \"rsLoRA: A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA\" (2023)",
    "Dettmers et al., \"QLoRA\" (2023)",
    T('참고(미탑재): Echo-LoRA (arXiv 2605.08177), PEFT with Learnable Rank (arXiv 2606.04325), GraLoRA'),
]


@dataclass
class TrainConfig:
    engine: str = "mlx"
    mode: str = "lora"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 2
    iters: int = 100
    max_seq_len: int = 2048
    lr_sweep: bool = True
    grad_checkpoint: bool = True
    seed: int = 42
    steps_per_report: int = 10
    steps_per_eval: int = 50
    n_train: int = 0
    n_valid: int = 0

    @property
    def scale(self) -> float:
        """LoRA 스케일: 기본 α/r, rsLoRA는 α/√r."""
        if self.mode == "rslora":
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scale"] = round(self.scale, 4)
        return d


@dataclass
class Recommendation:
    config: TrainConfig
    rationale: list[str] = field(default_factory=list)


def choose_rank(model: ModelInfo, data_tokens: int) -> tuple[int, str]:
    per_rank = lora_params_per_rank(model)
    data_bits = data_tokens * BITS_PER_TOKEN
    for r in RANKS:
        if per_rank * r * BITS_PER_PARAM >= data_bits:
            return r, (T('데이터 정보량 상한 ≈ {v:.2f}M bit, rank {r} 어댑터 용량 ≈ {v2:.1f}M bit → 용량이 데이터를 넘는 최소 rank', v=data_bits / 1e6, r=r, v2=per_rank * r * BITS_PER_PARAM / 1e6))
    return RANKS[-1], T('데이터가 매우 커서 최대 rank 사용(전체 파인튜닝도 고려)')


def recommend(goal_type: str, n_samples: int, avg_tokens: int, model: ModelInfo | None, report: HardwareReport | None,
              engine: str, overrides: dict | None = None,
              fits: Callable[[str, int, int, int], bool] | None = None) -> Recommendation:
    """overrides: 사용자가 UI에서 직접 지정한 값(mode/rank/learning_rate/epochs/batch_size/max_seq_len). 0/auto면 자동."""
    o = overrides or {}
    why: list[str] = []
    feasible = report.feasible_modes if report else ["lora"]
    total_tokens = max(1, n_samples * avg_tokens)

    # 1) 모드
    mode = o.get("mode") or "auto"
    if mode == "auto":
        if goal_type == "knowledge" and "full" in feasible:
            mode = "full"
            why.append(T("목표가 '지식 대량 주입'이고 메모리가 충분 → 전체 파인튜닝 (LoRA Learns Less and Forgets Less)"))
        elif "lora" in feasible:
            mode = "lora"
            why.append(T('바닐라 LoRA 기본 — LR을 맞추면 변형 간 차이는 1~2% (Learning Rate Matters, 2026)'))
        elif "qlora" in feasible:
            mode = "qlora"
            why.append(T('메모리가 부족해 4bit 베이스(QLoRA) 사용'))
        elif feasible:
            mode = feasible[0]
            why.append(T('가능한 모드 중 가장 가벼운 {mode} 사용', mode=mode))
        else:
            mode = "lora"
            why.append(T('사양 정보가 없어 LoRA로 가정(초안)'))
    else:
        why.append(T('사용자가 지정한 모드: {mode}', mode=mode))

    # 2) rank
    rank = int(o.get("rank") or 0)
    if mode == "full":
        rank = 0
    elif rank <= 0:
        if model:
            rank, r_why = choose_rank(model, total_tokens)
            why.append(r_why)
        else:
            rank = 16
            why.append(T('모델 정보 없음 → rank 16'))
        if o.get("mode", "auto") == "auto" and mode == "lora" and rank >= 64 and engine in ("mlx", "torch"):
            mode = "rslora"
            why.append(T('rank ≥ 64 → 스케일을 α/√r로 안정화(rsLoRA)'))
    else:
        why.append(T('사용자가 지정한 rank: {rank}', rank=rank))

    # 3) 시퀀스 길이
    max_seq = int(o.get("max_seq_len") or 0)
    if max_seq <= 0:
        need = int(avg_tokens * 1.8) + 64
        max_seq = min(4096, max(256, 1 << (need - 1).bit_length()))
        why.append(T('평균 문항 길이 ≈ {avg_tokens} 토큰 → 최대 길이 {max_seq}', avg_tokens=avg_tokens, max_seq=max_seq))

    # 4) 배치: 작은 배치 권장(<32). 메모리에 맞는 가장 큰 값(8→4→2→1)
    batch = int(o.get("batch_size") or 0)
    if batch <= 0:
        batch = 1
        for b in (8, 4, 2, 1):
            if fits is None or fits(mode, b, max_seq, max(rank, 1)):
                batch = b
                break
        why.append(T('배치 {batch} (LoRA는 큰 배치에 약함 — 32 미만 권장)', batch=batch))

    # 5) 에폭/스텝
    epochs = int(o.get("epochs") or 0)
    if epochs <= 0:
        epochs = 3 if n_samples < 500 else 2 if n_samples < 5000 else 1
    iters = max(1, math.ceil(epochs * max(1, n_samples) / batch))

    # 6) 학습률
    lr = float(o.get("learning_rate") or 0)
    if lr <= 0:
        if mode == "full":
            lr = FULL_FT_BASE_LR
            why.append(T('전체 파인튜닝 기본 LR {lr:.0e}', lr=lr))
        else:
            mult = SHORT_RUN_LR_MULT if iters < 100 else LORA_LR_MULT
            lr = FULL_FT_BASE_LR * mult
            why.append(T('LoRA LR = 전체 FT LR × {mult:g} = {lr:.1e} (LoRA Without Regret; 10배는 출발점이므로 탐색으로 확정)', mult=mult, lr=lr))
            if mode in ("dpo", "orpo"):
                lr /= 10
                why.append(T('선호도 학습은 SFT보다 LR을 10배 낮춤'))
    else:
        why.append(T('사용자가 지정한 LR: {lr:g}', lr=lr))

    sweep = bool(o.get("lr_sweep", True))
    if sweep and mode == "full":
        sweep = False
        why.append(T('전체 파인튜닝은 탐색 비용이 커서 학습률 탐색 생략'))
    cfg = TrainConfig(engine=engine, mode=mode, rank=rank if mode != "full" else 1,
                      learning_rate=lr, batch_size=batch, epochs=epochs, iters=iters, max_seq_len=max_seq,
                      lr_sweep=sweep, steps_per_eval=max(10, iters // 5),
                      steps_per_report=max(1, min(10, iters // 20 or 1)))
    return Recommendation(config=cfg, rationale=why)


def lr_candidates(base_lr: float) -> list[float]:
    return [base_lr * 0.5, base_lr, base_lr * 2.0]


def render_markdown(stage: str, plan: dict, rec: Recommendation, report: HardwareReport | None = None,
                    sweep: list[dict] | None = None, data_stats: dict | None = None) -> str:
    c = rec.config
    now = f"{_dt.datetime.now():%Y-%m-%d %H:%M}"
    lines = [f"## {stage} ({now})", ""]
    if data_stats:
        lines += [T('### 데이터'), *(f"- {k}: {v}" for k, v in data_stats.items()), ""]
    lines += [T('### 학습 설정'), T('| 항목 | 값 |'), "|---|---|",
              T('| 학습 엔진 | {engine} |', engine=c.engine), T('| 모드 | {mode} |', mode=c.mode)]
    if c.mode != "full":
        lines += [T('| rank / α / 스케일 | {rank} / {alpha:g} / {scale:.3f} |', rank=c.rank, alpha=c.alpha, scale=c.scale),
                  T('| 대상 레이어 | 모든 선형층 (q,k,v,o,gate,up,down) |')]
    lines += [T('| 학습률 | {learning_rate:.2e} (상수 스케줄) |', learning_rate=c.learning_rate), T('| 배치 | {batch_size} |', batch_size=c.batch_size),
              T('| 에폭 / 스텝 | {epochs} / {iters} |', epochs=c.epochs, iters=c.iters), T('| 최대 길이 | {max_seq_len} |', max_seq_len=c.max_seq_len),
              T('| 그래디언트 체크포인팅 | {v} |', v=T('예') if c.grad_checkpoint else T('아니오')),
              T('| 학습률 탐색 | {v} |', v=T('예') if c.lr_sweep else T('아니오')), ""]
    lines += [T('### 선정 근거'), *(f"- {w}" for w in rec.rationale), ""]
    if sweep:
        lines += [T('### 학습률 탐색 결과'), T('| 학습률 | 검증 손실 |'), "|---|---|"]
        lines += [f"| {s['lr']:.2e} | {s['val_loss'] if s.get('val_loss') is not None else T('실패')} |" for s in sweep]
        lines.append("")
    if report:
        lines += [T('### 사양'), T('- GPU 할당 가능 메모리: {available_gb:.1f}GB', available_gb=report.available_gb),
                  T('- 이 엔진에서 가능한 모드: {v}', v=', '.join(report.feasible_modes) or T('없음')), ""]
    return "\n".join(lines)


def algorithm_document(plan: dict, draft_section: str, final_section: str = "") -> str:
    head = [T('# 학습 알고리즘 — {get}', get=plan.get('title', '')), "",
            T('이 문서는 학습방안 확정 시 **초안**이 만들어지고, 품질 평가 후 실제 데이터로 **확정본**이 추가됩니다.'), ""]
    tail = [T('## 참고문헌'), *(f"- {r}" for r in REFERENCES), ""]
    return "\n".join(head) + draft_section + "\n" + (final_section + "\n" if final_section else "") + "\n".join(tail)
