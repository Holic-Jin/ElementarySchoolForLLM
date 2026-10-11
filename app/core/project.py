"""프로젝트 폴더와 산출물 파일 이름.

데이터 폴더/projects/<이름>/
  project.json      프로젝트 상태
  1_plan/           학습방안 대화·확정안(영어)·데이터 명세·알고리즘
  2_generated/      교사1·교사2 생성 데이터, 평가 기록, 품질 보고서, 선호도 쌍
  3_dataset/        학습/검증 데이터 (엔진 입력용 사본 포함)
  4_training/       어댑터, 학습률 탐색, 병합 모델, 학습 로그
  5_reports/        사양 리포트, 학습 전후 비교, 출처 기록, 파이프라인 로그
단계별 폴더를 통째로 휴지통에 보내면 그 단계부터 다시 만들 수 있다(reset_from).
"""
from __future__ import annotations

import datetime as _dt
import re
import shutil
from pathlib import Path
from typing import Any

from . import paths
from .jsonutil import read_json, write_json

D_PLAN, D_GEN, D_DATA, D_TRAIN, D_REPORT = "1_plan", "2_generated", "3_dataset", "4_training", "5_reports"
SUBDIRS = (D_PLAN, D_GEN, D_DATA, D_TRAIN, D_REPORT)

F_CHAT = f"{D_PLAN}/00_plan_chat.json"
F_PLAN = f"{D_PLAN}/01_learning_plan.md"
F_SPEC = f"{D_PLAN}/02_data_spec.json"
F_ALGO = f"{D_PLAN}/03_algorithm.md"
F_ALGO_DRAFT = f"{D_PLAN}/.algorithm_draft.md"
F_T1 = f"{D_GEN}/05_teacher1.jsonl"
F_T2 = f"{D_GEN}/06_teacher2.jsonl"
F_QUALITY = f"{D_GEN}/07_quality_report.json"
F_JUDGE_LOG = f"{D_GEN}/07a_judgements.jsonl"
F_PREF = f"{D_GEN}/07b_preference_pairs.jsonl"
F_TRAIN = f"{D_DATA}/08_train.jsonl"
F_VALID = f"{D_DATA}/08_valid.jsonl"
F_TRAIN_LOG = f"{D_TRAIN}/09_training_log.txt"
F_HW = f"{D_REPORT}/04_hardware_report.json"
F_EVAL = f"{D_REPORT}/09b_eval_results.json"
F_PROV = f"{D_REPORT}/10_provenance.md"
F_PIPE_LOG = f"{D_REPORT}/pipeline_log.txt"
F_STATE = "project.json"

# 이전 버전(모든 파일이 프로젝트 폴더에 평평하게 있던 구조) → 새 위치
_LEGACY = {Path(f).name: f for f in (F_CHAT, F_PLAN, F_SPEC, F_ALGO, F_ALGO_DRAFT, F_T1, F_T2, F_QUALITY, F_JUDGE_LOG,
                                    F_PREF, F_TRAIN, F_VALID, F_TRAIN_LOG, F_HW, F_EVAL, F_PROV, F_PIPE_LOG)}
_LEGACY.update({"adapters": f"{D_TRAIN}/adapters", "lr_sweep": f"{D_TRAIN}/lr_sweep", "fused_model": f"{D_TRAIN}/fused_model",
                "Modelfile": f"{D_TRAIN}/Modelfile", "train_data": f"{D_DATA}/train_data",
                "train_data_pref": f"{D_DATA}/train_data_pref"})

# reset_from(단계): 그 단계와 이후 단계 폴더를 휴지통으로 보내고 상태를 되돌린다
RESET_STAGES = ("spec", "generated", "training")


def safe_name(name: str) -> str:
    name = re.sub(r"[^\w가-힣ぁ-んァ-ン一-龥\-]+", "_", name.strip())[:60].strip("_")
    return name or f"project_{_dt.datetime.now():%Y%m%d_%H%M%S}"


class Project:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self._migrate_flat()

    @classmethod
    def create(cls, name: str) -> "Project":
        d = paths.WORKSPACE / safe_name(name)
        p = cls(d)
        if not (d / F_STATE).exists():
            p.save_state({"name": name, "created": _dt.datetime.now().isoformat(timespec="seconds"), "status": "planning"})
        return p

    @staticmethod
    def list_all() -> list[str]:
        if not paths.WORKSPACE.exists():
            return []
        return sorted(p.name for p in paths.WORKSPACE.iterdir() if (p / F_STATE).exists())

    def _migrate_flat(self) -> None:
        for old, new in _LEGACY.items():
            src = self.dir / old
            if src.exists() and not (self.dir / new).exists():
                (self.dir / new).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(self.dir / new))

    def path(self, name: str) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def sub(self, name: str) -> Path:
        d = self.dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def exists(self, name: str) -> bool:
        return (self.dir / name).exists()

    @property
    def name(self) -> str:
        return self.dir.name

    def state(self) -> dict[str, Any]:
        return read_json(self.dir / F_STATE, {}) or {}

    def save_state(self, st: dict[str, Any]) -> None:
        write_json(self.dir / F_STATE, st)

    def update_state(self, **kw: Any) -> dict[str, Any]:
        st = self.state()
        st.update(kw)
        self.save_state(st)
        return st

    @property
    def plan(self) -> dict[str, Any]:
        return self.state().get("plan", {})

    @property
    def spec(self) -> dict[str, Any]:
        return read_json(self.dir / F_SPEC, {}) or {}

    def adapters_dir(self) -> Path:
        return self.sub(D_TRAIN) / "adapters"

    def data_dir(self) -> Path:
        return self.sub(D_DATA) / "train_data"

    def size_of(self, sub: str) -> int:
        d = self.dir / sub
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0

    # ------------------------------------------------------------------ 정리(휴지통)
    def reset_from(self, stage: str) -> list[str]:
        """stage: 'spec'(데이터 명세부터) | 'generated'(생성 데이터부터) | 'training'(학습부터).
        해당 단계 이후의 산출물을 휴지통으로 보내고(복구 가능) 상태를 되돌린다."""
        targets: list[Path] = []
        if stage == "spec":
            targets += [self.dir / F_SPEC]
        if stage in ("spec", "generated"):
            targets += [self.dir / D_GEN]
        targets += [self.dir / D_DATA, self.dir / D_TRAIN, self.dir / F_EVAL, self.dir / F_PROV]
        trashed = []
        for t in targets:
            if t.exists():
                paths.move_to_trash(t)
                trashed.append(str(t))
        st = self.state()
        st.pop("train", None)
        if stage in ("spec", "generated"):
            st.pop("baseline", None)
            st.pop("eval_ids", None)
        if st.get("plan"):
            st["status"] = "planned"
        self.save_state(st)
        return trashed

    def trash(self) -> None:
        """프로젝트 전체를 휴지통으로."""
        paths.move_to_trash(self.dir)
