"""설정 로드/저장. 비밀정보(API 키, HF 토큰)는 keyring을 통해 macOS Keychain에 저장하고 config.json에는 넣지 않는다."""
from __future__ import annotations

from ..i18n import T

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from . import paths

KEYRING_SERVICE = "ElementarySchoolForLLM"
ROLES = ("orchestrator", "teacher1", "teacher2")
ROLE_NAMES = {"orchestrator": T('오케스트레이터'), "teacher1": T('교사1 (질문+답변1)'), "teacher2": T('교사2 (독립 답변2)'), "student": T('학생')}


@dataclass
class RoleConfig:
    preset: str = "ollama"  # providers.PRESETS 의 ID
    kind: str = "auto"  # auto | openai(Chat Completions) | anthropic(Messages) | gemini(generateContent)
    base_url: str = "http://localhost:11434/v1"
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    concurrency: int = 2
    api_key: str = field(default="", repr=False)  # 메모리에서만 사용, 파일에는 저장하지 않음


@dataclass
class StudentConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"  # HF 저장소 ID 또는 로컬 경로
    engine: str = "mlx"  # mlx | torch | mlxext | auto
    mode: str = "auto"  # auto | lora | qlora | dora | rslora | pissa | full | dpo | orpo
    rank: int = 0  # 0 = 자동
    learning_rate: float = 0.0  # 0 = 자동
    epochs: int = 0  # 0 = 자동
    batch_size: int = 0  # 0 = 자동
    max_seq_len: int = 0  # 0 = 자동
    lr_sweep: bool = True
    baseline_parallel: bool = True
    hf_token: str = field(default="", repr=False)


@dataclass
class PipelineConfig:
    pass_threshold: float = 7.0  # 루브릭 종합 점수(1~10) 기준
    max_regenerations: int = 2
    min_pass_ratio: float = 0.6
    max_topup_rounds: int = 2
    teacher1_batch: int = 5  # 교사1이 한 번에 만드는 문항 수
    eval_holdout: int = 20  # 학습 전후 비교에 쓸 문항 수(학습에서 제외)
    dedup_threshold: float = 0.85


@dataclass
class AppConfig:
    roles: dict[str, RoleConfig] = field(default_factory=lambda: {r: RoleConfig() for r in ROLES})
    student: StudentConfig = field(default_factory=StudentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    tos_acknowledged: dict[str, bool] = field(default_factory=dict)  # base_url → 확인 여부
    ui_language: str = ""  # ko | en | ja (비어 있으면 시스템 언어)
    data_root: str = ""  # 데이터 폴더 (비어 있으면 ~/Documents/ElementarySchoolForLLM)
    last_project: str = ""

    # ------------------------------------------------------------------ 저장/로드
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for r in d["roles"].values():
            r.pop("api_key", None)
        d["student"].pop("hf_token", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppConfig":
        cfg = cls()
        from .providers import LEGACY_PRESETS, PRESETS

        for r in ROLES:
            if r in d.get("roles", {}):
                rc = _fill(RoleConfig(), d["roles"][r])
                rc.preset = LEGACY_PRESETS.get(rc.preset, rc.preset)
                if rc.preset not in PRESETS:
                    rc.preset = "custom"
                cfg.roles[r] = rc
        cfg.student = _fill(StudentConfig(), d.get("student", {}))
        cfg.pipeline = _fill(PipelineConfig(), d.get("pipeline", {}))
        cfg.tos_acknowledged = dict(d.get("tos_acknowledged", {}))
        cfg.last_project = d.get("last_project", "")
        cfg.ui_language = d.get("ui_language", "")
        cfg.data_root = d.get("data_root", "")
        return cfg


def _fill(obj, data: dict):
    for k, v in data.items():
        if hasattr(obj, k) and k not in ("api_key", "hf_token"):
            setattr(obj, k, v)
    return obj


def _keyring():
    try:
        import keyring  # noqa: WPS433

        return keyring
    except Exception:  # pragma: no cover - keyring 미설치 환경
        return None


def get_secret(name: str) -> str:
    kr = _keyring()
    if kr is None:
        return ""
    try:
        return kr.get_password(KEYRING_SERVICE, name) or ""
    except Exception:
        return ""


def set_secret(name: str, value: str) -> None:
    kr = _keyring()
    if kr is None:
        return
    try:
        if value:
            kr.set_password(KEYRING_SERVICE, name, value)
        else:
            try:
                kr.delete_password(KEYRING_SERVICE, name)
            except Exception:
                pass
    except Exception:
        pass


def load_config() -> AppConfig:
    paths.ensure_dirs()
    cfg = AppConfig()
    if paths.CONFIG_FILE.exists():
        try:
            cfg = AppConfig.from_dict(json.loads(paths.CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception:
            cfg = AppConfig()
    for r in ROLES:
        cfg.roles[r].api_key = get_secret(f"api_key:{r}")
    cfg.student.hf_token = get_secret("hf_token")
    return cfg


def save_config(cfg: AppConfig) -> None:
    paths.ensure_dirs()
    paths.CONFIG_FILE.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    for r in ROLES:
        set_secret(f"api_key:{r}", cfg.roles[r].api_key)
    set_secret("hf_token", cfg.student.hf_token)


def all_secrets(cfg: AppConfig) -> list[str]:
    return [cfg.roles[r].api_key for r in ROLES] + [cfg.student.hf_token]
