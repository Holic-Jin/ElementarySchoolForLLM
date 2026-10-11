"""경로 상수.

- 프로그램(코드): 저장소 폴더
- 프로그램 실행 환경·설정·키: ~/.ElementarySchoolForLLM (venv, Python, 설정, 로그, 사용자 프롬프트)
- 데이터(프로그램 외): 데이터 폴더 — 기본 ~/Documents/ElementarySchoolForLLM
    projects/<프로젝트>/1_plan · 2_generated · 3_dataset · 4_training · 5_reports
    models/  (내려받은 학생 모델)
  설정(data_root) 또는 환경변수 ESLLM_DATA 로 바꿀 수 있다.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_HOME = Path(os.environ.get("ESLLM_HOME", Path.home() / ".ElementarySchoolForLLM"))
DEFAULT_PROMPTS_DIR = REPO_ROOT / "app" / "prompts"
USER_PROMPTS_DIR = APP_HOME / "prompts"
CONFIG_FILE = APP_HOME / "config.json"
LOG_DIR = APP_HOME / "logs"
TOOLS_DIR = APP_HOME / "tools"
PYTHON = sys.executable
DEFAULT_DATA_ROOT = Path.home() / "Documents" / "ElementarySchoolForLLM"
LEGACY_WORKSPACE = REPO_ROOT / "workspace"  # 0.1.0 초기 버전의 프로젝트 위치(저장소 안)
LEGACY_MODELS = APP_HOME / "models"


def _configured_root() -> Path:
    if os.environ.get("ESLLM_DATA"):
        return Path(os.environ["ESLLM_DATA"]).expanduser()
    try:
        root = json.loads(CONFIG_FILE.read_text(encoding="utf-8")).get("data_root", "")
        if root:
            return Path(root).expanduser()
    except Exception:
        pass
    return DEFAULT_DATA_ROOT


DATA_ROOT = _configured_root()
WORKSPACE = DATA_ROOT / "projects"
MODELS_DIR = DATA_ROOT / "models"


def set_data_root(root: Path) -> None:
    global DATA_ROOT, WORKSPACE, MODELS_DIR
    DATA_ROOT = Path(root).expanduser()
    WORKSPACE = DATA_ROOT / "projects"
    MODELS_DIR = DATA_ROOT / "models"


def uv_binary() -> str | None:
    """부트스트랩이 설치한 uv(없으면 PATH의 uv)."""
    local = APP_HOME / "bin" / "uv"
    if local.exists():
        return str(local)
    return shutil.which("uv")


def ensure_dirs() -> None:
    for d in (APP_HOME, USER_PROMPTS_DIR, LOG_DIR, TOOLS_DIR, WORKSPACE, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def migrate_legacy() -> list[str]:
    """이전 버전 위치(저장소 안 workspace/, ~/.ElementarySchoolForLLM/models)의 데이터를 데이터 폴더로 옮긴다."""
    moved = []
    if os.environ.get("ESLLM_DATA"):  # 테스트·개발용 임시 데이터 폴더에서는 실제 데이터를 옮기지 않는다
        return moved
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if LEGACY_WORKSPACE.is_dir() and LEGACY_WORKSPACE.resolve() != WORKSPACE.resolve():
        for p in sorted(LEGACY_WORKSPACE.iterdir()):
            if p.is_dir() and (p / "project.json").exists():
                dest = WORKSPACE / p.name
                if dest.exists():
                    dest = WORKSPACE / f"{p.name}_migrated"
                shutil.move(str(p), str(dest))
                moved.append(str(dest))
    if LEGACY_MODELS.is_dir() and LEGACY_MODELS.resolve() != MODELS_DIR.resolve():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for p in sorted(LEGACY_MODELS.iterdir()):
            dest = MODELS_DIR / p.name
            if not dest.exists():
                shutil.move(str(p), str(dest))
                moved.append(str(dest))
    return moved


def move_to_trash(path: Path) -> None:
    """복구할 수 있도록 macOS 휴지통으로 보낸다.
    macOS 15+ 내장 /usr/bin/trash(권한 창 없음, Finder '되돌려 놓기' 가능) → Finder 스크립트 → ~/.Trash 이동 순으로 시도."""
    import subprocess

    path = Path(path)
    if not path.exists():
        return
    if Path("/usr/bin/trash").exists():
        r = subprocess.run(["/usr/bin/trash", str(path)], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and not path.exists():
            return
    script = f'tell application "Finder" to delete (POSIX file "{path}" as alias)'
    try:
        r = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and not path.exists():
            return
    except subprocess.TimeoutExpired:
        pass
    trash = Path.home() / ".Trash"
    dest = trash / path.name
    i = 2
    while dest.exists():
        dest = trash / f"{path.name} {i}"
        i += 1
    shutil.move(str(path), str(dest))
