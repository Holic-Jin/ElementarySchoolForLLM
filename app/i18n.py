"""UI 다국어 지원 (한국어 · English · 日本語).

- 원문(키)은 한국어 문자열이고, app/locales/en.py · ja.py 에 번역을 둔다.
- 언어는 설정(ui_language) → macOS 시스템 언어 순으로 정하며, 바꾸면 앱을 다시 시작할 때 적용된다.
- 모듈 수준 상수도 올바른 언어가 되도록, 이 모듈은 가져오는 순간 언어를 결정한다.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

LANGUAGES = {"ko": "한국어", "en": "English", "ja": "日本語"}
_lang = "ko"
_catalog: dict[str, str] = {}


def _system_language() -> str:
    candidates = []
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True, timeout=3).stdout
        candidates += [x.strip().strip('",()') for x in out.split() if x.strip().strip('",()')]
    except Exception:
        pass
    candidates.append(os.environ.get("LANG", ""))
    for c in candidates:
        code = c.lower()[:2]
        if code in LANGUAGES:
            return code
    return "en"


def _configured_language() -> str:
    if os.environ.get("ESLLM_LANG") in LANGUAGES:
        return os.environ["ESLLM_LANG"]
    home = Path(os.environ.get("ESLLM_HOME", Path.home() / ".ElementarySchoolForLLM"))
    try:
        lang = json.loads((home / "config.json").read_text(encoding="utf-8")).get("ui_language", "")
        if lang in LANGUAGES:
            return lang
    except Exception:
        pass
    return _system_language()


def set_language(code: str) -> None:
    global _lang, _catalog
    _lang = code if code in LANGUAGES else "ko"
    _catalog = {}
    if _lang == "en":
        from .locales.en import MESSAGES
        _catalog = MESSAGES
    elif _lang == "ja":
        from .locales.ja import MESSAGES
        _catalog = MESSAGES


def language() -> str:
    return _lang


def T(text: str, **kw) -> str:
    """번역. 번역이 없으면 원문(한국어)을 쓴다. kw가 있으면 str.format으로 채운다."""
    out = _catalog.get(text, text) if _catalog else text
    if kw:
        try:
            return out.format(**kw)
        except (KeyError, IndexError, ValueError):
            return text.format(**kw)
    return out


set_language(_configured_language())
