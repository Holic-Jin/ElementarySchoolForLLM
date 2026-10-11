"""프롬프트 탭: 역할별 프롬프트를 열람·수정(사용자 수정본은 ~/.ElementarySchoolForLLM/prompts 에 저장)."""
from __future__ import annotations

from ..i18n import T

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from ..core import paths
from ..core.roles import PROMPT_FILES, load_prompt


class PromptsEditor(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        info = QLabel(T('역할별 프롬프트입니다. {{이름}} 형태의 자리표시자는 그대로 두세요. JSON 출력 형식을 바꾸면 파이프라인이 응답을 읽지 못할 수 있습니다.'))
        info.setWordWrap(True)
        lay.addWidget(info)
        split = QSplitter()
        self.list = QListWidget()
        self.keys = list(PROMPT_FILES.keys())
        for k in self.keys:
            self.list.addItem(PROMPT_FILES[k])
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Menlo", 12))
        split.addWidget(self.list)
        split.addWidget(self.editor)
        split.setSizes([260, 800])
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        self.state = QLabel("")
        save = QPushButton(T('저장'))
        reset = QPushButton(T('기본값으로 되돌리기'))
        row.addWidget(self.state, 1)
        row.addWidget(reset)
        row.addWidget(save)
        lay.addLayout(row)
        self.list.currentRowChanged.connect(self._load)
        save.clicked.connect(self._save)
        reset.clicked.connect(self._reset)
        self.list.setCurrentRow(0)

    def _key(self) -> str:
        return self.keys[max(0, self.list.currentRow())]

    def _load(self, _row: int = 0) -> None:
        k = self._key()
        self.editor.setPlainText(load_prompt(k))
        modified = (paths.USER_PROMPTS_DIR / f"{k}.md").exists()
        self.state.setText(T('사용자 수정본 사용 중') if modified else T('기본 프롬프트'))

    def _save(self) -> None:
        paths.ensure_dirs()
        (paths.USER_PROMPTS_DIR / f"{self._key()}.md").write_text(self.editor.toPlainText(), encoding="utf-8")
        self.state.setText(T('✅ 저장됨 (사용자 수정본 사용 중)'))

    def _reset(self) -> None:
        f = paths.USER_PROMPTS_DIR / f"{self._key()}.md"
        if f.exists() and QMessageBox.question(self, T('되돌리기'), T('수정본을 지우고 기본 프롬프트로 되돌릴까요?')) == QMessageBox.Yes:
            f.unlink()
        self._load()
