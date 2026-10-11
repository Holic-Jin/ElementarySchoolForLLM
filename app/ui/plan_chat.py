"""학습방안 탭: 프로젝트 선택 + 오케스트레이터와의 Q&A 채팅 + 학습방안 미리보기."""
from __future__ import annotations

from ..i18n import T

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from ..core.config import AppConfig, save_config
from ..core.planning import finalize, interview_turn, load_chat, save_chat
from ..core.project import F_PLAN, Project
from ..core.activity import HUB
from .async_runner import run_async
from .status_light import StatusLight
from .dialogs import ensure_tos

GREETING = T('학습방안을 함께 정리해 볼까요? 학생 모델이 무엇을 잘하게 되면 좋을지 자유롭게 알려 주세요.')


class PlanChat(QWidget):
    project_changed = Signal(object)  # Project
    plan_finalized = Signal(object)

    def __init__(self, cfg: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self.project: Project | None = None
        self.history: list[dict[str, str]] = []
        self.busy = False

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel(T('프로젝트')))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(280)
        new_btn = QPushButton(T('새 프로젝트…'))
        top.addWidget(self.combo, 1)
        top.addWidget(new_btn)
        lay.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(T('오케스트레이터에게 답하기 (⌘+Enter로 전송)'))
        self.input.setMaximumHeight(110)
        row = QHBoxLayout()
        self.status = QLabel("")
        self.send_btn = QPushButton(T('보내기'))
        self.final_btn = QPushButton(T('방안 확정'))
        self.final_btn.setToolTip(T('지금까지의 대화로 학습방안(01)과 알고리즘 초안(03)을 만듭니다'))
        self.reset_btn = QPushButton(T('대화 초기화'))
        self.activity = HUB.get("chat:orchestrator", T("오케스트레이터"))
        self.light = StatusLight(self.activity)
        row.addWidget(self.light, 1)
        row.addWidget(self.status)
        row.addWidget(self.reset_btn)
        row.addWidget(self.send_btn)
        row.addWidget(self.final_btn)
        ll.addWidget(self.view, 1)
        ll.addWidget(self.input)
        ll.addLayout(row)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel(T('학습방안 미리보기')))
        self.preview = QTextBrowser()
        rl.addWidget(self.preview, 1)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([620, 420])
        lay.addWidget(split, 1)

        new_btn.clicked.connect(self._new_project)
        self.combo.activated.connect(lambda _i: self._open(self.combo.currentText()))
        self.send_btn.clicked.connect(self._send)
        self.final_btn.clicked.connect(self._finalize)
        self.reset_btn.clicked.connect(self._reset)
        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self._send)
        self.refresh_projects(self.cfg.last_project)

    # ------------------------------------------------------------------ 프로젝트
    def refresh_projects(self, select: str = "") -> None:
        names = Project.list_all()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(names)
        self.combo.blockSignals(False)
        if select and select in names:
            self.combo.setCurrentText(select)
            self._open(select)
        elif names:
            self._open(names[-1])
            self.combo.setCurrentText(names[-1])
        else:
            self._set_enabled(False)
            self.view.setHtml(T("<p style='color:gray'>[새 프로젝트…]를 눌러 시작하세요.</p>"))

    def _new_project(self) -> None:
        name, ok = QInputDialog.getText(self, T('새 프로젝트'), T('프로젝트 이름:'))
        if not ok or not name.strip():
            return
        p = Project.create(name.strip())
        self.refresh_projects(p.name)

    def _open(self, name: str) -> None:
        if not name:
            return
        from ..core import paths

        self.project = Project(paths.WORKSPACE / name)
        self.cfg.last_project = name
        save_config(self.cfg)
        self.history = load_chat(self.project)
        self._render()
        if self.project.exists(F_PLAN):
            self.preview.setMarkdown(self.project.path(F_PLAN).read_text(encoding="utf-8"))
        else:
            self.preview.setMarkdown(self.project.state().get("summary", ""))
        self._set_enabled(True)
        self.project_changed.emit(self.project)

    def _set_enabled(self, on: bool) -> None:
        for w in (self.send_btn, self.final_btn, self.reset_btn, self.input):
            w.setEnabled(on and not self.busy)

    # ------------------------------------------------------------------ 채팅
    def _render(self) -> None:
        parts = [T("<p style='color:#2a6'><b>오케스트레이터</b><br>{escape}</p>", escape=html.escape(GREETING))]
        for m in self.history:
            who = T('나') if m["role"] == "user" else T('오케스트레이터')
            color = "#246" if m["role"] == "user" else "#2a6"
            body = html.escape(m["content"]).replace("\n", "<br>")
            parts.append(f"<p style='color:{color}'><b>{who}</b><br><span style='color:palette(text)'>{body}</span></p>")
        self.view.setHtml("".join(parts))
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _busy(self, on: bool, msg: str = "") -> None:
        self.busy = on
        self.status.setText(msg)
        self._set_enabled(self.project is not None)

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self.busy or not self.project:
            return
        if not self.cfg.roles["orchestrator"].model:
            QMessageBox.warning(self, T('설정 필요'), T('설정 탭에서 오케스트레이터 모델을 먼저 지정하고 저장하세요.'))
            return
        if not ensure_tos(self, self.cfg, ["orchestrator"]):
            return
        self.input.clear()
        self.history.append({"role": "user", "content": text})
        save_chat(self.project, self.history)
        self._render()
        self._busy(True, T('오케스트레이터가 생각 중…'))
        hist = list(self.history)

        def done(res):
            body, summary = res
            self.history.append({"role": "assistant", "content": body})
            save_chat(self.project, self.history)
            if summary:
                self.preview.setMarkdown(summary)
                self.project.update_state(summary=summary)
            self._render()
            self._busy(False)

        def fail(e):
            self._busy(False, f"❌ {e[:160]}")

        run_async(lambda: interview_turn(self.cfg, hist, self.activity), done, fail)

    def _finalize(self) -> None:
        if not self.project or self.busy:
            return
        if not self.history:
            QMessageBox.information(self, T('학습방안'), T('먼저 오케스트레이터와 대화를 나눠 주세요.'))
            return
        if not ensure_tos(self, self.cfg, ["orchestrator"]):
            return
        self._busy(True, T('학습방안 확정 중…'))
        hist = list(self.history)
        project = self.project

        def done(plan):
            self.preview.setMarkdown(project.path(F_PLAN).read_text(encoding="utf-8"))
            self._busy(False, T('✅ 학습방안 확정 — [파이프라인] 탭에서 시작하세요.'))
            self.plan_finalized.emit(project)

        run_async(lambda: finalize(self.cfg, project, hist, self.activity), done, lambda e: self._busy(False, f"❌ {e[:160]}"))

    def _reset(self) -> None:
        if not self.project:
            return
        if QMessageBox.question(self, T('대화 초기화'), T('이 프로젝트의 대화 기록을 지울까요? (확정된 학습방안 파일은 남습니다)')) != QMessageBox.Yes:
            return
        self.history = []
        save_chat(self.project, self.history)
        self._render()
