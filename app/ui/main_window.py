"""메인 창: 설정 / 학습방안 / 파이프라인 / 결과 / 프롬프트 탭."""
from __future__ import annotations

from ..i18n import T

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from .. import APP_NAME, VERSION
from ..core.config import AppConfig
from ..core.project import Project
from . import strings as S
from .dialogs import about
from .pipeline_view import PipelineView
from .plan_chat import PlanChat
from .prompts_editor import PromptsEditor
from .results_view import ResultsView
from .settings_panel import SettingsPanel


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1180, 860)
        self.tabs = QTabWidget()
        self.settings = SettingsPanel(cfg)
        self.plan = PlanChat(cfg)
        self.pipeline = PipelineView(cfg)
        self.results = ResultsView(cfg, self.pipeline.reporter)
        self.prompts = PromptsEditor()
        self.tabs.addTab(self.settings, S.TAB_SETTINGS)
        self.tabs.addTab(self.plan, S.TAB_PLAN)
        self.tabs.addTab(self.pipeline, S.TAB_PIPELINE)
        self.tabs.addTab(self.results, S.TAB_RESULTS)
        self.tabs.addTab(self.prompts, S.TAB_PROMPTS)
        self.setCentralWidget(self.tabs)

        self.plan.project_changed.connect(self._on_project)
        self.plan.plan_finalized.connect(lambda p: (self._on_project(p), self.tabs.setCurrentWidget(self.pipeline)))
        self.pipeline.run_finished.connect(self.results.refresh)
        self.results.is_busy = lambda: self.pipeline.running
        self.results.project_deleted.connect(lambda: self.plan.refresh_projects())
        self.pipeline.log_line.connect(lambda lvl, msg: self.statusBar().showMessage(msg[:200], 8000))
        self.tabs.currentChanged.connect(lambda i: self.tabs.widget(i) is self.results and self.results.refresh())

        help_menu = self.menuBar().addMenu(T('도움말'))
        act = QAction(T('{APP_NAME} 정보', APP_NAME=APP_NAME), self)
        act.triggered.connect(lambda: about(self))
        help_menu.addAction(act)
        if self.plan.project:
            self._on_project(self.plan.project)
        if not any(cfg.roles[r].model for r in cfg.roles):
            self.tabs.setCurrentWidget(self.settings)
            self.statusBar().showMessage(T('처음이시군요! 오케스트레이터·교사1·교사2의 주소와 모델명을 설정하고 [설정 저장]을 누르세요.'))
        else:
            self.tabs.setCurrentWidget(self.plan)

    def _on_project(self, project: Project) -> None:
        self.pipeline.set_project(project)
        self.results.set_project(project)
        self.statusBar().showMessage(T('프로젝트: {name}', name=project.name), 5000)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.pipeline.running:
            if QMessageBox.question(self, T('종료'), T('파이프라인이 실행 중입니다. 모든 작업을 취소하고 종료할까요?')) != QMessageBox.Yes:
                event.ignore()
                return
        self.pipeline.shutdown()
        self.results.shutdown()
        self.settings.shutdown()
        event.accept()
