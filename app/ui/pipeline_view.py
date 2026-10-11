"""파이프라인 탭: 전체·단계별 진행률(ETA) + 로그 뷰 + 시작/일시정지/취소."""
from __future__ import annotations

from ..i18n import T

import datetime as _dt
import time
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from ..core.cancel import Cancelled, CancelToken
from ..core.config import ROLES, AppConfig, all_secrets
from ..core.events import STAGES, mask_secret
from ..core.hardware import format_report_table
from ..core.activity import GRAY, GREEN, ORANGE, LightState
from ..core.pipeline import PIPELINE_ACTIVITIES, Pipeline, SpecInsufficientError, pipeline_activity
from ..core.project import Project
from . import strings as S
from .async_runner import AskBridge, QtReporter
from .dialogs import ensure_tos, spec_error
from .status_light import StatusLight

LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
FILTERS = [(T('전체 (서브프로세스 출력 포함)'), 0), (T('정보 이상'), 1), (T('경고 이상'), 2), (T('오류만'), 3)]


def fmt_eta(sec: float | None) -> str:
    if sec is None or sec <= 0 or sec > 7 * 24 * 3600:
        return ""
    sec = int(sec)
    if sec < 60:
        return T('약 {sec}초 남음', sec=sec)
    if sec < 3600:
        return T('약 {v}분 남음', v=sec // 60)
    return T('약 {v}시간 {v2}분 남음', v=sec // 3600, v2=sec % 3600 // 60)


class PipelineWorker(QThread):
    finished_ok = Signal()
    failed = Signal(str)
    cancelled = Signal(str)
    spec_failed = Signal(str)

    def __init__(self, project: Project, cfg: AppConfig, reporter: QtReporter, cancel: CancelToken, ask: AskBridge):
        super().__init__()
        self.project, self.cfg, self.reporter, self.cancel, self.ask = project, cfg, reporter, cancel, ask

    def run(self) -> None:
        import asyncio

        try:
            asyncio.run(Pipeline(self.project, self.cfg, self.reporter, self.cancel, self.ask).run())
        except SpecInsufficientError as e:
            self.spec_failed.emit(format_report_table(e.report))
        except Cancelled as e:
            self.cancelled.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.reporter.log("DEBUG", traceback.format_exc())
            self.failed.emit(str(e) or e.__class__.__name__)
        else:
            self.finished_ok.emit()


# 단계 → 그 단계에서 실제로 일하는 작업(신호등)
STAGE_ACTIVITY = {"spec": "orchestrator", "teacher1": "teacher1", "teacher2": "teacher2", "judge": "orchestrator",
                  "download": "download", "baseline": "student", "lr_sweep": "train", "train": "train",
                  "eval": "student"}


class StageRow:
    def __init__(self, grid: QGridLayout, row: int, name: str, sid: str = ""):
        self.act = pipeline_activity(STAGE_ACTIVITY[sid]) if sid in STAGE_ACTIVITY else None
        self.start: float | None = None
        self.start_done = 0
        self.frac = 0.0
        self.light = StatusLight(evaluator=self.light_state, show_text=False)
        self.light.activity = self.act
        self.label = QLabel(name)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.info = QLabel("")
        self.info.setMinimumWidth(320)
        self.eta = QLabel("")
        self.eta.setMinimumWidth(110)
        for c, w in enumerate((self.light, self.label, self.bar, self.info, self.eta)):
            grid.addWidget(w, row, c)

    def light_state(self) -> LightState:
        if self.start is not None and self.frac >= 1:
            return LightState(GREEN, T('완료'))
        if self.start is None:
            return LightState(GRAY, T("대기"))
        if self.act is None:
            return LightState(ORANGE, T("진행 중"))
        if self.act.inflight > 0 or self.act.error:
            return self.act.evaluate()
        return LightState(GRAY, T("다른 단계의 결과를 기다리는 중"))

    def reset(self) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.info.setText("")
        self.eta.setText("")
        self.start = None
        self.frac = 0.0

    def update(self, done: int, total: int, info: str) -> None:
        total = max(1, total)
        done = max(0, min(done, total))
        now = time.time()
        if self.start is None or done < self.start_done:
            self.start, self.start_done = now, done
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.info.setText(info)
        self.frac = done / total
        elapsed = now - self.start
        progressed = done - self.start_done
        if done >= total:
            self.eta.setText(T('완료'))
        elif progressed > 0 and elapsed > 2:
            self.eta.setText(fmt_eta((total - done) * elapsed / progressed))


class PipelineView(QWidget):
    log_line = Signal(str, str)
    run_finished = Signal()

    def __init__(self, cfg: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self.project: Project | None = None
        self.worker: PipelineWorker | None = None
        self.cancel: CancelToken | None = None
        self.entries: list[tuple[int, str]] = []
        self.started_at: float | None = None

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.proj_lbl = QLabel(T('프로젝트: -'))
        self.start_btn = QPushButton(T('▶ 시작'))
        self.pause_btn = QPushButton(T('⏸ 일시정지'))
        self.cancel_btn = QPushButton(T('■ 취소'))
        self.pause_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        top.addWidget(self.proj_lbl, 1)
        for b in (self.start_btn, self.pause_btn, self.cancel_btn):
            top.addWidget(b)
        lay.addLayout(top)

        ov = QHBoxLayout()
        ov.addWidget(QLabel(T('전체 진행률')))
        self.overall = QProgressBar()
        self.overall.setRange(0, 1000)
        self.overall_eta = QLabel("")
        self.overall_eta.setMinimumWidth(160)
        ov.addWidget(self.overall, 1)
        ov.addWidget(self.overall_eta)
        lay.addLayout(ov)

        lights = QGroupBox(T("작업 신호등 — 🟠 진행 중 · 🔴 멈춤/헛돎 · 🟢 완료"))
        lg = QGridLayout(lights)
        for i, key in enumerate(PIPELINE_ACTIVITIES):
            act = pipeline_activity(key)
            lg.addWidget(StatusLight(act, title=act.label), i // 2, i % 2)
        lay.addWidget(lights)

        box = QGroupBox(T('단계별 진행'))
        grid = QGridLayout(box)
        grid.setColumnStretch(2, 1)
        self.rows = {sid: StageRow(grid, i, name, sid) for i, (sid, name, _) in enumerate(STAGES)}
        self.weights = {sid: w for sid, _, w in STAGES}
        lay.addWidget(box)

        lh = QHBoxLayout()
        lh.addWidget(QLabel(T('로그')))
        self.filter = QComboBox()
        for name, lvl in FILTERS:
            self.filter.addItem(name, lvl)
        self.filter.setCurrentIndex(1)
        self.autoscroll = QCheckBox(T('자동 스크롤'))
        self.autoscroll.setChecked(True)
        save_btn = QPushButton(T('로그 저장…'))
        clear_btn = QPushButton(T('지우기'))
        lh.addWidget(self.filter)
        lh.addWidget(self.autoscroll)
        lh.addStretch(1)
        lh.addWidget(save_btn)
        lh.addWidget(clear_btn)
        lay.addLayout(lh)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        self.log_view.setFont(QFont("Menlo", 11))
        lay.addWidget(self.log_view, 1)

        self.reporter = QtReporter(self)
        self.reporter.log_signal.connect(self.append_log)
        self.reporter.stage_signal.connect(self.on_stage)
        self.ask = AskBridge(self)
        self.ask.ask.connect(self._on_ask)
        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.cancel_btn.clicked.connect(self.cancel_run)
        self.filter.currentIndexChanged.connect(self._refilter)
        save_btn.clicked.connect(self._save_log)
        clear_btn.clicked.connect(lambda: (self.entries.clear(), self.log_view.clear()))

    # ------------------------------------------------------------------ 상태
    def set_project(self, project: Project | None) -> None:
        if self.running:
            return
        self.project = project
        self.proj_lbl.setText(T('프로젝트: {v}', v=project.name if project else '-'))

    @property
    def running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ------------------------------------------------------------------ 로그
    def append_log(self, level: str, msg: str) -> None:
        msg = mask_secret(msg, all_secrets(self.cfg))
        lvl = LEVELS.get(level, 1)
        line = f"{_dt.datetime.now():%H:%M:%S} [{level}] {msg}"
        self.entries.append((lvl, line))
        if len(self.entries) > 50000:
            self.entries = self.entries[-40000:]
        if lvl >= self.filter.currentData():
            self.log_view.appendPlainText(line)
            if self.autoscroll.isChecked():
                self.log_view.moveCursor(QTextCursor.End)
        if lvl >= 1:
            self.log_line.emit(level, msg)

    def _refilter(self) -> None:
        th = self.filter.currentData()
        self.log_view.setPlainText("\n".join(l for lv, l in self.entries if lv >= th))
        self.log_view.moveCursor(QTextCursor.End)

    def _save_log(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, T('로그 저장'), "pipeline_log.txt", "Text (*.txt)")
        if f:
            with open(f, "w", encoding="utf-8") as fp:
                fp.write("\n".join(l for _, l in self.entries))

    # ------------------------------------------------------------------ 진행률
    def on_stage(self, sid: str, done: int, total: int, info: str) -> None:
        row = self.rows.get(sid)
        if not row:
            return
        row.update(done, total, info)
        tot_w = sum(self.weights.values())
        frac = sum(self.rows[s].frac * w for s, w in self.weights.items()) / tot_w
        self.overall.setValue(int(frac * 1000))
        if self.started_at and frac > 0.02:
            el = time.time() - self.started_at
            self.overall_eta.setText(fmt_eta(el * (1 - frac) / frac))

    # ------------------------------------------------------------------ 실행 제어
    def start(self) -> None:
        if self.running:
            return
        if not self.project:
            QMessageBox.warning(self, T('프로젝트 없음'), T('[학습방안] 탭에서 프로젝트를 만들고 방안을 확정하세요.'))
            return
        if not self.project.plan:
            QMessageBox.warning(self, T('학습방안 미확정'), T('[학습방안] 탭에서 오케스트레이터와 대화한 뒤 [방안 확정]을 누르세요.'))
            return
        if not ensure_tos(self, self.cfg, list(ROLES)):
            return
        warns = [S.COST_NOTE]
        t1, t2, o = (self.cfg.roles[r] for r in ("teacher1", "teacher2", "orchestrator"))
        if t1.model == t2.model and t1.base_url == t2.base_url:
            warns.append(S.SAME_MODEL_TEACHERS)
        if o.base_url in (t1.base_url, t2.base_url) and o.model in (t1.model, t2.model):
            warns.append(S.SAME_MODEL_ORCH)
        n = self.project.plan.get("target_samples", 0)
        msg = T('목표 {n}문항으로 파이프라인을 시작할까요?\n\n', n=n) + "\n\n".join(warns)
        if QMessageBox.question(self, T('파이프라인 시작'), msg) != QMessageBox.Yes:
            return
        for r in self.rows.values():
            r.reset()
        self.overall.setValue(0)
        self.overall_eta.setText("")
        self.started_at = time.time()
        self.cancel = CancelToken()
        self.worker = PipelineWorker(self.project, self.cfg, self.reporter, self.cancel, self.ask)
        self.worker.finished_ok.connect(self._on_ok)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.spec_failed.connect(self._on_spec_failed)
        self.worker.finished.connect(self._on_thread_end)
        self._set_running(True)
        self.append_log("INFO", T('파이프라인 시작: {name}', name=self.project.name))
        self.worker.start()

    def toggle_pause(self) -> None:
        if not self.cancel:
            return
        if self.cancel.paused:
            self.cancel.resume()
            self.pause_btn.setText(T('⏸ 일시정지'))
            self.append_log("INFO", T('재개'))
        else:
            self.cancel.pause()
            self.pause_btn.setText(T('▶ 재개'))
            self.append_log("INFO", T('일시정지 — 진행 중인 요청이 끝나면 멈춥니다 (학습 중이면 학습 단계가 끝난 뒤 멈춤)'))

    def cancel_run(self) -> None:
        if self.cancel and self.running:
            if QMessageBox.question(self, T('취소'), T('진행 중인 모든 작업을 취소할까요? (생성된 데이터는 남아 다음에 이어서 진행됩니다)')) == QMessageBox.Yes:
                self.cancel.cancel(T('사용자 취소'))
                self.append_log("WARN", T('취소 요청 — 진행 중인 요청과 하위 프로세스를 종료합니다.'))

    def _set_running(self, on: bool) -> None:
        self.start_btn.setEnabled(not on)
        self.pause_btn.setEnabled(on)
        self.cancel_btn.setEnabled(on)
        self.pause_btn.setText(T('⏸ 일시정지'))

    def _on_ok(self) -> None:
        self.append_log("INFO", T('✅ 파이프라인 완료'))
        QMessageBox.information(self, T('완료'), T('학습이 끝났습니다. [결과] 탭에서 학습 전후 비교와 테스트 채팅을 확인하세요.'))

    def _on_failed(self, msg: str) -> None:
        self.append_log("ERROR", msg)
        QMessageBox.critical(self, T('오류로 중단'), msg)

    def _on_cancelled(self, msg: str) -> None:
        self.append_log("WARN", T('취소됨: {msg}', msg=msg))

    def _on_spec_failed(self, table: str) -> None:
        self.append_log("ERROR", T('컴퓨터 사양 부족으로 모든 작업을 취소했습니다.'))
        spec_error(self, table)

    def _on_thread_end(self) -> None:
        self._set_running(False)
        self.run_finished.emit()

    def _on_ask(self, title: str, message: str, holder: dict) -> None:
        holder["answer"] = QMessageBox.question(self, title, message) == QMessageBox.Yes
        holder["event"].set()

    def shutdown(self) -> None:
        if self.running and self.cancel:
            self.cancel.cancel(T('앱 종료'))
            self.worker.wait(15000)
