"""결과 탭: 산출물 파일, 품질 통계, 학습 전후 비교, 학생 테스트 채팅, Ollama 내보내기."""
from __future__ import annotations

from ..i18n import T

import html
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
                               QMessageBox, QToolButton,
                               QPushButton, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..core.cancel import CancelToken
from ..core.config import AppConfig
from ..core.exporter import export_to_ollama
from ..core.jsonutil import read_json
from ..core.local_backends import StudentServer
from ..core.project import D_TRAIN, F_EVAL, F_HW, F_QUALITY, SUBDIRS, Project
from ..core.providers import OpenAICompatibleProvider, ProviderConfig
from ..core.trainers.base import local_model_dir
from ..core.trainers.registry import get_engine
from ..core.activity import HUB
from .async_runner import QtReporter, run_async
from .status_light import StatusLight


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class ResultsView(QWidget):
    project_deleted = Signal()

    def __init__(self, cfg: AppConfig, reporter: QtReporter, parent: QWidget | None = None):
        super().__init__(parent)
        self.is_busy = lambda: False
        self.cfg = cfg
        self.reporter = reporter
        self.project: Project | None = None
        self.server: StudentServer | None = None
        self.chat_history: list[dict[str, str]] = []

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel(T('프로젝트: -'))
        refresh = QPushButton(T('새로고침'))
        open_dir = QPushButton(T('폴더 열기'))
        self.export_btn = QPushButton(T('Ollama로 내보내기…'))
        top.addWidget(self.title, 1)
        top.addWidget(refresh)
        top.addWidget(open_dir)
        top.addWidget(self.export_btn)
        self.clean_btn = QToolButton()
        self.clean_btn.setText(T("정리…"))
        self.clean_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.clean_btn)
        for stage, label in (("spec", T("데이터 명세부터 다시 만들기")), ("generated", T("생성 데이터부터 다시 만들기")),
                             ("training", T("학습만 다시 하기"))):
            menu.addAction(label, lambda st=stage, lb=label: self._reset(st, lb))
        menu.addSeparator()
        menu.addAction(T("프로젝트 전체를 휴지통으로"), self._trash_project)
        self.clean_btn.setMenu(menu)
        top.addWidget(self.clean_btn)
        lay.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.files = QListWidget()
        self.files.setToolTip(T('더블클릭하면 기본 앱으로 엽니다'))
        split.addWidget(self.files)
        tabs = QTabWidget()
        self.stats = QTextBrowser()
        self.evalv = QTextBrowser()
        tabs.addTab(self.stats, T('품질 통계'))
        tabs.addTab(self.evalv, T('학습 전후 비교'))
        tabs.addTab(self._chat_widget(), T('학생 테스트 채팅'))
        split.addWidget(tabs)
        split.setSizes([260, 800])
        lay.addWidget(split, 1)

        refresh.clicked.connect(self.refresh)
        open_dir.clicked.connect(lambda: self.project and QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.dir))))
        self.files.itemDoubleClicked.connect(
            lambda it: it.data(Qt.UserRole) and QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.UserRole))))
        self.export_btn.clicked.connect(self._export)

    def _chat_widget(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        r = QHBoxLayout()
        self.use_adapter = QCheckBox(T('학습 결과 적용'))
        self.use_adapter.setChecked(True)
        self.srv_btn = QPushButton(T('학생 서버 시작'))
        self.srv_status = QLabel("")
        r.addWidget(self.use_adapter)
        r.addWidget(self.srv_btn)
        r.addWidget(self.srv_status, 1)
        l.addLayout(r)
        self.activity = HUB.get("results:student", T("학생 모델"), first_grace=180.0, stall_after=60.0)
        l.addWidget(StatusLight(self.activity, title=T("학생 모델")))
        self.chat_view = QTextBrowser()
        l.addWidget(self.chat_view, 1)
        r2 = QHBoxLayout()
        self.chat_in = QLineEdit()
        self.chat_in.setPlaceholderText(T('학생에게 질문하기'))
        send = QPushButton(T('보내기'))
        r2.addWidget(self.chat_in, 1)
        r2.addWidget(send)
        l.addLayout(r2)
        self.srv_btn.clicked.connect(self._toggle_server)
        send.clicked.connect(self._chat_send)
        self.chat_in.returnPressed.connect(self._chat_send)
        return w

    # ------------------------------------------------------------------ 표시
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        p = self.project
        self.title.setText(T('프로젝트: {v}', v=p.name if p else '-'))
        self.files.clear()
        if not p:
            return
        for sub in SUBDIRS:
            d = p.dir / sub
            if not d.exists():
                continue
            head = QListWidgetItem(f"{sub}/  ({_human(p.size_of(sub))})")
            head.setData(Qt.UserRole, str(d))
            f = head.font()
            f.setBold(True)
            head.setFont(f)
            self.files.addItem(head)
            for c in sorted(d.iterdir()):
                if c.name.startswith("."):
                    continue
                size = sum(x.stat().st_size for x in c.rglob("*") if x.is_file()) if c.is_dir() else c.stat().st_size
                it = QListWidgetItem(f"    {c.name}{'/' if c.is_dir() else ''}  ({_human(size)})")
                it.setData(Qt.UserRole, str(c))
                self.files.addItem(it)
        self._render_stats()
        self._render_eval()
        self.export_btn.setEnabled(bool(p.state().get("train")))

    def _render_stats(self) -> None:
        q = read_json(self.project.path(F_QUALITY), None)
        st = self.project.state()
        hw = read_json(self.project.path(F_HW), None)
        h = [f"<h3>{html.escape(self.project.plan.get('title', self.project.name))}</h3>",
             T('<p>상태: <b>{get}</b></p>', get=st.get('status', '-'))]
        if q:
            h.append(T('<h4>데이터 품질</h4><table border=0 cellpadding=3>'))
            rows = [(T('전체 질문'), q["total_questions"]), (T('평가 완료'), q["judged"]), (T('채택'), q["accepted"]),
                    (T('통과율'), f"{q['pass_rate'] * 100:.0f}%"),
                    (T('채택 답변 출처'), T('교사1 {chosen_teacher1} / 교사2 {chosen_teacher2}', chosen_teacher1=q['chosen_teacher1'], chosen_teacher2=q['chosen_teacher2'])),
                    (T('평균 점수(채택분)'), T('교사1 {avg_score_teacher1} / 교사2 {avg_score_teacher2}', avg_score_teacher1=q['avg_score_teacher1'], avg_score_teacher2=q['avg_score_teacher2'])),
                    (T('두 답변 모순'), q["contradictions"]), (T('선호도 쌍'), q["preference_pairs"]),
                    (T('판정 분포'), ", ".join(f"{k} {v}" for k, v in q["status_counts"].items()))]
            h += [f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows]
            h.append(T('</table><h4>카테고리별 (채택/목표 · 난이도 분포)</h4><table border=0 cellpadding=3>'))
            for c, v in q["per_category"].items():
                d = v.get("difficulty", {})
                h.append(T('<tr><td>{escape}</td><td>{accepted}/{target}</td><td>쉬움 {get} · 보통 {get2} · 어려움 {get3}</td></tr>', escape=html.escape(c), accepted=v['accepted'], target=v['target'], get=d.get('easy', 0), get2=d.get('medium', 0), get3=d.get('hard', 0)))
            h.append("</table>")
        tr = st.get("train")
        if tr:
            c = tr.get("config", {})
            h.append(T('<h4>학습</h4>'))
            h.append(T('<p>엔진 {engine} · 모드 {get} · rank {get2} · LR {get3:.2e} · 배치 {get4} · {get5} step<br>최종 학습 손실 {get6} · 검증 손실 {get7}</p>', engine=tr['engine'], get=c.get('mode'), get2=c.get('rank'), get3=c.get('learning_rate', 0), get4=c.get('batch_size'), get5=c.get('iters'), get6=tr.get('final_train_loss'), get7=tr.get('final_val_loss')))
        if hw:
            peak = hw.get("measured_peak_gb")
            est = next((e["total_gb"] for e in hw.get("estimates", []) if tr and e["mode"] == tr.get("config", {}).get("mode")), None)
            h.append(T('<h4>사양</h4><p>GPU 할당 가능 {get}GB · 가능 모드 {join}', get=hw.get('available_gb'), join=', '.join(hw.get('feasible_modes', [])))
                     + (T('<br>메모리 추정(배치 1) {est}GB → 실제 최대 {peak:.2f}GB', est=est, peak=peak) if peak else "") + "</p>")
        self.stats.setHtml("".join(h) if (q or tr) else T("<p style='color:gray'>아직 결과가 없습니다.</p>"))

    def _render_eval(self) -> None:
        ev = read_json(self.project.path(F_EVAL), None)
        if not ev:
            self.evalv.setHtml(T("<p style='color:gray'>학습이 끝나면 학습 전후 비교가 표시됩니다.</p>"))
            return
        h = [T('<h3>평균 점수 {avg_before} → <b>{avg_after}</b> (10점 만점, {n}문항)</h3>', avg_before=ev['avg_before'], avg_after=ev['avg_after'], n=ev['n'])]
        for it in ev["items"]:
            h.append(T("<hr><p><b>Q.</b> {escape}</p><p><b>학습 전</b> ({before_score}): {escape2}</p><p><b>학습 후</b> ({after_score}): {escape3}</p><p style='color:gray'>{escape4}</p>", escape=html.escape(it['question']), before_score=it['before_score'], escape2=html.escape(it['before'][:800]), after_score=it['after_score'], escape3=html.escape(it['after'][:800]), escape4=html.escape(it.get('reason', ''))))
        self.evalv.setHtml("".join(h))

    # ------------------------------------------------------------------ 학생 테스트 채팅
    def _toggle_server(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None
            self.srv_btn.setText(T('학생 서버 시작'))
            self.srv_status.setText(T('중지됨'))
            return
        if not self.project:
            return
        tr = self.project.state().get("train", {})
        eng = get_engine(tr.get("engine") or (self.cfg.student.engine if self.cfg.student.engine != "auto" else "mlx"))
        model_path = Path(tr["model_path"]) if tr.get("model_path") else local_model_dir(self.cfg.student.model_id)
        adapter = Path(tr["adapter_path"]) if (tr.get("adapter_path") and self.use_adapter.isChecked()) else None
        if not (model_path / "config.json").exists():
            QMessageBox.warning(self, T('학생 모델 없음'), T('학생 모델이 아직 내려받아지지 않았습니다. 파이프라인을 먼저 실행하세요.'))
            return
        srv = StudentServer(lambda port: eng.serve_command(model_path, adapter, port), cancel=CancelToken(),
                            model_name=eng.serve_model_name, activity=self.activity)
        self.srv_status.setText(T('시작 중… (모델 로딩)'))
        self.srv_btn.setEnabled(False)

        def done(_):
            self.server = srv
            self.srv_btn.setEnabled(True)
            self.srv_btn.setText(T('학생 서버 중지'))
            self.srv_status.setText(f"✅ {srv.base_url} " + (T('(학습 결과 적용)') if adapter else T('(학습 전 베이스)')))
            self.chat_history = []

        def fail(e):
            srv.stop()
            self.srv_btn.setEnabled(True)
            self.srv_status.setText(f"❌ {e[:160]}")

        run_async(srv.start, done, fail)

    def _chat_send(self) -> None:
        text = self.chat_in.text().strip()
        if not text or not self.server:
            if not self.server:
                self.srv_status.setText(T('먼저 학생 서버를 시작하세요.'))
            return
        self.chat_in.clear()
        spec = self.project.spec if self.project else {}
        system = spec.get("system_prompt", "")
        self.chat_history.append({"role": "user", "content": text})
        msgs = ([{"role": "system", "content": system}] if system else []) + list(self.chat_history)
        srv = self.server
        self._render_chat()

        async def go():
            p = OpenAICompatibleProvider(ProviderConfig(kind="openai", base_url=srv.base_url, model=srv.model_name,
                                                        temperature=0.3, max_tokens=1024, concurrency=1, retries=0),
                                         activity=self.activity)
            try:
                return await p.chat(msgs)
            finally:
                await p.aclose()

        def done(ans):
            self.chat_history.append({"role": "assistant", "content": ans})
            self._render_chat()

        run_async(go, done, lambda e: (self.chat_history.append({"role": "assistant", "content": T('(오류: {e})', e=e)}),
                                       self._render_chat()))

    def _render_chat(self) -> None:
        parts = []
        for m in self.chat_history:
            who = T('나') if m["role"] == "user" else T('학생')
            parts.append(f"<p><b>{who}</b><br>{html.escape(m['content']).replace(chr(10), '<br>')}</p>")
        self.chat_view.setHtml("".join(parts))
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    # ------------------------------------------------------------------ 내보내기
    def _export(self) -> None:
        if not self.project:
            return
        tr = self.project.state().get("train")
        if not tr:
            return
        name, ok = QInputDialog.getText(self, T('Ollama로 내보내기'), T('Ollama 모델 이름:'),
                                        text=self.project.name.lower().replace("_", "-") + "-student")
        if not ok or not name.strip():
            return
        self.export_btn.setEnabled(False)
        system = self.project.spec.get("system_prompt", "")
        project = self.project

        def done(n):
            self.export_btn.setEnabled(True)
            QMessageBox.information(self, T('내보내기 완료'), T("Ollama 모델 '{n}' 등록 완료.\n터미널에서: ollama run {n2}", n=n, n2=n))
            self.refresh()

        def fail(e):
            self.export_btn.setEnabled(True)
            QMessageBox.critical(self, T('내보내기 실패'), e)

        run_async(lambda: export_to_ollama(tr["engine"], Path(tr["model_path"]), Path(tr["adapter_path"]), project.sub(D_TRAIN),
                                           name.strip(), system, self.reporter, CancelToken()), done, fail)

    # ------------------------------------------------------------------ 정리(휴지통)
    def _reset(self, stage: str, label: str) -> None:
        if not self.project:
            return
        if self.is_busy():
            QMessageBox.warning(self, T("정리"), T("파이프라인이 실행 중입니다. 먼저 취소하세요."))
            return
        msg = T("'{label}'\n\n이 단계 이후의 파일을 휴지통으로 보냅니다(Finder 휴지통에서 복구 가능). 다음 실행 때 이 단계부터 다시 만듭니다. 계속할까요?",
                label=label)
        if QMessageBox.question(self, T("정리"), msg) != QMessageBox.Yes:
            return
        try:
            trashed = self.project.reset_from(stage)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, T("정리"), str(e))
            return
        self.reporter.log("INFO", T("휴지통으로 보냄: {n}개 항목", n=len(trashed)))
        self.refresh()

    def _trash_project(self) -> None:
        if not self.project:
            return
        if self.is_busy():
            QMessageBox.warning(self, T("정리"), T("파이프라인이 실행 중입니다. 먼저 취소하세요."))
            return
        msg = T("프로젝트 '{name}' 전체를 휴지통으로 보낼까요? (Finder 휴지통에서 복구 가능)", name=self.project.name)
        if QMessageBox.question(self, T("정리"), msg) != QMessageBox.Yes:
            return
        try:
            self.project.trash()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, T("정리"), str(e))
            return
        self.project = None
        self.refresh()
        self.project_deleted.emit()

    def shutdown(self) -> None:
        if self.server:
            self.server.stop()
