"""설정 탭: 역할별 AI(주소·API 키·모델명), 학생 모델·학습 엔진, 파이프라인 옵션, 로컬 백엔드."""
from __future__ import annotations

from ..i18n import LANGUAGES, T, language

import time
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..core import hardware, paths
from ..core.config import ROLE_NAMES, ROLES, AppConfig, RoleConfig, save_config
from ..core.local_backends import (ensure_ollama, install_llama_cpp, install_ollama, is_ollama_url, ollama_alive,
                                   ollama_binary, ollama_models, ollama_pull, ollama_root, start_llama_server)
from ..core.providers import (KINDS, guess_kind, is_local_url, kind_label, make_provider, normalize_base_url, resolve_kind,
                              provider_config_from_role, tos_url_for)
from ..core.trainers.registry import MODE_LABELS, engines
from . import strings as S
from ..core.activity import HUB
from .async_runner import QtReporter, run_async
from .status_light import StatusLight


class RoleCard(QGroupBox):
    def __init__(self, role: str, parent: QWidget | None = None):
        super().__init__(ROLE_NAMES[role], parent)
        self.role = role
        form = QFormLayout(self)
        self.kind = QComboBox()
        for k in KINDS:
            self.kind.addItem(kind_label(k), k)
        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText(T('예: http://localhost:11434/v1'))
        key_row = QHBoxLayout()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText(T('로컬은 비워 두세요 (Keychain에 안전하게 저장)'))
        show = QPushButton(T('표시'))
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.api_key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        key_row.addWidget(self.api_key)
        key_row.addWidget(show)
        model_row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumWidth(260)
        self.list_btn = QPushButton(T('모델 목록'))
        self.pull_btn = QPushButton(T('모델 받기'))
        self.pull_btn.setToolTip(T('Ollama에 모델이 없으면 내려받습니다'))
        model_row.addWidget(self.model, 1)
        model_row.addWidget(self.list_btn)
        model_row.addWidget(self.pull_btn)
        opt_row = QHBoxLayout()
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 2)
        self.temperature.setSingleStep(0.1)
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(256, 32768)
        self.max_tokens.setSingleStep(256)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 32)
        for lbl, w in ((T('온도'), self.temperature), (T('최대 토큰'), self.max_tokens), (T('동시 요청'), self.concurrency)):
            opt_row.addWidget(QLabel(lbl))
            opt_row.addWidget(w)
        opt_row.addStretch(1)
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton(T('연결 테스트'))
        self.status = QLabel("")
        self.status.setWordWrap(True)
        btn_row.addWidget(self.test_btn)
        btn_row.addWidget(self.status, 1)
        self.activity = HUB.get(f"card:{role}", ROLE_NAMES[role], first_grace=20.0, stall_after=15.0)
        self.light = StatusLight(self.activity)
        self.pull_bar = QProgressBar()
        self.pull_bar.setVisible(False)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setOpenExternalLinks(True)
        self.warn.setStyleSheet("QLabel{background:#fff3cd;color:#664d03;border:1px solid #ffe69c;border-radius:4px;padding:6px;}")

        form.addRow(T('API 형식'), self.kind)
        form.addRow(T('주소'), self.base_url)
        form.addRow(T('API 키'), key_row)
        form.addRow(T('모델명'), model_row)
        form.addRow(T('옵션'), opt_row)
        form.addRow(btn_row)
        form.addRow(T("작업 상태"), self.light)
        form.addRow(self.pull_bar)
        form.addRow(self.warn)

        self.base_url.textChanged.connect(self._update_warn)
        self.test_btn.clicked.connect(self._test)
        self.list_btn.clicked.connect(self._list_models)
        self.pull_btn.clicked.connect(self._pull)

    # ------------------------------------------------------------------ 값 입출력
    def load(self, rc: RoleConfig) -> None:
        self.kind.setCurrentIndex(max(0, self.kind.findData(rc.kind)))
        self.base_url.setText(rc.base_url)
        self.api_key.setText(rc.api_key)
        self.model.setCurrentText(rc.model)
        self.temperature.setValue(rc.temperature)
        self.max_tokens.setValue(rc.max_tokens)
        self.concurrency.setValue(rc.concurrency)
        self._update_warn()

    def apply(self, rc: RoleConfig) -> None:
        self._normalize_fields()
        rc.preset = "custom"
        rc.kind = self.kind.currentData()
        rc.base_url = self.base_url.text().strip()
        rc.api_key = self.api_key.text().strip()
        rc.model = self.model.currentText().strip()
        rc.temperature = self.temperature.value()
        rc.max_tokens = self.max_tokens.value()
        rc.concurrency = self.concurrency.value()

    def _normalize_fields(self) -> None:
        """붙여 넣은 주소에서 엔드포인트를 떼고, 주소에 맞지 않는 API 형식을 바로잡아 화면에도 반영한다."""
        url = self.base_url.text().strip()
        if not url:
            return
        selected = self.kind.currentData()
        guessed = guess_kind(url)
        kind = resolve_kind(selected, url)
        norm = normalize_base_url(kind, url)
        notes = []
        if selected != "auto" and guessed and guessed != selected:
            self.kind.setCurrentIndex(max(0, self.kind.findData(guessed)))
            notes.append(T("API 형식을 '{kind}'(으)로 바꿨습니다", kind=kind_label(guessed)))
        if norm != url:
            self.base_url.setText(norm)
            notes.append(T("주소를 {url}(으)로 정리했습니다", url=norm))
        if notes:
            self.status.setText("ℹ️ " + " · ".join(notes))

    def current(self) -> RoleConfig:
        rc = RoleConfig()
        self.apply(rc)
        return rc

    # ------------------------------------------------------------------ 동작
    def _update_warn(self) -> None:
        url = self.base_url.text().strip()
        remote = bool(url) and not is_local_url(url)
        self.pull_btn.setVisible(is_ollama_url(url))
        if remote:
            tos = tos_url_for(url)
            link = T(' <a href="{tos}">이용약관 열기</a>', tos=tos) if tos else ""
            self.warn.setText(S.TOS_BANNER + link)
        self.warn.setVisible(remote)

    def _test(self) -> None:
        rc = self.current()
        note = self.status.text() if self.status.text().startswith("ℹ️") else ""
        self.status.setText((note + " · " if note else "") + T('테스트 중…'))

        prefix = note + " · " if note else ""
        step = {"n": 1, "t0": time.monotonic()}
        labels = {1: T('1/2 주소·API 키 확인 중'), 2: T('2/2 모델 응답 확인 중')}
        rep = QtReporter(self)
        rep.stage_signal.connect(lambda _s, n, _t, _i: step.update(n=n))
        timer = QTimer(self)

        def tick():
            self.status.setText(prefix + T('{label}… {sec}초', label=labels[step["n"]],
                                           sec=int(time.monotonic() - step["t0"])))

        timer.timeout.connect(tick)
        timer.start(1000)
        tick()
        self.test_btn.setEnabled(False)

        async def go():
            async with make_provider(provider_config_from_role(rc), activity=self.activity) as p:
                return await p.quick_test(lambda n: rep.stage("test", n, 2))

        def done(r):
            timer.stop()
            self.test_btn.setEnabled(True)
            msg = T('✅ 연결 성공 ({sec:.1f}초): {v}', sec=r["seconds"], v=(r["reply"] or "(empty)")[:60])
            if r.get("model_listed") is False:
                msg += " · " + T('⚠ 모델 목록에 이 모델명이 없습니다. [모델 목록]으로 확인하세요.')
            self.status.setText(prefix + msg)

        def fail(e):
            timer.stop()
            self.test_btn.setEnabled(True)
            self.status.setText(T('❌ 실패: {v}', v=e[:300]))

        run_async(go, done, fail)

    def _list_models(self) -> None:
        rc = self.current()
        self.status.setText(T('모델 목록 조회 중…'))

        async def go():
            async with make_provider(provider_config_from_role(rc), require_model=False) as p:
                return await p.list_models()

        def done(models):
            cur = self.model.currentText()
            self.model.clear()
            self.model.addItems(models)
            self.model.setCurrentText(cur or (models[0] if models else ""))
            self.status.setText(T('모델 {len}개', len=len(models)))

        run_async(go, done, lambda e: self.status.setText(T('❌ 목록 조회 실패: {v}', v=e[:200])))

    def _pull(self) -> None:
        name = self.model.currentText().strip()
        if not name:
            self.status.setText(T('받을 모델명을 입력하세요 (예: qwen2.5:7b)'))
            return
        root = ollama_root(self.base_url.text().strip())
        self.pull_bar.setVisible(True)
        self.pull_bar.setRange(0, 0)
        self.status.setText(T('{name} 받는 중…', name=name))
        rep = QtReporter(self)
        rep.stage_signal.connect(lambda _s, d, t, info: (self.pull_bar.setRange(0, max(1, t)), self.pull_bar.setValue(d),
                                                          self.status.setText(info)))

        async def go():
            if not await ensure_ollama(root):
                raise RuntimeError(T('Ollama 서버를 시작할 수 없습니다.'))
            await ollama_pull(name, lambda c, t, s: rep.stage("pull", c, t, f"{s} {c / 1e9:.2f}/{t / 1e9:.2f}GB" if t else s), root,
                              activity=self.activity)

        def done(_):
            self.pull_bar.setVisible(False)
            self.status.setText(T('✅ {name} 준비 완료', name=name))

        def fail(e):
            self.pull_bar.setVisible(False)
            self.status.setText(T('❌ 받기 실패: {v}', v=e[:200]))

        run_async(go, done, fail)


class StudentCard(QGroupBox):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(T('학생 (로컬 LLM)'), parent)
        form = QFormLayout(self)
        row = QHBoxLayout()
        self.model_id = QLineEdit()
        self.model_id.setPlaceholderText(T('Hugging Face 모델 ID (예: Qwen/Qwen2.5-1.5B-Instruct) 또는 로컬 폴더'))
        browse = QPushButton(T('폴더…'))
        browse.clicked.connect(self._browse)
        self.size_btn = QPushButton(T('크기 확인'))
        self.size_btn.clicked.connect(self._check_size)
        row.addWidget(self.model_id, 1)
        row.addWidget(browse)
        row.addWidget(self.size_btn)
        self.size_lbl = QLabel("")
        self.size_lbl.setWordWrap(True)
        self.hf_token = QLineEdit()
        self.hf_token.setEchoMode(QLineEdit.Password)
        self.hf_token.setPlaceholderText(T('승인이 필요한 모델(Llama 등)만 입력 — Keychain에 저장'))

        eng_row = QHBoxLayout()
        self.engine = QComboBox()
        self.engine.addItem(T('자동 (메모리 효율 우선)'), "auto")
        for e in engines().values():
            self.engine.addItem(e.name, e.id)
        self.install_btn = QPushButton(T('엔진 설치'))
        self.install_btn.clicked.connect(self._install_engine)
        eng_row.addWidget(self.engine, 1)
        eng_row.addWidget(self.install_btn)
        self.engine_desc = QLabel("")
        self.engine_desc.setWordWrap(True)
        self.engine_desc.setStyleSheet("color: gray;")
        self.install_bar = QProgressBar()
        self.install_bar.setRange(0, 0)
        self.install_bar.setVisible(False)
        self.mode = QComboBox()

        adv = QGridLayout()
        self.rank = QSpinBox()
        self.rank.setRange(0, 512)
        self.rank.setSpecialValueText(T('자동'))
        self.lr = QLineEdit()
        self.lr.setPlaceholderText(T('자동'))
        self.epochs = QSpinBox()
        self.epochs.setRange(0, 50)
        self.epochs.setSpecialValueText(T('자동'))
        self.batch = QSpinBox()
        self.batch.setRange(0, 64)
        self.batch.setSpecialValueText(T('자동'))
        self.seq = QSpinBox()
        self.seq.setRange(0, 32768)
        self.seq.setSingleStep(256)
        self.seq.setSpecialValueText(T('자동'))
        self.sweep = QCheckBox(T('학습률 탐색 (짧은 학습 3회로 최적 LR 선택)'))
        self.baseline = QCheckBox(T('메모리 여유가 있으면 학습 전 기준 응답을 데이터 생성과 병렬로 생성'))
        for i, (lbl, w) in enumerate((("rank", self.rank), (T('학습률'), self.lr), (T('에폭'), self.epochs),
                                      (T('배치'), self.batch), (T('최대 길이'), self.seq))):
            adv.addWidget(QLabel(lbl), i // 3, (i % 3) * 2)
            adv.addWidget(w, i // 3, (i % 3) * 2 + 1)

        form.addRow(T('모델'), row)
        form.addRow("", self.size_lbl)
        form.addRow(T('HF 토큰'), self.hf_token)
        form.addRow(T('학습 엔진'), eng_row)
        form.addRow("", self.engine_desc)
        form.addRow("", self.install_bar)
        self.install_activity = HUB.get("engine-install", T("엔진 설치"), stall_after=120.0, first_grace=120.0)
        form.addRow(T("설치 상태"), StatusLight(self.install_activity))
        form.addRow(T('학습 방식'), self.mode)
        form.addRow(T('고급'), adv)
        form.addRow("", self.sweep)
        form.addRow("", self.baseline)
        self.engine.currentIndexChanged.connect(self._on_engine)

    def load(self, cfg: AppConfig) -> None:
        s = cfg.student
        self.model_id.setText(s.model_id)
        self.hf_token.setText(s.hf_token)
        self.engine.setCurrentIndex(max(0, self.engine.findData(s.engine)))
        self._on_engine(select_mode=s.mode)
        self.rank.setValue(s.rank)
        self.lr.setText("" if not s.learning_rate else f"{s.learning_rate:g}")
        self.epochs.setValue(s.epochs)
        self.batch.setValue(s.batch_size)
        self.seq.setValue(s.max_seq_len)
        self.sweep.setChecked(s.lr_sweep)
        self.baseline.setChecked(s.baseline_parallel)

    def apply(self, cfg: AppConfig) -> None:
        s = cfg.student
        s.model_id = self.model_id.text().strip()
        s.hf_token = self.hf_token.text().strip()
        s.engine = self.engine.currentData()
        s.mode = self.mode.currentData() or "auto"
        s.rank = self.rank.value()
        try:
            s.learning_rate = float(self.lr.text()) if self.lr.text().strip() else 0.0
        except ValueError:
            s.learning_rate = 0.0
        s.epochs = self.epochs.value()
        s.batch_size = self.batch.value()
        s.max_seq_len = self.seq.value()
        s.lr_sweep = self.sweep.isChecked()
        s.baseline_parallel = self.baseline.isChecked()

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, T('학생 모델 폴더 선택 (config.json 포함)'))
        if d:
            self.model_id.setText(d)

    def _on_engine(self, *_args, select_mode: str | None = None) -> None:
        eid = self.engine.currentData()
        prev = select_mode or self.mode.currentData() or "auto"
        if eid == "auto":
            self.engine_desc.setText(T('원하는 학습 방식을 지원하고 설치된 엔진 중 메모리를 가장 적게 쓰는 엔진을 고릅니다.'))
            self.install_btn.setVisible(False)
            self._fill_modes(["lora", "qlora", "dora", "rslora", "full"], prev)
            return
        eng = engines()[eid]
        installed = eng.is_installed()
        self.engine_desc.setText(eng.description + ("" if installed else T('  — 설치 필요')))
        self.install_btn.setVisible(not installed)

        async def go():
            return await eng.supported_modes()

        if installed:
            run_async(go, lambda modes: self._fill_modes(modes, prev), lambda e: self._fill_modes([], prev))
        else:
            self._fill_modes([], prev)

    def _fill_modes(self, modes: list[str], select: str) -> None:
        self.mode.clear()
        self.mode.addItem(MODE_LABELS["auto"], "auto")
        for m in modes:
            self.mode.addItem(MODE_LABELS.get(m, m), m)
        self.mode.setCurrentIndex(max(0, self.mode.findData(select)))

    def _install_engine(self) -> None:
        eid = self.engine.currentData()
        eng = engines()[eid]
        if QMessageBox.question(self, T('엔진 설치'), T('{name}을(를) 설치할까요?\n\n{description}', name=eng.name, description=eng.description)) != QMessageBox.Yes:
            return
        self.install_bar.setVisible(True)
        self.install_btn.setEnabled(False)
        rep = QtReporter(self)
        rep.log_signal.connect(lambda lvl, msg: self.engine_desc.setText(msg[-160:]))

        def done(_):
            self.install_bar.setVisible(False)
            self.install_btn.setEnabled(True)
            QMessageBox.information(self, T('설치 완료'), T('{name} 설치가 끝났습니다. 반영되지 않으면 앱을 다시 시작하세요.', name=eng.name))
            self._on_engine()

        def fail(e):
            self.install_bar.setVisible(False)
            self.install_btn.setEnabled(True)
            QMessageBox.critical(self, T('설치 실패'), e)

        run_async(lambda: eng.install(rep, activity=self.install_activity), done, fail)

    def _check_size(self) -> None:
        mid, tok = self.model_id.text().strip(), self.hf_token.text().strip()
        self.size_lbl.setText(T('확인 중…'))

        async def go():
            import asyncio

            cfg = await asyncio.to_thread(hardware.load_model_config, mid, tok)
            m = hardware.model_info_from_config(mid, cfg)
            hw = await asyncio.to_thread(hardware.detect)
            return m, hw

        def done(res):
            m, hw = res
            est = hardware.estimate_training_memory(m, "mlx", "lora", 4, 1024, 16)
            self.size_lbl.setText(T('약 {params_b:.2f}B 파라미터 ({model_type}) · 추론 약 {inference_memory_gb:.1f}GB · LoRA 학습(배치4·길이1024) 약 {total_gb:.1f}GB · 이 Mac의 GPU 할당 가능 {gpu_mem_gb:.1f}GB', params_b=m.params_b, model_type=m.model_type, inference_memory_gb=hardware.inference_memory_gb(m), total_gb=est.total_gb, gpu_mem_gb=hw.gpu_mem_gb))

        run_async(go, done, lambda e: self.size_lbl.setText(T('❌ 확인 실패: {v}', v=e[:200])))


class SettingsPanel(QWidget):
    saved = Signal()

    def __init__(self, cfg: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)

        note = QLabel(T('오케스트레이터·교사1·교사2는 <b>주소 + API 키 + 모델명</b>으로 원격 API 또는 로컬 서버(Ollama, LM Studio, mlx_lm.server, llama.cpp)를 사용합니다. <b>원격 API를 쓸 때는 해당 AI의 이용약관을 반드시 확인하세요</b> — 출력물을 다른 모델 학습에 쓰는 것이 금지·제한될 수 있습니다.'))
        note.setWordWrap(True)
        lay.addWidget(note)
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("언어 / Language / 言語"))
        self.lang = QComboBox()
        for code, name in LANGUAGES.items():
            self.lang.addItem(name, code)
        lang_row.addWidget(self.lang)
        lang_row.addStretch(1)
        lay.addLayout(lang_row)
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel(T("데이터 폴더")))
        self.data_root = QLineEdit(str(paths.DATA_ROOT))
        self.data_root.setReadOnly(True)
        self.data_root.setToolTip(T("학습방안·생성 데이터·학습 결과·학생 모델은 프로그램 폴더 밖 이 폴더에 저장됩니다."))
        b_change = QPushButton(T("변경…"))
        b_open = QPushButton(T("열기"))
        b_change.clicked.connect(self._choose_data_root)
        b_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.DATA_ROOT))))
        data_row.addWidget(self.data_root, 1)
        data_row.addWidget(b_change)
        data_row.addWidget(b_open)
        lay.addLayout(data_row)
        self.cards = {r: RoleCard(r) for r in ROLES}
        for r in ROLES:
            lay.addWidget(self.cards[r])
        self.student = StudentCard()
        lay.addWidget(self.student)
        lay.addWidget(self._pipeline_box())
        lay.addWidget(self._backends_box())
        lay.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        save = QPushButton(T('설정 저장'))
        save.setDefault(True)
        save.clicked.connect(self.save)
        bottom.addWidget(save)
        outer.addLayout(bottom)
        self.load()

    def _pipeline_box(self) -> QGroupBox:
        box = QGroupBox(T('품질 평가 · 파이프라인'))
        g = QGridLayout(box)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(1, 10)
        self.threshold.setSingleStep(0.5)
        self.max_regen = QSpinBox()
        self.max_regen.setRange(0, 5)
        self.min_ratio = QDoubleSpinBox()
        self.min_ratio.setRange(0.1, 1.0)
        self.min_ratio.setSingleStep(0.05)
        self.topup = QSpinBox()
        self.topup.setRange(0, 5)
        self.t1_batch = QSpinBox()
        self.t1_batch.setRange(1, 20)
        self.holdout = QSpinBox()
        self.holdout.setRange(1, 200)
        items = ((T('통과 기준 점수(1~10)'), self.threshold), (T('최대 재생성 횟수'), self.max_regen),
                 (T('최소 채택 비율'), self.min_ratio), (T('추가 생성 라운드'), self.topup),
                 (T('교사1 1회 생성 문항 수'), self.t1_batch), (T('학습 전후 비교 문항 수'), self.holdout))
        for i, (lbl, w) in enumerate(items):
            g.addWidget(QLabel(lbl), i // 2, (i % 2) * 2)
            g.addWidget(w, i // 2, (i % 2) * 2 + 1)
        return box

    def _backends_box(self) -> QGroupBox:
        box = QGroupBox(T('로컬 LLM 백엔드'))
        lay = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.ollama_status = QLabel(T('Ollama: 확인 전'))
        b_check = QPushButton(T('상태 확인'))
        b_start = QPushButton(T('Ollama 실행'))
        b_inst = QPushButton(T('Ollama 설치'))
        b_check.clicked.connect(self._ollama_check)
        b_start.clicked.connect(self._ollama_start)
        b_inst.clicked.connect(self._ollama_install)
        for w in (self.ollama_status, b_check, b_start, b_inst):
            r1.addWidget(w)
        r1.addStretch(1)
        r2 = QHBoxLayout()
        self.llama_status = QLabel("llama.cpp")
        b_li = QPushButton(T('llama.cpp 설치'))
        b_ls = QPushButton(T('GGUF로 서버 실행 (포트 8081)'))
        b_li.clicked.connect(self._llama_install)
        b_ls.clicked.connect(self._llama_start)
        for w in (self.llama_status, b_li, b_ls):
            r2.addWidget(w)
        r2.addStretch(1)
        r3 = QHBoxLayout()
        lm = QLabel(T('LM Studio는 독점 소프트웨어라 자동 설치하지 않습니다. 직접 설치 후 로컬 서버(포트 1234)를 켜세요.'))
        lm.setWordWrap(True)
        b_lm = QPushButton(T('LM Studio 사이트'))
        b_lm.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://lmstudio.ai")))
        r3.addWidget(lm, 1)
        r3.addWidget(b_lm)
        r4 = QLabel(T("mlx_lm.server: 기본 설치에 포함 — 터미널에서 `mlx_lm.server --model <모델>` 실행 후 프리셋 'mlx_lm.server (로컬)' 사용"))
        r4.setWordWrap(True)
        r4.setStyleSheet("color: gray;")
        for r in (r1, r2, r3):
            lay.addLayout(r)
        lay.addWidget(r4)
        self._llama_proc = None
        return box

    # ------------------------------------------------------------------ 로드/저장
    def load(self) -> None:
        self.lang.setCurrentIndex(max(0, self.lang.findData(self.cfg.ui_language or language())))
        for r in ROLES:
            self.cards[r].load(self.cfg.roles[r])
        self.student.load(self.cfg)
        p = self.cfg.pipeline
        self.threshold.setValue(p.pass_threshold)
        self.max_regen.setValue(p.max_regenerations)
        self.min_ratio.setValue(p.min_pass_ratio)
        self.topup.setValue(p.max_topup_rounds)
        self.t1_batch.setValue(p.teacher1_batch)
        self.holdout.setValue(p.eval_holdout)

    def apply(self) -> None:
        for r in ROLES:
            self.cards[r].apply(self.cfg.roles[r])
        self.student.apply(self.cfg)
        p = self.cfg.pipeline
        p.pass_threshold = self.threshold.value()
        p.max_regenerations = self.max_regen.value()
        p.min_pass_ratio = self.min_ratio.value()
        p.max_topup_rounds = self.topup.value()
        p.teacher1_batch = self.t1_batch.value()
        p.eval_holdout = self.holdout.value()

    def _choose_data_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, T("데이터 폴더 선택"), str(paths.DATA_ROOT.parent))
        if not d:
            return
        self.data_root.setText(d)
        self.cfg.data_root = d
        save_config(self.cfg)
        QMessageBox.information(self, T("데이터 폴더"),
                                T("앱을 다시 시작하면 새 데이터 폴더를 사용합니다. 기존 데이터는 자동으로 옮겨지지 않으니, 필요하면 Finder에서 '{old}' 안의 projects·models 폴더를 옮기세요.",
                                  old=str(paths.DATA_ROOT)))

    def save(self) -> None:
        old_lang = self.cfg.ui_language or language()
        self.cfg.ui_language = self.lang.currentData()
        self.apply()
        save_config(self.cfg)
        warns = []
        t1, t2, o = (self.cfg.roles[r] for r in ("teacher1", "teacher2", "orchestrator"))
        if t1.model and t1.model == t2.model and t1.base_url == t2.base_url:
            warns.append(S.SAME_MODEL_TEACHERS)
        if o.model and o.base_url in (t1.base_url, t2.base_url) and o.model in (t1.model, t2.model):
            warns.append(S.SAME_MODEL_ORCH)
        self.saved.emit()
        if self.cfg.ui_language != old_lang:
            QMessageBox.information(self, "Language", T("언어 변경은 앱을 다시 시작하면 적용됩니다."))
        if warns:
            QMessageBox.warning(self, T('설정 저장됨 — 확인 필요'), "\n\n".join(warns))

    # ------------------------------------------------------------------ 로컬 백엔드 동작
    def _ollama_check(self) -> None:
        async def go():
            alive = await ollama_alive()
            models = await ollama_models() if alive else {}
            return alive, models

        def done(res):
            alive, models = res
            if alive:
                self.ollama_status.setText(T('Ollama: ✅ 실행 중 · 모델 {len}개', len=len(models)))
            elif ollama_binary():
                self.ollama_status.setText(T('Ollama: 설치됨 · 꺼져 있음'))
            else:
                self.ollama_status.setText(T('Ollama: 미설치'))

        run_async(go, done, lambda e: self.ollama_status.setText(T('Ollama: 오류 {v}', v=e[:80])))

    def _ollama_start(self) -> None:
        run_async(lambda: ensure_ollama(), lambda ok: self.ollama_status.setText(T('Ollama: ✅ 실행 중') if ok else T('Ollama: 시작 실패')),
                  lambda e: self.ollama_status.setText(T('Ollama: 오류 {v}', v=e[:80])))

    def _ollama_install(self) -> None:
        if ollama_binary() and QMessageBox.question(self, "Ollama", T('이미 설치되어 있습니다. 다시 설치할까요?')) != QMessageBox.Yes:
            return
        rep = QtReporter(self)
        rep.log_signal.connect(lambda lvl, m: self.ollama_status.setText(m))
        run_async(lambda: install_ollama(rep), lambda _: self._ollama_check(),
                  lambda e: QMessageBox.critical(self, T('Ollama 설치 실패'), e))

    def _llama_install(self) -> None:
        rep = QtReporter(self)
        rep.log_signal.connect(lambda lvl, m: self.llama_status.setText(m[:120]))
        run_async(lambda: install_llama_cpp(rep), lambda b: self.llama_status.setText(T('llama.cpp: ✅ 설치됨')),
                  lambda e: QMessageBox.critical(self, T('llama.cpp 설치 실패'), e))

    def _llama_start(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, T('GGUF 모델 선택'), str(Path.home()), "GGUF (*.gguf)")
        if not f:
            return
        try:
            if self._llama_proc and self._llama_proc.poll() is None:
                self._llama_proc.terminate()
            self._llama_proc = start_llama_server(f, 8081)
            self.llama_status.setText(T('llama.cpp: ✅ 서버 실행 중 (http://localhost:8081/v1)'))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "llama.cpp", str(e))

    def shutdown(self) -> None:
        if self._llama_proc and self._llama_proc.poll() is None:
            self._llama_proc.terminate()
