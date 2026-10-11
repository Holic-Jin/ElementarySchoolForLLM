"""작업 신호등 위젯: 🔴 멈춤·헛돎 / 🟠 진행 중(추론·로딩) / 🟢 완료 / ⚪ 대기."""
from __future__ import annotations

import time
import weakref
from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from ..core.activity import GRAY, GREEN, ORANGE, RED, Activity, LightState
from ..i18n import T

COLORS = {GRAY: "#9e9e9e", ORANGE: "#f39c12", RED: "#e5484d", GREEN: "#30a46c"}
_lights: "weakref.WeakSet[StatusLight]" = weakref.WeakSet()
_timer: QTimer | None = None


def _tick() -> None:
    for w in list(_lights):
        try:
            w.refresh()
        except RuntimeError:  # 삭제된 위젯
            pass


class StatusLight(QWidget):
    """Activity 하나(또는 평가 함수)를 0.5초마다 다시 평가해 색과 설명을 보여준다."""

    def __init__(self, activity: Activity | None = None, title: str = "", show_text: bool = True,
                 evaluator: Callable[[], LightState] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        global _timer
        self.activity = activity
        self.evaluator = evaluator
        self.title = title
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.dot = QLabel()
        self.dot.setFixedSize(14, 14)
        lay.addWidget(self.dot)
        self.name = QLabel(title)
        if title:
            self.name.setStyleSheet("font-weight: 600;")
            lay.addWidget(self.name)
        self.text = QLabel("")
        self.text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.text.setVisible(show_text)
        lay.addWidget(self.text, 1)
        self._blink = False
        self._last = None
        _lights.add(self)
        if _timer is None:
            _timer = QTimer()
            _timer.timeout.connect(_tick)
            _timer.start(500)
        self.refresh()

    def set_activity(self, activity: Activity | None) -> None:
        self.activity = activity
        self.refresh()

    def state(self) -> LightState:
        if self.evaluator:
            return self.evaluator()
        if self.activity is None:
            return LightState(GRAY, T("대기"))
        return self.activity.evaluate()

    def refresh(self) -> None:
        st = self.state()
        # 진행 중(주황)일 때 방금 신호가 들어왔으면 깜빡여서 '살아 있음'을 보여준다
        alive = (st.color == ORANGE and self.activity is not None
                 and time.monotonic() - self.activity.last_beat < 1.5)
        self._blink = not self._blink if alive else False
        color = COLORS[st.color]
        key = (st.color, self._blink)
        if key != self._last:
            border = "#ffffff" if self._blink else color
            self.dot.setStyleSheet(f"background:{color}; border:2px solid {border}; border-radius:7px;")
            self._last = key
        self.text.setText(st.text)
        self.setToolTip(f"{self.title}: {st.text}" if self.title else st.text)
