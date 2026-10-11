"""대화상자: API 이용약관 확인, 사양 부족 에러, 정보."""
from __future__ import annotations

from ..i18n import T

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QLabel, QMessageBox, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from .. import VERSION
from ..core.config import ROLE_NAMES, AppConfig, save_config
from ..core.providers import is_local_url, tos_url_for
from . import strings as S


class ToSDialog(QDialog):
    def __init__(self, parent: QWidget | None, remote: list[tuple[str, str, str]]):
        super().__init__(parent)
        self.setWindowTitle(S.TOS_TITLE)
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        body = QLabel(S.TOS_BODY)
        body.setWordWrap(True)
        lay.addWidget(body)
        rows = []
        for role, url, tos in remote:
            link = f'<a href="{tos}">{tos}</a>' if tos else T('제공자 웹사이트에서 직접 확인')
            rows.append(T('<li><b>{role}</b> — {url}<br>이용약관: {link}</li>', role=role, url=url, link=link))
        lst = QLabel("<ul>" + "".join(rows) + "</ul>")
        lst.setOpenExternalLinks(True)
        lst.setTextInteractionFlags(Qt.TextBrowserInteraction)
        lst.setWordWrap(True)
        lay.addWidget(lst)
        self.check = QCheckBox(S.TOS_CHECK)
        lay.addWidget(self.check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.check.toggled.connect(self.buttons.button(QDialogButtonBox.Ok).setEnabled)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)


def ensure_tos(parent: QWidget | None, cfg: AppConfig, roles: list[str]) -> bool:
    """원격 API를 쓰는 역할이 있으면 약관 확인을 받는다. 주소별로 한 번 확인하면 기억한다."""
    remote = []
    for r in roles:
        rc = cfg.roles[r]
        if rc.base_url and not is_local_url(rc.base_url) and not cfg.tos_acknowledged.get(rc.base_url):
            remote.append((ROLE_NAMES[r], rc.base_url, tos_url_for(rc.base_url)))
    if not remote:
        return True
    dlg = ToSDialog(parent, remote)
    if dlg.exec() != QDialog.Accepted:
        return False
    for _, url, _ in remote:
        cfg.tos_acknowledged[url] = True
    save_config(cfg)
    return True


def spec_error(parent: QWidget | None, report_text: str) -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(S.SPEC_TITLE)
    box.setText(S.SPEC_BODY)
    box.setInformativeText(T('필요 사양과 현재 사양은 아래 [자세히]에서 확인하세요.'))
    box.setDetailedText(report_text)
    box.exec()


def show_text(parent: QWidget | None, title: str, text: str) -> None:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(720, 520)
    lay = QVBoxLayout(dlg)
    ed = QPlainTextEdit(text)
    ed.setReadOnly(True)
    lay.addWidget(ed)
    bb = QDialogButtonBox(QDialogButtonBox.Close)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    dlg.exec()


def about(parent: QWidget | None) -> None:
    QMessageBox.about(parent, T('ElementarySchoolForLLM 정보'), S.ABOUT.format(version=VERSION))
