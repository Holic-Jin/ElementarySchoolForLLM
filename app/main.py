"""ElementarySchoolForLLM 진입점. Copyright 2026 HolicJin — Apache-2.0."""
from __future__ import annotations

from .i18n import T

import os
import platform
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from . import APP_NAME, AUTHOR, VERSION

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)
    app.setOrganizationName(AUTHOR)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        QMessageBox.critical(None, APP_NAME, T('이 프로그램은 Apple Silicon Mac 전용입니다. (Intel Mac은 지원하지 않습니다)'))
        return 1

    from .core import paths
    from .core.config import load_config
    from .ui.main_window import MainWindow

    paths.ensure_dirs()
    try:
        moved = paths.migrate_legacy()  # 이전 버전의 저장소 안 workspace/ 등을 데이터 폴더로 이동
    except Exception as e:  # noqa: BLE001
        moved = []
        QMessageBox.warning(None, APP_NAME, str(e))
    win = MainWindow(load_config())
    if moved:
        QMessageBox.information(win, APP_NAME, T("이전 위치의 데이터를 데이터 폴더로 옮겼습니다:\n{paths}",
                                                 paths="\n".join(moved)))
    win.show()
    code = app.exec()
    # 응답 없는 네트워크 요청이 백그라운드 스레드에 남아 있어도 앱이 바로 닫히도록 즉시 종료
    # (실행 중인 파이프라인·학생 서버 등 하위 프로세스는 창을 닫을 때 이미 정리됨)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    sys.exit(main())
