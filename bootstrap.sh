#!/bin/bash
# ElementarySchoolForLLM 부트스트랩
# - 아무것도 설치되지 않은 Apple Silicon Mac에서 필요한 모든 것을 사용자 폴더에 설치한 뒤 앱을 실행한다.
# - 관리자 권한(sudo), Homebrew, Xcode Command Line Tools가 필요 없다.
# Copyright 2026 HolicJin — Apache-2.0
set -euo pipefail

APP_NAME="ElementarySchoolForLLM"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESLLM_HOME="${ESLLM_HOME:-$HOME/.ElementarySchoolForLLM}"
BIN_DIR="$ESLLM_HOME/bin"
VENV_DIR="$ESLLM_HOME/venv"
MARKER="$ESLLM_HOME/.bootstrapped"
PY_VERSION="3.12"
# uv 버전 고정: UV_VERSION=0.x.y 로 지정 가능. 기본값은 최신 릴리스(다운로드 후 SHA256 검증).
UV_VERSION="${UV_VERSION:-latest}"
OLLAMA_URL="https://ollama.com/download/Ollama-darwin.zip"

export UV_PYTHON_INSTALL_DIR="$ESLLM_HOME/python"
export UV_CACHE_DIR="$ESLLM_HOME/cache/uv"
export UV_PYTHON_BIN_DIR="$ESLLM_HOME/python/bin"  # ~/.local/bin 등 사용자 PATH를 건드리지 않음
export UV_TOOL_BIN_DIR="$ESLLM_HOME/python/bin"
export UV_NO_PROGRESS=0

# 안내 문구 언어: ESLLM_LANG → 앱 설정(ui_language) → macOS 시스템 언어
LANG_CODE="${ESLLM_LANG:-}"
if [ -z "$LANG_CODE" ] && [ -f "$ESLLM_HOME/config.json" ]; then
  LANG_CODE="$(/usr/bin/sed -n 's/.*"ui_language": *"\([a-z]*\)".*/\1/p' "$ESLLM_HOME/config.json" | head -1)"
fi
if [ -z "$LANG_CODE" ]; then
  LANG_CODE="$(/usr/bin/defaults read -g AppleLanguages 2>/dev/null | /usr/bin/sed -n '2s/[^a-z]*\([a-z][a-z]\).*/\1/p')"
fi
case "$LANG_CODE" in ko|ja|en) ;; *) LANG_CODE="en" ;; esac
m() { case "$LANG_CODE" in ja) printf '%s' "$2" ;; en) printf '%s' "$3" ;; *) printf '%s' "$1" ;; esac; }

say()  { printf '\033[1;34m[%s]\033[0m %s\n' "$APP_NAME" "$*"; }
warn() { printf '\033[1;33m[%s]\033[0m %s\n' "$(m 주의 注意 Warning)" "$*"; }
die() {
  printf '\033[1;31m[%s]\033[0m %s\n' "$(m 오류 エラー Error)" "$*" >&2
  /usr/bin/osascript -e "display dialog \"$*\" with title \"$APP_NAME\" buttons {\"OK\"} default button 1 with icon stop" >/dev/null 2>&1 || true
  exit 1
}

req_hash() {
  cat "$REPO_DIR/requirements.txt" | /usr/bin/shasum -a 256 | awk '{print $1}'
}

launch() {
  cd "$REPO_DIR"
  if [ "${ESLLM_NO_LAUNCH:-0}" = "1" ]; then  # 설치 검증용
    say "ESLLM_NO_LAUNCH=1 — $(m "앱을 실행하지 않고 종료합니다." "アプリを起動せずに終了します。" "exiting without launching the app.")"
    exit 0
  fi
  exec "$VENV_DIR/bin/python" -m app.main ${@+"$@"}
}

# 앱 런처(.app) 생성 — 이후 Finder/Dock에서 실행. 폴더를 옮기거나 이름을 바꿔도 실행할 때마다 경로를 고친다.
APP_BUNDLE="$HOME/Applications/$APP_NAME.app"
make_launcher() {
  mkdir -p "$APP_BUNDLE/Contents/MacOS"
  cat > "$APP_BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>io.github.holic-jin.elementaryschoolforllm</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.5</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Copyright 2026 HolicJin. Apache-2.0</string>
</dict>
</plist>
PLIST
  cat > "$APP_BUNDLE/Contents/MacOS/$APP_NAME" <<LAUNCHER
#!/bin/bash
exec /bin/bash "$REPO_DIR/bootstrap.sh" \${@+"\$@"} >>"$ESLLM_HOME/logs/launcher.log" 2>&1
LAUNCHER
  chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
}

# ---------------------------------------------------------------- 0. 환경 확인
[ "$(uname -m)" = "arm64" ] || die "$(m "이 프로그램은 Apple Silicon Mac 전용입니다. (Intel Mac은 지원하지 않습니다)" "このプログラムは Apple Silicon Mac 専用です。(Intel Mac には対応していません)" "This program is for Apple Silicon Macs only. (Intel Macs are not supported)")"

MACOS_VER="$(sw_vers -productVersion)"
MAJOR="${MACOS_VER%%.*}"
REST="${MACOS_VER#*.}"; MINOR="${REST%%.*}"
[ "$MAJOR" = "$MACOS_VER" ] && MINOR=0
if [ "$MAJOR" -lt 13 ] || { [ "$MAJOR" -eq 13 ] && [ "$MINOR" -lt 5 ]; }; then
  die "$(m "macOS 13.5 이상이 필요합니다." "macOS 13.5 以降が必要です。" "macOS 13.5 or later is required.") ($MACOS_VER)"
fi

# 이미 설치되어 있고 requirements.txt가 바뀌지 않았으면 바로 실행
if [ -f "$MARKER" ] && [ -x "$VENV_DIR/bin/python" ] && [ "$(cat "$MARKER")" = "$(req_hash)" ]; then
  if [ -d "$APP_BUNDLE" ] && ! grep -qF "$REPO_DIR/bootstrap.sh" "$APP_BUNDLE/Contents/MacOS/$APP_NAME" 2>/dev/null; then
    make_launcher  # 프로그램 폴더가 옮겨졌으면 런처 경로 갱신
  fi
  launch ${@+"$@"}
fi

say "$(m "첫 실행 설치를 시작합니다. 설치 위치:" "初回セットアップを開始します。インストール先:" "Starting first-run setup. Install location:") $ESLLM_HOME"
mkdir -p "$BIN_DIR" "$ESLLM_HOME/logs" "$HOME/Applications"

/usr/bin/curl -fsS --max-time 15 -o /dev/null https://pypi.org/simple/ \
  || die "$(m "인터넷에 연결할 수 없습니다. 첫 실행 설치에는 인터넷 연결이 필요합니다." "インターネットに接続できません。初回セットアップにはインターネット接続が必要です。" "Cannot connect to the internet. First-run setup needs an internet connection.")"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# ---------------------------------------------------------------- 1. uv (Python/패키지 관리자 단일 바이너리)
if [ ! -x "$BIN_DIR/uv" ]; then
  if [ "$UV_VERSION" = "latest" ]; then
    UV_BASE="https://github.com/astral-sh/uv/releases/latest/download"
  else
    UV_BASE="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
  fi
  UV_ASSET="uv-aarch64-apple-darwin.tar.gz"
  say "$(m "uv 다운로드 중" "uv をダウンロード中" "Downloading uv") ($UV_VERSION)"
  /usr/bin/curl -fL --retry 3 --progress-bar -o "$TMP_DIR/$UV_ASSET" "$UV_BASE/$UV_ASSET" || die "$(m "uv 다운로드 실패" "uv のダウンロードに失敗" "uv download failed")"
  /usr/bin/curl -fsL --retry 3 -o "$TMP_DIR/$UV_ASSET.sha256" "$UV_BASE/$UV_ASSET.sha256" || die "$(m "uv 체크섬 다운로드 실패" "uv のチェックサムのダウンロードに失敗" "uv checksum download failed")"
  EXPECTED="$(awk '{print $1}' "$TMP_DIR/$UV_ASSET.sha256")"
  ACTUAL="$(/usr/bin/shasum -a 256 "$TMP_DIR/$UV_ASSET" | awk '{print $1}')"
  [ -n "$EXPECTED" ] && [ "$EXPECTED" = "$ACTUAL" ] || die "$(m "uv SHA256 검증 실패" "uv の SHA256 検証に失敗" "uv SHA256 verification failed") ($EXPECTED / $ACTUAL)"
  tar -xzf "$TMP_DIR/$UV_ASSET" -C "$TMP_DIR"
  cp "$TMP_DIR/uv-aarch64-apple-darwin/uv" "$BIN_DIR/uv"
  chmod +x "$BIN_DIR/uv"
  say "$(m "uv 설치 완료 (SHA256 검증됨)" "uv のインストール完了 (SHA256 検証済み)" "uv installed (SHA256 verified)")"
fi
UV="$BIN_DIR/uv"

# ---------------------------------------------------------------- 2. Python + 가상환경 + 패키지
say "Python $PY_VERSION — $(m "준비 중" "準備中" "preparing")"
PY_OUT="$("$UV" python install "$PY_VERSION" 2>&1)" || die "$(m "Python 설치 실패" "Python のインストールに失敗" "Python installation failed"): $PY_OUT"
printf '%s\n' "$PY_OUT" | grep -v "is not on your PATH" || true
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV" venv --python "$PY_VERSION" "$VENV_DIR" || die "$(m "가상환경 생성 실패" "仮想環境の作成に失敗" "Failed to create the virtual environment")"
fi
say "$(m "필수 패키지 설치 중 (PySide6, MLX 등 — 수 분 걸릴 수 있습니다)" "必須パッケージをインストール中 (PySide6、MLX など — 数分かかる場合があります)" "Installing required packages (PySide6, MLX, ... — this can take a few minutes)")"
"$UV" pip install --python "$VENV_DIR/bin/python" -r "$REPO_DIR/requirements.txt" || die "$(m "패키지 설치 실패. 네트워크 상태를 확인하고 다시 실행해 주세요." "パッケージのインストールに失敗しました。ネットワークを確認して再実行してください。" "Package installation failed. Check your network and run again.")"

# ---------------------------------------------------------------- 3. Ollama (로컬 LLM 백엔드, 선택이지만 기본 설치)
find_ollama() {
  for p in "$(command -v ollama 2>/dev/null || true)" \
           "$HOME/Applications/Ollama.app/Contents/Resources/ollama" \
           "/Applications/Ollama.app/Contents/Resources/ollama" \
           "/usr/local/bin/ollama"; do
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
if ! find_ollama >/dev/null; then
  say "$(m "Ollama 다운로드 중 (로컬 LLM 실행용)" "Ollama をダウンロード中 (ローカル LLM 実行用)" "Downloading Ollama (for running local LLMs)")"
  if /usr/bin/curl -fL --retry 3 --progress-bar -o "$TMP_DIR/Ollama-darwin.zip" "$OLLAMA_URL"; then
    /usr/bin/ditto -x -k "$TMP_DIR/Ollama-darwin.zip" "$TMP_DIR/ollama_unzip"
    if /usr/bin/codesign --verify --deep --strict "$TMP_DIR/ollama_unzip/Ollama.app" 2>/dev/null; then
      rm -rf "$HOME/Applications/Ollama.app"
      mv "$TMP_DIR/ollama_unzip/Ollama.app" "$HOME/Applications/Ollama.app"
      say "$(m "Ollama 설치 완료 (코드 서명 검증됨)" "Ollama のインストール完了 (コード署名を検証済み)" "Ollama installed (code signature verified)"): ~/Applications/Ollama.app"
    else
      warn "$(m "Ollama 코드 서명 검증 실패 — 설치하지 않습니다. 필요하면 https://ollama.com 에서 직접 설치하세요." "Ollama のコード署名検証に失敗 — インストールしません。必要なら https://ollama.com から直接インストールしてください。" "Ollama code signature verification failed — not installing. Install it from https://ollama.com if needed.")"
    fi
  else
    warn "$(m "Ollama 다운로드 실패 — 로컬 LLM이 필요하면 나중에 설정 화면에서 다시 시도하세요." "Ollama のダウンロードに失敗 — ローカル LLM が必要なら後で設定画面から再試行してください。" "Ollama download failed — retry later from the Settings screen if you need local LLMs.")"
  fi
fi

# ---------------------------------------------------------------- 4. 앱 런처
make_launcher
say "$(m "앱 런처 생성" "アプリランチャーを作成" "Created app launcher"): $APP_BUNDLE"

req_hash > "$MARKER"
say "$(m "설치 완료. 앱을 시작합니다." "セットアップ完了。アプリを起動します。" "Setup complete. Starting the app.")"
say "$(m "참고: macOS가 실행을 막으면 Finder에서 start.command를 우클릭 → 열기를 선택하세요." "参考: macOS が実行をブロックした場合は、Finder で start.command を右クリック → 開く を選んでください。" "Note: if macOS blocks it, right-click start.command in Finder and choose Open.")"
launch ${@+"$@"}
