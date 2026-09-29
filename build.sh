#!/bin/sh
# Builds a self-contained module binary with PyInstaller and packages it for the registry.
# Run ./setup.sh first (the Viam cloud build does this automatically). Works on Linux and, under Git Bash,
# on Windows (where it produces dist/main.exe and points meta.json at it).
set -e
cd "$(dirname "$0")"

VENV_NAME="venv"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) PYTHON="$VENV_NAME/Scripts/python.exe"; EXE="dist/main.exe" ;;
    *)                    PYTHON="$VENV_NAME/bin/python";         EXE="dist/main" ;;
esac

if ! $PYTHON -m pip install "pyinstaller>=6.0" -Uqq; then
    exit 1
fi

# main.spec collects the OpenVINO native plugins (see comments there).
$PYTHON -m PyInstaller --clean --noconfirm main.spec

if [ "$EXE" != "dist/main" ]; then
    $PYTHON - <<'PY'
import json
m = json.load(open("meta.json"))
m["entrypoint"] = "dist/main.exe"
json.dump(m, open("meta.json", "w"), indent=2)
PY
fi

TAR_FILES="meta.json $EXE"
FIRST_RUN=$($PYTHON -c "import json; print(json.load(open('meta.json')).get('first_run', ''))" 2>/dev/null)
if [ -n "$FIRST_RUN" ] && [ -f "$FIRST_RUN" ]; then
    TAR_FILES="$TAR_FILES $FIRST_RUN"
fi
tar -czvf dist/archive.tar.gz $TAR_FILES
