#!/bin/sh
# Builds a self-contained module binary with PyInstaller and packages it for the registry.
# Run ./setup.sh first (the Viam cloud build does this automatically). Works on Linux and, under Git Bash,
# on Windows (where it produces dist/main.exe and packages it without first_run.sh).
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

# meta.json keeps "entrypoint": "dist/main" on every platform. The registry stores one entrypoint for the
# module, set by whichever upload runs last, so a Windows-only "dist/main.exe" breaks Linux. viam-server
# starts the module by absolute path, and on Windows Go's exec resolves an extensionless absolute path via
# PATHEXT, so "dist/main" finds dist/main.exe.
if [ "$EXE" != "dist/main" ]; then
    # first_run.sh is a POSIX shell script; Windows drivers come from Intel's installer. Package a copy of
    # meta.json without it, leaving the repo's meta.json (which the CLI uploads) untouched.
    STAGE="dist/package"
    rm -rf "$STAGE" && mkdir -p "$STAGE/dist"
    cp "$EXE" "$STAGE/dist/"
    $PYTHON - "$STAGE/meta.json" <<'PY'
import json, sys
m = json.load(open("meta.json"))
m["first_run"] = ""
json.dump(m, open(sys.argv[1], "w"), indent=2)
PY
    tar -czvf dist/archive.tar.gz -C "$STAGE" meta.json "$EXE"
    exit 0
fi

TAR_FILES="meta.json $EXE"
FIRST_RUN=$($PYTHON -c "import json; print(json.load(open('meta.json')).get('first_run', ''))" 2>/dev/null)
if [ -n "$FIRST_RUN" ] && [ -f "$FIRST_RUN" ]; then
    TAR_FILES="$TAR_FILES $FIRST_RUN"
fi
tar -czvf dist/archive.tar.gz $TAR_FILES
