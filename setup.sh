#!/bin/sh
# Creates ./venv and installs the pinned requirements. Used by the Viam cloud build (Linux) and by the
# Windows GitHub Actions job (run under Git Bash). Override the interpreter with PYTHON=/path/to/python3.x;
# OpenVINO publishes wheels for Python 3.9 through 3.13.
cd `dirname $0`

VENV_NAME="venv"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        VENV_PYTHON="$VENV_NAME/Scripts/python.exe"
        DEFAULT_PYTHON="python"
        ;;
    *)
        VENV_PYTHON="$VENV_NAME/bin/python"
        DEFAULT_PYTHON="python3"
        ;;
esac
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
ENV_ERROR="This module requires Python >=3.9,<=3.13, pip, and virtualenv to be installed."

if ! $PYTHON -m venv $VENV_NAME >/dev/null 2>&1; then
    echo "Failed to create virtualenv."
    if command -v apt-get >/dev/null; then
        echo "Detected Debian/Ubuntu, attempting to install python3-venv automatically."
        SUDO="sudo"
        if ! command -v $SUDO >/dev/null; then
            SUDO=""
        fi
        if ! apt info python3-venv >/dev/null 2>&1; then
            echo "Package info not found, trying apt update"
            $SUDO apt -qq update >/dev/null
        fi
        $SUDO apt install -qqy python3-venv >/dev/null 2>&1
        if ! $PYTHON -m venv $VENV_NAME >/dev/null 2>&1; then
            echo $ENV_ERROR >&2
            exit 1
        fi
    else
        echo $ENV_ERROR >&2
        exit 1
    fi
fi

# -qq suppresses extraneous output from pip
echo "Virtualenv found/created. Installing/upgrading Python packages..."
if ! [ -f .installed ]; then
    if ! $VENV_PYTHON -m pip install -r requirements.txt -Uqq; then
        exit 1
    else
        touch .installed
    fi
fi
