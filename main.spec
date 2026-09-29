# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the erh:openvino module binary. Invoked by build.sh.
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# The OpenVINO wheel ships its runtime, frontends and device plugins (CPU, GPU, NPU, AUTO, HETERO, ...)
# as shared libraries next to the Python package; PyInstaller does not discover them on its own.
ov_datas, ov_binaries, ov_hiddenimports = collect_all("openvino")

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=ov_binaries,
    datas=ov_datas,
    hiddenimports=ov_hiddenimports + ["googleapiclient"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Usage telemetry is blocked in src/main.py; keep it out of the binary entirely.
    excludes=["openvino_telemetry"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="main",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
