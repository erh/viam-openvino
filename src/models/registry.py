"""In-process registry of live ``erh:openvino:mlmodel`` engines.

The mlmodel gRPC API has no DoCommand RPC (only Infer, Metadata and GetStatus), so clients cannot
call DoCommand on an mlmodel service. Diagnostics (list_devices, get_compiled_properties, benchmark,
stats) are therefore exposed through ``erh:openvino:diagnostics`` (a generic service) and through
``erh:openvino:yolo``'s DoCommand, both of which reach the engine through this registry when the
mlmodel runs in the same module process.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional

from viam.utils import ValueTypes

if TYPE_CHECKING:  # pragma: no cover
    from .openvino_mlmodel import OpenVINOMLModel

_lock = threading.Lock()
_models: Dict[str, "OpenVINOMLModel"] = {}

DIAGNOSTIC_COMMANDS = ("list_devices", "get_compiled_properties", "benchmark", "stats")


def register(model: "OpenVINOMLModel") -> None:
    with _lock:
        _models[model.name] = model


def unregister(model: "OpenVINOMLModel") -> None:
    with _lock:
        if _models.get(model.name) is model:
            del _models[model.name]


def get(name: str) -> Optional["OpenVINOMLModel"]:
    with _lock:
        return _models.get(name)


def names() -> List[str]:
    with _lock:
        return sorted(_models)


async def run_diagnostic(command: Mapping[str, ValueTypes], mlmodel_name: Optional[str]) -> Mapping[str, ValueTypes]:
    """Route a diagnostics command to the named in-process mlmodel (or, for list_devices, to OpenVINO directly)."""
    cmd = command.get("command")
    if cmd == "list_devices":
        from .engine import list_devices

        target = get(mlmodel_name) if mlmodel_name else None
        core = target.engine.core if target is not None and target.engine is not None else None
        return await asyncio.get_running_loop().run_in_executor(None, list_devices, core)
    if cmd not in DIAGNOSTIC_COMMANDS:
        raise ValueError(f"unknown command '{cmd}'; expected one of {', '.join(DIAGNOSTIC_COMMANDS)}")
    if not mlmodel_name:
        raise ValueError(f"command '{cmd}' needs an mlmodel: set the 'mlmodel_name' attribute or pass 'mlmodel_name' in the command")
    target = get(mlmodel_name)
    if target is None:
        raise ValueError(
            f"mlmodel '{mlmodel_name}' is not a erh:openvino:mlmodel running in this module process "
            f"(known: {names() or 'none'}); diagnostics only work for that model"
        )
    return await target.do_command(command)
