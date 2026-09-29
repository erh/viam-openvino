"""``erh:openvino:mlmodel`` - an mlmodel service backed by the OpenVINO runtime."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import openvino as ov
from typing_extensions import Self
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.mlmodel import Metadata, MLModel, TensorInfo
from viam.utils import ValueTypes, dict_to_struct, struct_to_dict

from . import registry
from .config import MLModelConfig
from .engine import OpenVINOEngine, list_devices

LOGGER = getLogger(__name__)


class OpenVINOMLModel(MLModel, EasyResource):
    MODEL: ClassVar[Model] = "erh:openvino:mlmodel"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: Optional[MLModelConfig] = None
        self.engine: Optional[OpenVINOEngine] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._metadata: Optional[Metadata] = None
        self.logger = getattr(self, "logger", None) or LOGGER

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        registry.register(self)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        MLModelConfig.from_attributes(attrs)  # raises ValueError with an actionable message
        return [], []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = MLModelConfig.from_attributes(struct_to_dict(config.attributes))
        engine = OpenVINOEngine(cfg, self.logger, resource_name=config.name)
        engine.load()  # blocking; failures propagate and fail the resource build
        old_engine, old_executor = self.engine, self._executor
        self.cfg = cfg
        self.engine = engine
        self._executor = ThreadPoolExecutor(max_workers=max(1, engine.num_requests), thread_name_prefix=f"ov-{config.name}")
        self._metadata = self._build_metadata()
        if old_executor is not None:
            old_executor.shutdown(wait=False)
        if old_engine is not None:
            old_engine.close()

    async def close(self) -> None:
        registry.unregister(self)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    # ------------------------------------------------------------- mlmodel
    async def infer(
        self,
        input_tensors: Dict[str, Any],
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        engine = self._require_engine()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, engine.infer, input_tensors)

    async def metadata(self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> Metadata:
        self._require_engine()
        assert self._metadata is not None
        return self._metadata

    def _build_metadata(self) -> Metadata:
        assert self.engine is not None and self.cfg is not None
        engine, cfg = self.engine, self.cfg
        input_info = [
            TensorInfo(name=t.name, data_type=t.viam_dtype, shape=list(t.shape)) for t in engine.inputs
        ]
        output_info = []
        for t in engine.outputs:
            info = TensorInfo(name=t.name, data_type=t.viam_dtype, shape=list(t.shape))
            if cfg.label_path:
                # Same convention as the tflite_cpu and onnx-cpu modules: vision:mlmodel reads extra["labels"].
                info.extra.CopyFrom(dict_to_struct({"labels": cfg.label_path}))
            output_info.append(info)
        description = (
            f"OpenVINO {ov.get_version()}; model={cfg.model_path} ({cfg.model_format}); "
            f"device={cfg.device}; execution_devices={engine.execution_devices}"
        )
        return Metadata(name=cfg.model_name, type="openvino", description=description, input_info=input_info, output_info=output_info)

    # ----------------------------------------------------------- do_command
    # Note: the mlmodel gRPC API has no DoCommand RPC, so remote clients reach these commands through
    # erh:openvino:diagnostics or erh:openvino:yolo (see registry.py). This implementation is what they call.
    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if not isinstance(cmd, str):
            raise ValueError("do_command requires a string 'command' key: one of list_devices, get_compiled_properties, benchmark, stats")
        loop = asyncio.get_running_loop()
        if cmd == "list_devices":
            core = self.engine.core if self.engine is not None else None
            return await loop.run_in_executor(self._executor, list_devices, core)
        if cmd == "get_compiled_properties":
            return self._require_engine().compiled_properties()
        if cmd == "stats":
            return self._require_engine().stats.snapshot()
        if cmd == "benchmark":
            engine = self._require_engine()
            iterations = _int_arg(command, "iterations", 100)
            warmup = _int_arg(command, "warmup", 10)
            input_shape = command.get("input_shape")
            if input_shape is not None and not isinstance(input_shape, Mapping):
                raise ValueError("benchmark 'input_shape' must be an object mapping input name to a list of ints")
            shapes = {str(k): [int(d) for d in v] for k, v in (input_shape or {}).items()}
            return await loop.run_in_executor(None, engine.benchmark, iterations, warmup, shapes or None)
        raise ValueError(f"unknown command '{cmd}'; expected one of list_devices, get_compiled_properties, benchmark, stats")

    def _require_engine(self) -> OpenVINOEngine:
        if self.engine is None:
            raise RuntimeError(f"mlmodel '{self.name}' is not configured")
        return self.engine


def _int_arg(command: Mapping[str, Any], key: str, default: int) -> int:
    v = command.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < 0:
        raise ValueError(f"benchmark '{key}' must be a non-negative integer")
    return int(v)
