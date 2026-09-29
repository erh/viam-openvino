"""OpenVINO engine: model loading, preprocessing, compilation, and a pool of
infer requests so concurrent ``Infer`` calls run in parallel.

Everything here is synchronous and blocking; the resource layer runs it in a
thread pool.
"""

from __future__ import annotations

import glob
import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm

from .config import MLModelConfig, ParsedDevice, PreprocessConfig
from .stats import LatencyStats, summarize_ms

# OpenVINO element type name -> Viam mlmodel data_type string (matches the
# tflite_cpu / onnx-cpu modules: "float32", "uint8", ...).
OV_TO_VIAM_DTYPE: Dict[str, str] = {
    "f32": "float32",
    "f64": "float64",
    "f16": "float16",
    "bf16": "bfloat16",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "boolean": "bool",
}

OV_TO_NP_DTYPE: Dict[str, Any] = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "boolean": np.bool_,
}

_RESIZE = {
    "linear": ResizeAlgorithm.RESIZE_LINEAR,
    "nearest": ResizeAlgorithm.RESIZE_NEAREST,
    "cubic": ResizeAlgorithm.RESIZE_CUBIC,
}
_COLOR = {"RGB": ColorFormat.RGB, "BGR": ColorFormat.BGR}
# ov.Type("u8") does not reliably parse short names, so map explicitly.
_OV_TYPES = {
    "u8": ov.Type.u8, "i8": ov.Type.i8, "u16": ov.Type.u16, "i16": ov.Type.i16, "i32": ov.Type.i32,
    "i64": ov.Type.i64, "f16": ov.Type.f16, "f32": ov.Type.f32, "f64": ov.Type.f64,
}

DRIVER_HINTS = {
    "GPU": (
        "the Intel compute runtime is probably missing. The module's first_run.sh installs it on Ubuntu "
        "(run it as root if the automatic first run could not); otherwise install intel-opencl-icd, "
        "intel-level-zero-gpu (or libze-intel-gpu1) and libze1, add the viam-server user to the 'render' "
        "group and restart. See the module README."
    ),
    "NPU": (
        "the Intel NPU driver is probably missing. The module's first_run.sh installs it on Ubuntu "
        "(run it as root if the automatic first run could not); otherwise install the intel/linux-npu-driver "
        "packages (intel-driver-compiler-npu, intel-fw-npu, intel-level-zero-npu) and libze1, make sure "
        "/dev/accel/accel0 exists and is accessible (kernel 6.8+), then restart. See the module README."
    ),
}


@dataclass
class TensorSpec:
    name: str
    ov_type: str  # e.g. "f32"
    shape: List[int]  # -1 for dynamic dims

    @property
    def viam_dtype(self) -> str:
        return OV_TO_VIAM_DTYPE.get(self.ov_type, self.ov_type)

    @property
    def np_dtype(self) -> Any:
        return OV_TO_NP_DTYPE.get(self.ov_type)

    @property
    def is_dynamic(self) -> bool:
        return any(d < 0 for d in self.shape)


def _partial_shape_to_list(ps: ov.PartialShape) -> List[int]:
    if ps.rank.is_dynamic:
        return [-1]
    return [(-1 if d.is_dynamic else d.get_length()) for d in ps]


def _port_name(port: Any, fallback: str) -> str:
    try:
        return port.get_any_name()
    except Exception:
        return fallback


def _device_available(name: str, available: Sequence[str]) -> bool:
    if name in available:
        return True
    family = name.split(".")[0]
    # A single GPU is listed as "GPU"; multiple as "GPU.0", "GPU.1".
    return any(a == family or a.startswith(family + ".") for a in available) if "." not in name else False


def expand_auto(available: Sequence[str]) -> str:
    """Turn a bare AUTO into an explicit priority list over the accelerators that are actually present.

    OpenVINO's own AUTO only ever considers GPU and CPU; the NPU is ignored unless listed. This module prefers
    GPU (widest op support), then NPU, then CPU. AUTO falls back to the next candidate if a compile fails.
    """
    gpus = sorted(d for d in available if d == "GPU" or d.startswith("GPU."))
    npus = sorted(d for d in available if d == "NPU" or d.startswith("NPU."))
    order = gpus + npus + ["CPU"]
    return "AUTO:" + ",".join(order)


def list_devices(core: Optional[ov.Core] = None) -> Dict[str, Any]:
    core = core or ov.Core()
    devices: List[Dict[str, Any]] = []
    for name in core.available_devices:
        entry: Dict[str, Any] = {"name": name}
        for key, prop in (("full_name", "FULL_DEVICE_NAME"), ("type", "DEVICE_TYPE")):
            try:
                entry[key] = str(core.get_property(name, prop))
            except Exception as e:  # pragma: no cover - device specific
                entry[key] = f"unavailable ({e.__class__.__name__})"
        try:
            entry["capabilities"] = [str(c) for c in core.get_property(name, "OPTIMIZATION_CAPABILITIES")]
        except Exception:  # pragma: no cover - device specific
            entry["capabilities"] = []
        devices.append(entry)
    return {"devices": devices, "openvino_version": ov.get_version()}


class OpenVINOEngine:
    """Owns a compiled OpenVINO model and a pool of infer requests."""

    def __init__(self, cfg: MLModelConfig, logger: logging.Logger, resource_name: str = "") -> None:
        self.cfg = cfg
        self.logger = logger
        self.resource_name = resource_name
        self.core = ov.Core()
        self.stats = LatencyStats()

        self.compiled: Optional[ov.CompiledModel] = None
        self.compile_device: str = cfg.device  # may drop unavailable AUTO candidates
        self.inputs: List[TensorSpec] = []
        self.outputs: List[TensorSpec] = []
        self.execution_devices: List[str] = []
        self.loaded_from_cache: Optional[bool] = None
        self.compile_time_ms: float = 0.0
        self.num_requests: int = 0

        self._pool: "queue.LifoQueue[ov.InferRequest]" = queue.LifoQueue()
        self._auto_switch_pending = False  # AUTO is still serving on the CPU while the accelerator compiles
        self._closed = False
        self._logged_single_input_alias = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        cfg = self.cfg
        available = list(self.core.available_devices)
        self.logger.info(
            "OpenVINO %s; available devices: %s; requested device: %s",
            ov.get_version(), available or "[]", cfg.device,
        )
        if cfg.parsed_device.mode == "AUTO" and not cfg.parsed_device.devices:
            self.compile_device = expand_auto(available)
            self.logger.info("device AUTO -> '%s' (GPU, then NPU, then CPU, among available devices)", self.compile_device)
        self._check_device_availability(cfg.parsed_device, available)

        model = self.core.read_model(cfg.model_path)
        self._apply_input_shape(model)
        model = self._apply_preprocess(model)

        # Caller-facing tensor specs come from the (possibly pre-processed) model.
        self.inputs = [
            TensorSpec(_port_name(p, f"input{i}"), p.get_element_type().get_type_name(), _partial_shape_to_list(p.get_partial_shape()))
            for i, p in enumerate(model.inputs)
        ]
        self.outputs = [
            TensorSpec(_port_name(p, f"output{i}"), p.get_element_type().get_type_name(), _partial_shape_to_list(p.get_partial_shape()))
            for i, p in enumerate(model.outputs)
        ]

        compile_config = self._compile_config()
        if cfg.extra_config:
            self.logger.info("extra OpenVINO properties: %s", cfg.extra_config)
        self.compiled = self._compile_with_cache_recovery(model, compile_config)

        self.execution_devices = self.current_execution_devices()
        self._auto_switch_pending = any(d.startswith("(") for d in self.execution_devices)
        self.loaded_from_cache = self._safe_property("LOADED_FROM_CACHE", None)

        if cfg.num_requests:
            self.num_requests = cfg.num_requests
        else:
            self.num_requests = int(self._safe_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS", 1) or 1)
        for _ in range(self.num_requests):
            self._pool.put(self.compiled.create_infer_request())

        self.logger.info(
            "compiled %s (%s) for '%s' -> execution devices %s; cache %s; compile time %.0f ms; "
            "%d infer request(s); inputs=%s outputs=%s%s",
            os.path.basename(cfg.model_path), cfg.model_format, cfg.device, self.execution_devices,
            "hit" if self.loaded_from_cache else ("miss" if self.loaded_from_cache is not None else "n/a"),
            self.compile_time_ms, self.num_requests,
            [(t.name, t.viam_dtype, t.shape) for t in self.inputs],
            [(t.name, t.viam_dtype, t.shape) for t in self.outputs],
            " [serving on CPU until the accelerator compile finishes]" if self._auto_switch_pending else "",
        )

    def _check_device_availability(self, parsed: ParsedDevice, available: Sequence[str]) -> None:
        missing = [d for d in parsed.devices if not _device_available(d, available)]
        if not missing:
            return
        details = []
        for d in missing:
            family = d.split(".")[0]
            hint = DRIVER_HINTS.get(family)
            details.append(f"'{d}' is not in available devices {list(available)}" + (f": {hint}" if hint else ""))
        msg = "; ".join(details)
        if parsed.is_auto:
            remaining = [d for d in parsed.devices if d not in missing]
            self.compile_device = "AUTO:" + ",".join(remaining) if remaining else "AUTO"
            self.logger.warning(
                "device '%s': %s. Continuing with whatever AUTO selects (compiling for '%s').",
                parsed.raw, msg, self.compile_device,
            )
            return
        self.logger.error("device '%s' cannot be used: %s", parsed.raw, msg)
        raise RuntimeError(f"requested device '{parsed.raw}' is unavailable: {msg}")

    def _apply_input_shape(self, model: ov.Model) -> None:
        if not self.cfg.input_shape:
            return
        names = {_port_name(p, f"input{i}") for i, p in enumerate(model.inputs)}
        unknown = sorted(set(self.cfg.input_shape) - names)
        if unknown:
            raise ValueError(f"input_shape references unknown input(s) {unknown}; model inputs are {sorted(names)}")
        shapes = {name: ov.PartialShape(dims) for name, dims in self.cfg.input_shape.items()}
        try:
            model.reshape(shapes)
        except Exception as e:
            raise ValueError(f"failed to reshape model inputs to {self.cfg.input_shape}: {e}") from e

    def _apply_preprocess(self, model: ov.Model) -> ov.Model:
        pp: Optional[PreprocessConfig] = self.cfg.preprocess
        if pp is None or not pp.enabled:
            return model
        names = [_port_name(p, f"input{i}") for i, p in enumerate(model.inputs)]
        if pp.input_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"preprocess.input_name is required because the model has {len(names)} inputs: {names}"
                )
            input_name = names[0]
        else:
            input_name = pp.input_name
            if input_name not in names:
                raise ValueError(f"preprocess.input_name '{input_name}' is not a model input; inputs are {names}")

        ppp = PrePostProcessor(model)
        inp = ppp.input(input_name)
        tensor = inp.tensor()
        steps = inp.preprocess()
        model_elem_type = model.input(input_name).get_element_type()

        if pp.tensor_element_type is not None:
            tensor.set_element_type(_OV_TYPES[pp.tensor_element_type])
        if pp.tensor_layout is not None:
            tensor.set_layout(ov.Layout(pp.tensor_layout))
        if pp.tensor_color_format is not None:
            tensor.set_color_format(_COLOR[pp.tensor_color_format])
        if pp.resize != "none":
            tensor.set_spatial_dynamic_shape()

        if pp.tensor_element_type is not None and _OV_TYPES[pp.tensor_element_type] != model_elem_type:
            steps.convert_element_type(model_elem_type)
        if pp.tensor_color_format is not None and pp.tensor_color_format != pp.model_color_format:
            steps.convert_color(_COLOR[pp.model_color_format])
        if pp.resize != "none":
            steps.resize(_RESIZE[pp.resize])
        if pp.mean is not None:
            steps.mean(pp.mean if len(pp.mean) > 1 else pp.mean[0])
        if pp.scale is not None:
            steps.scale(pp.scale if len(pp.scale) > 1 else pp.scale[0])
        if pp.model_layout is not None:
            inp.model().set_layout(ov.Layout(pp.model_layout))
        try:
            return ppp.build()
        except Exception as e:
            raise ValueError(f"failed to build preprocessing for input '{input_name}': {e}") from e

    def _compile_config(self) -> Dict[str, Any]:
        cfg = self.cfg
        config: Dict[str, Any] = {"PERFORMANCE_HINT": cfg.performance_hint}
        if cfg.cache_dir:
            os.makedirs(cfg.cache_dir, exist_ok=True)
            config["CACHE_DIR"] = cfg.cache_dir
        if cfg.inference_precision:
            config["INFERENCE_PRECISION_HINT"] = cfg.inference_precision
        if cfg.num_threads:
            if cfg.parsed_device.mode is None and cfg.parsed_device.devices == ["CPU"]:
                config["INFERENCE_NUM_THREADS"] = cfg.num_threads
            else:
                self.logger.warning("num_threads only applies when device is exactly 'CPU'; ignoring for '%s'", cfg.device)
        if cfg.num_requests and cfg.performance_hint != "LATENCY":
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = cfg.num_requests
        config.update(cfg.extra_config)
        return config

    def _compile_with_cache_recovery(self, model: ov.Model, config: Dict[str, Any]) -> ov.CompiledModel:
        cfg = self.cfg
        start = time.perf_counter()
        try:
            compiled = self.core.compile_model(model, self.compile_device, config)
        except Exception as first_err:
            if not cfg.cache_dir:
                raise RuntimeError(f"failed to compile model on device '{cfg.device}': {first_err}") from first_err
            self.logger.warning(
                "compile on '%s' failed (%s); clearing compiled-model cache at %s and retrying once",
                cfg.device, first_err, cfg.cache_dir,
            )
            self._clear_cache_dir(cfg.cache_dir)
            start = time.perf_counter()
            try:
                compiled = self.core.compile_model(model, self.compile_device, config)
            except Exception as second_err:
                raise RuntimeError(f"failed to compile model on device '{cfg.device}': {second_err}") from second_err
        self.compile_time_ms = (time.perf_counter() - start) * 1000.0
        return compiled

    @staticmethod
    def _clear_cache_dir(cache_dir: str) -> None:
        for path in glob.glob(os.path.join(cache_dir, "*.blob")) + glob.glob(os.path.join(cache_dir, "*.cl_cache")):
            try:
                os.remove(path)
            except OSError:
                pass
        for path in glob.glob(os.path.join(cache_dir, "*")):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def current_execution_devices(self) -> List[str]:
        """Devices running inference right now. Under AUTO this starts as "(CPU)" (CPU serving requests while
        the accelerator compiles) and later switches to the selected device, so callers may re-query."""
        devs = self._safe_property("EXECUTION_DEVICES", [])
        if isinstance(devs, str):  # some plugins return a plain string, e.g. "NPU"
            return [devs]
        return [str(d) for d in devs]

    def _safe_property(self, name: str, default: Any) -> Any:
        try:
            return self.compiled.get_property(name)
        except Exception:
            return default

    # -------------------------------------------------------------- inference
    def infer(self, input_tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._closed or self.compiled is None:
            raise RuntimeError("OpenVINO engine is closed")
        feeds = self._prepare_inputs(input_tensors)
        req = self._pool.get()
        start = time.perf_counter()
        try:
            req.infer(feeds)
            outputs: Dict[str, np.ndarray] = {}
            for i, spec in enumerate(self.outputs):
                # Copy out of OpenVINO-owned memory before the request is reused.
                outputs[spec.name] = np.array(req.get_output_tensor(i).data, copy=True)
        except Exception:
            self.stats.record_error()
            raise
        finally:
            self._pool.put(req)
        self.stats.record((time.perf_counter() - start) * 1000.0)
        if self._auto_switch_pending:
            self._note_auto_switch()
        return outputs

    def _note_auto_switch(self) -> None:
        devs = self.current_execution_devices()
        if devs and not any(d.startswith("(") for d in devs):
            self._auto_switch_pending = False
            self.execution_devices = devs
            self.logger.info("AUTO finished compiling for the accelerator; inference now runs on %s", devs)

    @property
    def warming_up_on_cpu(self) -> bool:
        return self._auto_switch_pending

    def _prepare_inputs(self, input_tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        expected = {spec.name: spec for spec in self.inputs}
        provided = dict(input_tensors)
        if len(expected) == 1 and len(provided) == 1 and next(iter(provided)) not in expected:
            only_name = next(iter(expected))
            given_name = next(iter(provided))
            if not self._logged_single_input_alias:
                self.logger.debug("accepting input tensor '%s' as the model's only input '%s'", given_name, only_name)
                self._logged_single_input_alias = True
            provided = {only_name: provided[given_name]}
        unknown = sorted(set(provided) - set(expected))
        missing = sorted(set(expected) - set(provided))
        if unknown or missing:
            raise ValueError(
                f"input tensor names do not match the model: expected {sorted(expected)}, received {sorted(input_tensors)}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {unknown}" if unknown else "")
            )
        feeds: Dict[str, np.ndarray] = {}
        for name, spec in expected.items():
            feeds[name] = self._coerce(name, spec, provided[name])
        return feeds

    def _coerce(self, name: str, spec: TensorSpec, value: Any) -> np.ndarray:
        arr = np.asarray(value)
        target = spec.np_dtype
        if target is not None and arr.dtype != np.dtype(target):
            if not _safe_cast(arr.dtype, np.dtype(target)):
                raise ValueError(
                    f"input tensor '{name}' has dtype {arr.dtype} but the model expects {spec.viam_dtype}; "
                    "refusing an unsafe conversion (configure 'preprocess' to accept this dtype, or convert the tensor before sending)"
                )
            arr = arr.astype(target)
        expected_shape = spec.shape
        if expected_shape != [-1]:  # fully dynamic rank
            if arr.ndim == len(expected_shape) - 1 and expected_shape[0] in (1, -1):
                arr = arr[np.newaxis, ...]
            if arr.ndim != len(expected_shape):
                raise ValueError(
                    f"input tensor '{name}' has shape {list(arr.shape)} but the model expects rank {len(expected_shape)} "
                    f"shape {expected_shape} (-1 = any)"
                )
            for got, want in zip(arr.shape, expected_shape):
                if want != -1 and got != want:
                    raise ValueError(
                        f"input tensor '{name}' has shape {list(arr.shape)} but the model expects {expected_shape} (-1 = any)"
                    )
        return np.ascontiguousarray(arr)

    # ------------------------------------------------------------ diagnostics
    def compiled_properties(self) -> Dict[str, Any]:
        if self.compiled is not None:
            self.execution_devices = self.current_execution_devices()
        out: Dict[str, Any] = {
            "requested_device": self.cfg.device,
            "compile_device": self.compile_device,
            "execution_devices": list(self.execution_devices),
            "loaded_from_cache": self.loaded_from_cache,
            "cache_dir": self.cfg.cache_dir or "",
            "compile_time_ms": self.compile_time_ms,
            "num_requests": self.num_requests,
            "openvino_version": ov.get_version(),
            "model_path": self.cfg.model_path,
            "model_format": self.cfg.model_format,
        }
        if self.compiled is None:
            return out
        props: Dict[str, Any] = {}
        try:
            supported = list(self.compiled.get_property("SUPPORTED_PROPERTIES"))
        except Exception:
            supported = []
        for key in supported:
            key = str(key)
            if key == "SUPPORTED_PROPERTIES":
                continue
            try:
                props[key] = _jsonable(self.compiled.get_property(key))
            except Exception:
                continue
        # Under AUTO/MULTI the interesting properties live on the selected devices.
        device_props = props.get("DEVICE_PROPERTIES")
        if isinstance(device_props, dict):
            props["DEVICE_PROPERTIES"] = {str(k): _jsonable(v) for k, v in device_props.items()}
        out["properties"] = props
        return out

    def benchmark(self, iterations: int = 100, warmup: int = 10, input_shape: Optional[Dict[str, Sequence[int]]] = None) -> Dict[str, Any]:
        feeds: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(0)
        for spec in self.inputs:
            shape = list(spec.shape)
            if input_shape and spec.name in input_shape:
                shape = [int(d) for d in input_shape[spec.name]]
            if any(d < 0 for d in shape):
                raise ValueError(
                    f"input '{spec.name}' has dynamic shape {spec.shape}; pass an 'input_shape' argument, e.g. "
                    f'{{"{spec.name}": [1, 640, 640, 3]}}'
                )
            dtype = spec.np_dtype or np.float32
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                feeds[spec.name] = rng.integers(max(info.min, 0), min(info.max, 255) + 1, size=shape, dtype=dtype)
            elif dtype == np.bool_:
                feeds[spec.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
            else:
                feeds[spec.name] = rng.random(size=shape, dtype=np.float32).astype(dtype)
        for _ in range(max(0, warmup)):
            self.infer(feeds)
        latencies = []
        wall_start = time.perf_counter()
        for _ in range(max(1, iterations)):
            t = time.perf_counter()
            self.infer(feeds)
            latencies.append((time.perf_counter() - t) * 1000.0)
        wall = time.perf_counter() - wall_start
        result: Dict[str, Any] = summarize_ms(np.asarray(latencies))
        self.execution_devices = self.current_execution_devices()
        result.update({
            "iterations": len(latencies),
            "warmup": max(0, warmup),
            "throughput_fps": len(latencies) / wall if wall > 0 else 0.0,
            "device": self.cfg.device,
            "execution_devices": list(self.execution_devices),
            "input_shape": {k: list(v.shape) for k, v in feeds.items()},
        })
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Drain the pool so requests are released before the compiled model.
        while True:
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        self.compiled = None


def _safe_cast(src: np.dtype, dst: np.dtype) -> bool:
    if src == dst:
        return True
    if np.can_cast(src, dst, casting="safe"):
        return True
    # Allow float narrowing (f64 -> f32 -> f16) and int -> float: common and lossless enough for inference.
    if np.issubdtype(dst, np.floating) and (np.issubdtype(src, np.floating) or np.issubdtype(src, np.integer)):
        return True
    return False


def _jsonable(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)
