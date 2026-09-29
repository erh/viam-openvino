"""Configuration parsing and validation for ``erh:openvino:mlmodel``.

All validation errors are raised as ``ValueError`` with a message that names
the offending attribute, so that ``validate_config`` fails at config time
rather than at first inference.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

PERFORMANCE_HINTS = ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT")
INFERENCE_PRECISIONS = ("f32", "f16", "bf16")
RESIZE_ALGORITHMS = ("none", "linear", "nearest", "cubic")
COLOR_FORMATS = ("RGB", "BGR")
TENSOR_ELEMENT_TYPES = ("u8", "i8", "u16", "i16", "i32", "i64", "f16", "f32", "f64")

# Extension -> human readable format name. Anything else is a validation error.
MODEL_FORMATS: Dict[str, str] = {
    ".xml": "OpenVINO IR",
    ".onnx": "ONNX",
    ".tflite": "TFLite",
    ".pb": "TensorFlow frozen graph / SavedModel",
    ".pdmodel": "PaddlePaddle",
}

DEVICE_MODES = ("AUTO", "HETERO", "MULTI", "BATCH")
_DEVICE_RE = re.compile(r"^[A-Z][A-Z0-9_]*(\.\d+)?$")
_LAYOUT_RE = re.compile(r"^\[?[A-Z?.,]+\]?$")


def default_cache_dir() -> str:
    base = os.environ.get("VIAM_MODULE_DATA")
    if not base:
        base = os.path.join(tempfile.gettempdir(), "viam-openvino")
    return os.path.join(base, "ov_cache")


def detect_model_format(model_path: str) -> str:
    """Return the human readable format for ``model_path`` or raise ValueError."""
    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, "saved_model.pb")):
            return "TensorFlow SavedModel"
        raise ValueError(
            f"model_path '{model_path}' is a directory but does not contain saved_model.pb; "
            "only TensorFlow SavedModel directories are supported"
        )
    ext = os.path.splitext(model_path)[1].lower()
    if ext not in MODEL_FORMATS:
        raise ValueError(
            f"model_path '{model_path}' has unsupported extension '{ext or '(none)'}'; "
            f"supported: {', '.join(sorted(MODEL_FORMATS))}"
        )
    return MODEL_FORMATS[ext]


@dataclass
class ParsedDevice:
    raw: str
    mode: Optional[str]  # AUTO / HETERO / MULTI / BATCH or None for a plain device
    devices: List[str]  # base device names referenced, e.g. ["GPU.0", "CPU"]

    @property
    def is_auto(self) -> bool:
        return self.mode == "AUTO"


def parse_device(device: str) -> ParsedDevice:
    """Parse an OpenVINO device string such as ``GPU.1`` or ``AUTO:GPU,CPU``."""
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string such as 'CPU', 'GPU', 'NPU' or 'AUTO'")
    raw = device.strip().upper()
    mode: Optional[str] = None
    body = raw
    if ":" in raw:
        prefix, body = raw.split(":", 1)
        if prefix not in DEVICE_MODES:
            raise ValueError(
                f"device '{device}' has unknown prefix '{prefix}'; expected one of {', '.join(DEVICE_MODES)}"
            )
        mode = prefix
        if not body.strip():
            raise ValueError(f"device '{device}' has no devices after '{prefix}:'; use '{prefix}' alone or list devices")
    elif raw in DEVICE_MODES:
        mode = raw
        body = ""
    devices: List[str] = []
    if body:
        for part in body.split(","):
            name = re.sub(r"\(.*\)$", "", part.strip())  # strip e.g. GPU(1) priorities
            if not _DEVICE_RE.match(name):
                raise ValueError(f"device '{device}' contains an unparseable device name '{part}'")
            devices.append(name)
    elif mode is None:
        raise ValueError(f"device '{device}' could not be parsed")
    return ParsedDevice(raw=raw, mode=mode, devices=devices)


@dataclass
class PreprocessConfig:
    input_name: Optional[str] = None
    tensor_element_type: Optional[str] = None
    tensor_layout: Optional[str] = None
    tensor_color_format: Optional[str] = None
    model_layout: Optional[str] = None
    model_color_format: Optional[str] = None
    resize: str = "none"
    mean: Optional[List[float]] = None
    scale: Optional[List[float]] = None

    @classmethod
    def from_attributes(cls, attrs: Mapping[str, Any]) -> "PreprocessConfig":
        if not isinstance(attrs, Mapping):
            raise ValueError("preprocess must be an object")
        known = {
            "input_name", "tensor_element_type", "tensor_layout", "tensor_color_format",
            "model_layout", "model_color_format", "resize", "mean", "scale",
        }
        unknown = sorted(set(attrs) - known)
        if unknown:
            raise ValueError(f"preprocess has unknown field(s): {', '.join(unknown)}")

        def opt_str(key: str) -> Optional[str]:
            v = attrs.get(key)
            if v is None:
                return None
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"preprocess.{key} must be a non-empty string")
            return v.strip()

        def opt_floats(key: str) -> Optional[List[float]]:
            v = attrs.get(key)
            if v is None:
                return None
            if not isinstance(v, (list, tuple)) or not v:
                raise ValueError(f"preprocess.{key} must be a non-empty list of numbers")
            out: List[float] = []
            for x in v:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise ValueError(f"preprocess.{key} must contain only numbers")
                out.append(float(x))
            return out

        cfg = cls(
            input_name=opt_str("input_name"),
            tensor_element_type=opt_str("tensor_element_type"),
            tensor_layout=opt_str("tensor_layout"),
            tensor_color_format=opt_str("tensor_color_format"),
            model_layout=opt_str("model_layout"),
            model_color_format=opt_str("model_color_format"),
            resize=(opt_str("resize") or "none").lower(),
            mean=opt_floats("mean"),
            scale=opt_floats("scale"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.tensor_element_type is not None and self.tensor_element_type not in TENSOR_ELEMENT_TYPES:
            raise ValueError(
                f"preprocess.tensor_element_type '{self.tensor_element_type}' is not one of "
                f"{', '.join(TENSOR_ELEMENT_TYPES)}"
            )
        for key in ("tensor_layout", "model_layout"):
            v = getattr(self, key)
            if v is not None:
                if not _LAYOUT_RE.match(v.upper()):
                    raise ValueError(f"preprocess.{key} '{v}' is not a valid OpenVINO layout (e.g. NHWC, NCHW)")
                setattr(self, key, v.upper())
        for key in ("tensor_color_format", "model_color_format"):
            v = getattr(self, key)
            if v is not None:
                if v.upper() not in COLOR_FORMATS:
                    raise ValueError(f"preprocess.{key} '{v}' must be one of {', '.join(COLOR_FORMATS)}")
                setattr(self, key, v.upper())
        if self.resize not in RESIZE_ALGORITHMS:
            raise ValueError(f"preprocess.resize '{self.resize}' must be one of {', '.join(RESIZE_ALGORITHMS)}")
        if (self.tensor_color_format is None) != (self.model_color_format is None):
            raise ValueError(
                "preprocess.tensor_color_format and preprocess.model_color_format must be set together"
            )
        if self.resize != "none" or self.tensor_color_format is not None:
            if self.tensor_layout is None or self.model_layout is None:
                raise ValueError(
                    "preprocess.tensor_layout and preprocess.model_layout are required when resize or a "
                    "color format is configured"
                )
        if self.mean is not None and self.scale is not None and len(self.mean) != len(self.scale):
            raise ValueError("preprocess.mean and preprocess.scale must have the same length")
        if self.scale is not None and any(s == 0 for s in self.scale):
            raise ValueError("preprocess.scale must not contain zeros")

    @property
    def enabled(self) -> bool:
        return any(
            v is not None for v in (
                self.tensor_element_type, self.tensor_layout, self.tensor_color_format,
                self.model_layout, self.mean, self.scale,
            )
        ) or self.resize != "none"


@dataclass
class MLModelConfig:
    model_path: str
    label_path: Optional[str] = None
    device: str = "AUTO"
    performance_hint: str = "LATENCY"
    num_requests: Optional[int] = None
    inference_precision: Optional[str] = None
    cache_dir: Optional[str] = field(default_factory=default_cache_dir)
    num_threads: Optional[int] = None
    input_shape: Dict[str, List[int]] = field(default_factory=dict)
    preprocess: Optional[PreprocessConfig] = None
    extra_config: Dict[str, Any] = field(default_factory=dict)

    # populated by the parser
    model_format: str = ""
    parsed_device: ParsedDevice = field(default_factory=lambda: parse_device("AUTO"))

    @classmethod
    def from_attributes(cls, attrs: Mapping[str, Any]) -> "MLModelConfig":
        attrs = dict(attrs or {})

        model_path = attrs.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path is required and must be a non-empty string")
        model_path = os.path.expanduser(model_path.strip())
        if not os.path.exists(model_path):
            raise ValueError(f"model_path '{model_path}' does not exist")
        model_format = detect_model_format(model_path)
        if model_path.lower().endswith(".xml"):
            bin_path = os.path.splitext(model_path)[0] + ".bin"
            if not os.path.isfile(bin_path):
                raise ValueError(f"model_path '{model_path}' is OpenVINO IR but its sibling '{bin_path}' is missing")

        label_path = attrs.get("label_path")
        if label_path is not None:
            if not isinstance(label_path, str) or not label_path.strip():
                raise ValueError("label_path must be a non-empty string when set")
            label_path = os.path.expanduser(label_path.strip())
            if not os.path.isfile(label_path):
                raise ValueError(f"label_path '{label_path}' does not exist")

        device_attr = attrs.get("device", "AUTO")
        parsed_device = parse_device(device_attr)

        hint = attrs.get("performance_hint", "LATENCY")
        if not isinstance(hint, str) or hint.upper() not in PERFORMANCE_HINTS:
            raise ValueError(f"performance_hint '{hint}' must be one of {', '.join(PERFORMANCE_HINTS)}")
        hint = hint.upper()

        num_requests = _opt_positive_int(attrs, "num_requests")
        num_threads = _opt_positive_int(attrs, "num_threads")

        precision = attrs.get("inference_precision")
        if precision is not None:
            if not isinstance(precision, str) or precision.lower() not in INFERENCE_PRECISIONS:
                raise ValueError(
                    f"inference_precision '{precision}' must be one of {', '.join(INFERENCE_PRECISIONS)}"
                )
            precision = precision.lower()

        cache_dir: Optional[str]
        if "cache_dir" in attrs:
            v = attrs["cache_dir"]
            if v is None or (isinstance(v, str) and v.strip() == ""):
                cache_dir = None
            elif isinstance(v, str):
                cache_dir = os.path.expanduser(v.strip())
            else:
                raise ValueError("cache_dir must be a string")
        else:
            cache_dir = default_cache_dir()

        input_shape: Dict[str, List[int]] = {}
        raw_shape = attrs.get("input_shape")
        if raw_shape is not None:
            if not isinstance(raw_shape, Mapping):
                raise ValueError('input_shape must be an object mapping input name to a list of ints, e.g. {"images": [1, 3, 640, 640]}')
            for name, dims in raw_shape.items():
                if not isinstance(dims, (list, tuple)) or not dims:
                    raise ValueError(f"input_shape['{name}'] must be a non-empty list of ints")
                parsed_dims: List[int] = []
                for d in dims:
                    if isinstance(d, bool) or not isinstance(d, (int, float)) or int(d) != d or int(d) < -1 or int(d) == 0:
                        raise ValueError(f"input_shape['{name}'] contains invalid dimension {d!r}; use positive ints or -1")
                    parsed_dims.append(int(d))
                input_shape[str(name)] = parsed_dims

        preprocess = None
        if attrs.get("preprocess") is not None:
            preprocess = PreprocessConfig.from_attributes(attrs["preprocess"])

        extra_config = attrs.get("extra_config") or {}
        if not isinstance(extra_config, Mapping):
            raise ValueError("extra_config must be an object of OpenVINO properties")
        extra_config = {str(k): _plain(v) for k, v in extra_config.items()}

        return cls(
            model_path=model_path,
            label_path=label_path,
            device=parsed_device.raw,
            performance_hint=hint,
            num_requests=num_requests,
            inference_precision=precision,
            cache_dir=cache_dir,
            num_threads=num_threads,
            input_shape=input_shape,
            preprocess=preprocess,
            extra_config=extra_config,
            model_format=model_format,
            parsed_device=parsed_device,
        )

    @property
    def model_name(self) -> str:
        base = os.path.basename(os.path.normpath(self.model_path))
        return os.path.splitext(base)[0] if not os.path.isdir(self.model_path) else base


def _opt_positive_int(attrs: Mapping[str, Any], key: str) -> Optional[int]:
    v = attrs.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < 1:
        raise ValueError(f"{key} must be a positive integer")
    return int(v)


def _plain(v: Any) -> Any:
    """Convert protobuf Struct values (floats for ints) into plain Python values."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def load_labels(label_path: Optional[str]) -> List[str]:
    if not label_path:
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
