import os

import pytest

from models.config import MLModelConfig, PreprocessConfig, detect_model_format, parse_device


def test_minimal_config(tiny_ir):
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir})
    assert cfg.device == "AUTO"
    assert cfg.performance_hint == "LATENCY"
    assert cfg.model_format == "OpenVINO IR"
    assert cfg.model_name == "tiny"
    assert cfg.cache_dir.endswith("ov_cache")
    assert cfg.preprocess is None


def test_model_path_required():
    with pytest.raises(ValueError, match="model_path is required"):
        MLModelConfig.from_attributes({})


def test_model_path_missing(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        MLModelConfig.from_attributes({"model_path": str(tmp_path / "nope.onnx")})


def test_unsupported_extension(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported extension"):
        MLModelConfig.from_attributes({"model_path": str(p)})


def test_ir_requires_bin(tmp_path):
    p = tmp_path / "model.xml"
    p.write_text("<net/>")
    with pytest.raises(ValueError, match=r"sibling .*\.bin.* is missing"):
        MLModelConfig.from_attributes({"model_path": str(p)})


def test_saved_model_dir(tmp_path):
    d = tmp_path / "saved"
    d.mkdir()
    with pytest.raises(ValueError, match="saved_model.pb"):
        detect_model_format(str(d))
    (d / "saved_model.pb").write_bytes(b"")
    assert detect_model_format(str(d)) == "TensorFlow SavedModel"


def test_label_path_missing(tiny_ir, tmp_path):
    with pytest.raises(ValueError, match="label_path .* does not exist"):
        MLModelConfig.from_attributes({"model_path": tiny_ir, "label_path": str(tmp_path / "l.txt")})


@pytest.mark.parametrize("device,mode,devices", [
    ("CPU", None, ["CPU"]),
    ("gpu.1", None, ["GPU.1"]),
    ("AUTO", "AUTO", []),
    ("AUTO:GPU,CPU", "AUTO", ["GPU", "CPU"]),
    ("HETERO:GPU,CPU", "HETERO", ["GPU", "CPU"]),
    ("MULTI:GPU(1),CPU(2)", "MULTI", ["GPU", "CPU"]),
    ("NPU", None, ["NPU"]),
])
def test_parse_device(device, mode, devices):
    parsed = parse_device(device)
    assert parsed.mode == mode
    assert parsed.devices == devices


@pytest.mark.parametrize("device", ["", "FOO:CPU", "GPU:CPU", "gpu.x", "AUTO:"])
def test_parse_device_invalid(device):
    with pytest.raises(ValueError):
        parse_device(device)


def test_bad_performance_hint(tiny_ir):
    with pytest.raises(ValueError, match="performance_hint"):
        MLModelConfig.from_attributes({"model_path": tiny_ir, "performance_hint": "FAST"})


def test_bad_precision(tiny_ir):
    with pytest.raises(ValueError, match="inference_precision"):
        MLModelConfig.from_attributes({"model_path": tiny_ir, "inference_precision": "int8"})
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir, "inference_precision": "F16"})
    assert cfg.inference_precision == "f16"


def test_num_requests_and_threads(tiny_ir):
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir, "num_requests": 4.0, "num_threads": 2})
    assert cfg.num_requests == 4 and cfg.num_threads == 2
    for key in ("num_requests", "num_threads"):
        with pytest.raises(ValueError, match=key):
            MLModelConfig.from_attributes({"model_path": tiny_ir, key: 0})


def test_cache_dir_disable(tiny_ir):
    assert MLModelConfig.from_attributes({"model_path": tiny_ir, "cache_dir": ""}).cache_dir is None
    assert MLModelConfig.from_attributes({"model_path": tiny_ir, "cache_dir": "/tmp/x"}).cache_dir == "/tmp/x"


def test_default_cache_dir_uses_module_data(tiny_ir, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", "/data/mod")
    assert MLModelConfig.from_attributes({"model_path": tiny_ir}).cache_dir == os.path.join("/data/mod", "ov_cache")


def test_input_shape(tiny_ir):
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir, "input_shape": {"images": [1.0, 3.0, 640.0, 640.0]}})
    assert cfg.input_shape == {"images": [1, 3, 640, 640]}
    with pytest.raises(ValueError, match="input_shape"):
        MLModelConfig.from_attributes({"model_path": tiny_ir, "input_shape": {"images": [1, 0, 3]}})
    with pytest.raises(ValueError, match="input_shape"):
        MLModelConfig.from_attributes({"model_path": tiny_ir, "input_shape": [1, 3]})


def test_extra_config_ints(tiny_ir):
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir, "extra_config": {"NUM_STREAMS": 2.0, "X": "y"}})
    assert cfg.extra_config == {"NUM_STREAMS": 2, "X": "y"}


def test_preprocess_valid():
    pp = PreprocessConfig.from_attributes({
        "tensor_element_type": "u8", "tensor_layout": "nhwc", "tensor_color_format": "bgr",
        "model_layout": "NCHW", "model_color_format": "RGB", "resize": "linear",
        "mean": [0, 0, 0], "scale": [255, 255, 255],
    })
    assert pp.tensor_layout == "NHWC" and pp.tensor_color_format == "BGR" and pp.enabled


@pytest.mark.parametrize("attrs,msg", [
    ({"tensor_element_type": "float"}, "tensor_element_type"),
    ({"resize": "bicubic"}, "resize"),
    ({"tensor_color_format": "RGB"}, "must be set together"),
    ({"tensor_color_format": "RGB", "model_color_format": "GRAY"}, "model_color_format"),
    ({"resize": "linear"}, "tensor_layout and preprocess.model_layout are required"),
    ({"mean": [0, 0], "scale": [1]}, "same length"),
    ({"scale": [0, 1, 1]}, "zeros"),
    ({"tensor_layout": "N!HWC", "model_layout": "NCHW"}, "layout"),
    ({"bogus": 1}, "unknown field"),
    ({"mean": "x"}, "mean"),
])
def test_preprocess_invalid(attrs, msg):
    with pytest.raises(ValueError, match=msg):
        PreprocessConfig.from_attributes(attrs)


def test_preprocess_attached_to_config(tiny_ir):
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir, "preprocess": {"tensor_element_type": "u8"}})
    assert cfg.preprocess is not None and cfg.preprocess.enabled
