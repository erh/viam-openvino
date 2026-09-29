import concurrent.futures
import logging
import os

import numpy as np
import pytest

from models.config import MLModelConfig
from models.engine import OpenVINOEngine, list_devices

LOG = logging.getLogger("test")


def make_engine(model_path, tmp_path, **attrs):
    attrs = {"model_path": model_path, "device": "CPU", "cache_dir": str(tmp_path / "cache"), **attrs}
    engine = OpenVINOEngine(MLModelConfig.from_attributes(attrs), LOG)
    engine.load()
    return engine


def test_ir_and_onnx_match(tiny_ir, tiny_onnx, tmp_path):
    x = np.random.default_rng(0).random((1, 3, 8, 8), dtype=np.float32)
    # Pin f32: CPUs with native bf16 (AMX / AVX512_BF16) default to bf16 inference, which is not bit-comparable.
    e1 = make_engine(tiny_ir, tmp_path / "a", inference_precision="f32")
    e2 = make_engine(tiny_onnx, tmp_path / "b", inference_precision="f32")
    try:
        o1 = e1.infer({"images": x})
        o2 = e2.infer({"images": x})
        assert set(o1) == {"output0"} and set(o2) == {"output0"}
        np.testing.assert_allclose(o1["output0"], o2["output0"], rtol=1e-4, atol=1e-5)
        assert e1.cfg.model_format == "OpenVINO IR" and e2.cfg.model_format == "ONNX"
    finally:
        e1.close()
        e2.close()


def test_tensor_specs(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path)
    try:
        assert [(t.name, t.viam_dtype, t.shape) for t in e.inputs] == [("images", "float32", [1, 3, 8, 8])]
        assert [(t.name, t.viam_dtype, t.shape) for t in e.outputs] == [("output0", "float32", [1, 4, 8, 8])]
        assert e.execution_devices == ["CPU"]
        assert e.num_requests >= 1
    finally:
        e.close()


def test_preprocess_changes_caller_facing_spec(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path, preprocess={
        "tensor_element_type": "u8", "tensor_layout": "NHWC", "tensor_color_format": "BGR",
        "model_layout": "NCHW", "model_color_format": "RGB", "resize": "linear",
        "mean": [0, 0, 0], "scale": [255, 255, 255],
    })
    try:
        spec = e.inputs[0]
        assert spec.viam_dtype == "uint8"
        assert spec.shape == [1, -1, -1, 3]
        # A larger uint8 NHWC image is resized on device and produces the model's fixed output.
        img = np.random.default_rng(0).integers(0, 255, (1, 32, 24, 3), dtype=np.uint8)
        out = e.infer({"images": img})["output0"]
        assert out.shape == (1, 4, 8, 8) and out.dtype == np.float32
        # Batch-less HWC is accepted too.
        out2 = e.infer({"images": img[0]})["output0"]
        assert out2.shape == (1, 4, 8, 8)
    finally:
        e.close()


def test_preprocess_no_resize_keeps_static_spatial(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path, preprocess={
        "tensor_element_type": "u8", "tensor_layout": "NHWC", "model_layout": "NCHW", "scale": [255],
    })
    try:
        assert e.inputs[0].shape == [1, 8, 8, 3] and e.inputs[0].viam_dtype == "uint8"
    finally:
        e.close()


def test_preprocess_wrong_input_name(tiny_ir, tmp_path):
    with pytest.raises(ValueError, match="not a model input"):
        make_engine(tiny_ir, tmp_path, preprocess={"input_name": "img", "tensor_element_type": "u8"})


def test_input_shape_reshape(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path, input_shape={"images": [1, 3, 16, 16]})
    try:
        assert e.inputs[0].shape == [1, 3, 16, 16]
        assert e.outputs[0].shape == [1, 4, 16, 16]
    finally:
        e.close()
    with pytest.raises(ValueError, match="unknown input"):
        make_engine(tiny_ir, tmp_path, input_shape={"nope": [1, 3, 8, 8]})


def test_input_name_mismatch_errors(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path)
    try:
        x = np.zeros((1, 3, 8, 8), dtype=np.float32)
        # Single input under a different name is accepted.
        assert "output0" in e.infer({"image": x})
        with pytest.raises(ValueError, match=r"expected \['images'\]"):
            e.infer({"a": x, "b": x})
        with pytest.raises(ValueError, match="expected"):
            e.infer({})
    finally:
        e.close()


def test_dtype_conversions(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path)
    try:
        u8 = np.zeros((1, 3, 8, 8), dtype=np.uint8)
        e.infer({"images": u8})  # uint8 -> float32 is safe
        e.infer({"images": u8.astype(np.float64)})  # float64 -> float32 narrowing allowed
    finally:
        e.close()


def test_unsafe_dtype_rejected(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path, preprocess={"tensor_element_type": "u8", "tensor_layout": "NCHW", "model_layout": "NCHW"})
    try:
        assert e.inputs[0].viam_dtype == "uint8"
        with pytest.raises(ValueError, match="dtype float32 but the model expects uint8"):
            e.infer({"images": np.zeros((1, 3, 8, 8), dtype=np.float32)})
    finally:
        e.close()


def test_shape_mismatch_error_names_tensor(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path)
    try:
        with pytest.raises(ValueError, match=r"'images' has shape \[1, 3, 9, 9\] but the model expects \[1, 3, 8, 8\]"):
            e.infer({"images": np.zeros((1, 3, 9, 9), dtype=np.float32)})
        with pytest.raises(ValueError, match="rank"):
            e.infer({"images": np.zeros((8, 8), dtype=np.float32)})
    finally:
        e.close()


def test_concurrent_infer_matches_serial(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path, num_requests=4)
    try:
        rng = np.random.default_rng(7)
        xs = [rng.random((1, 3, 8, 8), dtype=np.float32) for _ in range(32)]
        serial = [e.infer({"images": x})["output0"] for x in xs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(lambda x: e.infer({"images": x})["output0"], xs))
        for a, b in zip(serial, parallel):
            np.testing.assert_array_equal(a, b)
        # Outputs are private copies: mutating one must not affect a later call.
        parallel[0][...] = -1
        np.testing.assert_array_equal(e.infer({"images": xs[0]})["output0"], serial[0])
        assert e.stats.snapshot()["total_calls"] == 65
    finally:
        e.close()


def test_cache_hit_on_second_load(tiny_ir, tmp_path):
    cache = tmp_path / "cache"
    e1 = make_engine(tiny_ir, tmp_path)
    assert e1.loaded_from_cache is False
    assert any(f.endswith(".blob") for f in os.listdir(cache))
    e1.close()
    e2 = make_engine(tiny_ir, tmp_path)
    assert e2.loaded_from_cache is True
    e2.close()


def test_corrupt_cache_is_recovered(tiny_ir, tmp_path, caplog):
    cache = tmp_path / "cache"
    e1 = make_engine(tiny_ir, tmp_path)
    e1.close()
    for f in os.listdir(cache):
        if f.endswith(".blob"):
            os.chmod(cache / f, 0o644)
            with open(cache / f, "wb") as fh:
                fh.write(b"garbage" * 100)
    # Either OpenVINO detects the corruption itself and recompiles, or our retry path clears the cache.
    e2 = make_engine(tiny_ir, tmp_path)
    try:
        out = e2.infer({"images": np.zeros((1, 3, 8, 8), dtype=np.float32)})
        assert out["output0"].shape == (1, 4, 8, 8)
    finally:
        e2.close()


def test_cache_disabled(tiny_ir, tmp_path):
    e = OpenVINOEngine(MLModelConfig.from_attributes({"model_path": tiny_ir, "device": "CPU", "cache_dir": ""}), LOG)
    e.load()
    try:
        assert e.cfg.cache_dir is None
        assert "cache" not in {k.lower() for k in e.compiled_properties()["properties"]}
    finally:
        e.close()


def test_explicit_missing_accelerator_fails(tiny_ir, tmp_path, caplog):
    import openvino as ov
    if "NPU" in ov.Core().available_devices:
        pytest.skip("NPU present on this machine")
    cfg = MLModelConfig.from_attributes({"model_path": tiny_ir, "device": "NPU", "cache_dir": str(tmp_path)})
    e = OpenVINOEngine(cfg, LOG)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="NPU.*unavailable.*NPU driver"):
            e.load()
    assert any("NPU driver" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)


def test_auto_with_missing_candidate_warns_and_continues(tiny_ir, tmp_path, caplog):
    import openvino as ov
    if "NPU" in ov.Core().available_devices:
        pytest.skip("NPU present on this machine")
    with caplog.at_level(logging.WARNING):
        e = make_engine(tiny_ir, tmp_path, device="AUTO:NPU,CPU")
    try:
        assert e.execution_devices == ["CPU"]
        assert e.compile_device == "AUTO:CPU"
        assert any("Continuing with whatever AUTO selects" in r.getMessage() for r in caplog.records)
    finally:
        e.close()


def test_benchmark_and_properties(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path)
    try:
        r = e.benchmark(iterations=5, warmup=1)
        assert r["iterations"] == 5 and r["throughput_fps"] > 0 and r["p50_ms"] >= 0
        assert r["input_shape"] == {"images": [1, 3, 8, 8]}
        props = e.compiled_properties()
        assert props["execution_devices"] == ["CPU"] and props["loaded_from_cache"] is False
        assert "PERFORMANCE_HINT" in props["properties"]
    finally:
        e.close()


def test_benchmark_dynamic_requires_shape(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path, preprocess={
        "tensor_element_type": "u8", "tensor_layout": "NHWC", "model_layout": "NCHW", "resize": "linear",
    })
    try:
        with pytest.raises(ValueError, match="input_shape"):
            e.benchmark(iterations=1, warmup=0)
        r = e.benchmark(iterations=2, warmup=0, input_shape={"images": [1, 16, 16, 3]})
        assert r["input_shape"] == {"images": [1, 16, 16, 3]}
    finally:
        e.close()


def test_list_devices():
    r = list_devices()
    assert r["openvino_version"]
    names = [d["name"] for d in r["devices"]]
    assert "CPU" in names
    cpu = next(d for d in r["devices"] if d["name"] == "CPU")
    assert cpu["full_name"] and isinstance(cpu["capabilities"], list)


def test_closed_engine_errors(tiny_ir, tmp_path):
    e = make_engine(tiny_ir, tmp_path)
    e.close()
    with pytest.raises(RuntimeError, match="closed"):
        e.infer({"images": np.zeros((1, 3, 8, 8), dtype=np.float32)})
