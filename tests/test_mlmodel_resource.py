import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct, struct_to_dict

from models.openvino_mlmodel import OpenVINOMLModel


def make_config(name="ov", **attrs):
    return ComponentConfig(name=name, api="rdk:service:mlmodel", model="erh:openvino:mlmodel", attributes=dict_to_struct(attrs))


@pytest.fixture
async def resource(tiny_ir, labels_file, tmp_path):
    cfg = make_config(model_path=tiny_ir, label_path=labels_file, device="CPU", cache_dir=str(tmp_path / "cache"))
    assert OpenVINOMLModel.validate_config(cfg) == ([], [])
    r = OpenVINOMLModel.new(cfg, {})
    yield r
    await r.close()


def test_validate_config_error(tmp_path):
    with pytest.raises(ValueError, match="model_path"):
        OpenVINOMLModel.validate_config(make_config(model_path=str(tmp_path / "x.onnx")))


async def test_metadata_matches_reference_conventions(resource, tiny_ir, labels_file):
    md = await resource.metadata()
    assert md.name == "tiny"
    assert md.type == "openvino"
    assert "OpenVINO" in md.description and tiny_ir in md.description and "device=CPU" in md.description
    assert [(t.name, t.data_type, list(t.shape)) for t in md.input_info] == [("images", "float32", [1, 3, 8, 8])]
    assert [(t.name, t.data_type, list(t.shape)) for t in md.output_info] == [("output0", "float32", [1, 4, 8, 8])]
    assert struct_to_dict(md.output_info[0].extra) == {"labels": labels_file}
    assert not md.input_info[0].HasField("extra")


async def test_metadata_without_labels(tiny_ir, tmp_path):
    r = OpenVINOMLModel.new(make_config(model_path=tiny_ir, device="CPU", cache_dir=""), {})
    try:
        md = await r.metadata()
        assert not md.output_info[0].HasField("extra")
    finally:
        await r.close()


async def test_infer(resource):
    x = np.random.default_rng(0).random((1, 3, 8, 8), dtype=np.float32)
    out = await resource.infer({"images": x})
    assert out["output0"].shape == (1, 4, 8, 8)


async def test_do_commands(resource):
    devices = await resource.do_command({"command": "list_devices"})
    assert "CPU" in [d["name"] for d in devices["devices"]]
    props = await resource.do_command({"command": "get_compiled_properties"})
    assert props["execution_devices"] == ["CPU"]
    bench = await resource.do_command({"command": "benchmark", "iterations": 3.0, "warmup": 1.0})
    assert bench["iterations"] == 3 and "p99_ms" in bench and bench["device"] == "CPU"
    stats = await resource.do_command({"command": "stats"})
    assert stats["total_calls"] == 4 and stats["p50_ms"] > 0
    with pytest.raises(ValueError, match="unknown command"):
        await resource.do_command({"command": "nope"})
    with pytest.raises(ValueError, match="'command'"):
        await resource.do_command({})


async def test_reconfigure_rebuilds_and_hits_cache(tiny_ir, tiny_onnx, tmp_path):
    cache = str(tmp_path / "cache")
    r = OpenVINOMLModel.new(make_config(model_path=tiny_ir, device="CPU", cache_dir=cache), {})
    try:
        assert r.engine.loaded_from_cache is False
        r.reconfigure(make_config(model_path=tiny_ir, device="CPU", cache_dir=cache, performance_hint="THROUGHPUT"), {})
        assert r.cfg.performance_hint == "THROUGHPUT"
        r.reconfigure(make_config(model_path=tiny_ir, device="CPU", cache_dir=cache), {})
        assert r.engine.loaded_from_cache is True
        md = await r.metadata()
        assert md.name == "tiny"
        r.reconfigure(make_config(model_path=tiny_onnx, device="CPU", cache_dir=cache), {})
        assert (await r.metadata()).description.count("ONNX") == 1
    finally:
        await r.close()
