import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from models import registry
from models.diagnostics import OpenVINODiagnostics
from models.openvino_mlmodel import OpenVINOMLModel


def ml_config(name, **attrs):
    return ComponentConfig(name=name, api="rdk:service:mlmodel", model="erh:openvino:mlmodel", attributes=dict_to_struct(attrs))


def diag_config(name="diag", **attrs):
    return ComponentConfig(name=name, api="rdk:service:generic", model="erh:openvino:diagnostics", attributes=dict_to_struct(attrs))


async def test_registry_lifecycle(tiny_ir, tmp_path):
    ml = OpenVINOMLModel.new(ml_config("ov1", model_path=tiny_ir, device="CPU", cache_dir=str(tmp_path)), {})
    assert registry.get("ov1") is ml and "ov1" in registry.names()
    await ml.close()
    assert registry.get("ov1") is None


async def test_diagnostics_routes_commands(tiny_ir, tmp_path):
    ml = OpenVINOMLModel.new(ml_config("ov2", model_path=tiny_ir, device="CPU", cache_dir=str(tmp_path)), {})
    try:
        assert OpenVINODiagnostics.validate_config(diag_config(mlmodel_name="ov2")) == ([], ["ov2"])
        assert OpenVINODiagnostics.validate_config(diag_config()) == ([], [])
        with pytest.raises(ValueError, match="mlmodel_name"):
            OpenVINODiagnostics.validate_config(diag_config(mlmodel_name=""))
        d = OpenVINODiagnostics.new(diag_config(mlmodel_name="ov2"), {})
        assert "CPU" in [x["name"] for x in (await d.do_command({"command": "list_devices"}))["devices"]]
        assert (await d.do_command({"command": "get_compiled_properties"}))["execution_devices"] == ["CPU"]
        bench = await d.do_command({"command": "benchmark", "iterations": 2, "warmup": 0})
        assert bench["iterations"] == 2
        assert (await d.do_command({"command": "stats"}))["total_calls"] == 2
        # Per-command override of the target mlmodel.
        with pytest.raises(ValueError, match="not a erh:openvino:mlmodel running in this module"):
            await d.do_command({"command": "stats", "mlmodel_name": "elsewhere"})
        with pytest.raises(ValueError, match="unknown command"):
            await d.do_command({"command": "nope"})
        # Without a configured mlmodel, only list_devices works.
        d2 = OpenVINODiagnostics.new(diag_config(), {})
        assert (await d2.do_command({"command": "list_devices"}))["openvino_version"]
        with pytest.raises(ValueError, match="needs an mlmodel"):
            await d2.do_command({"command": "stats"})
    finally:
        await ml.close()
