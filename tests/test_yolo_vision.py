import io
from typing import Dict, List, Optional

import numpy as np
import pytest
from PIL import Image
from viam.components.camera import Camera
from viam.errors import NotSupportedError
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata
from viam.services.mlmodel import Metadata, MLModel, TensorInfo
from viam.utils import dict_to_struct

from models.openvino_mlmodel import OpenVINOMLModel
from models.yolo_vision import YoloConfig, YoloVision


def make_config(name="yolo", **attrs):
    return ComponentConfig(name=name, api="rdk:service:vision", model="erh:openvino:yolo", attributes=dict_to_struct(attrs))


def make_image(w: int, h: int, mime=CameraMimeType.JPEG) -> ViamImage:
    img = Image.new("RGB", (w, h), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if mime == CameraMimeType.JPEG else "PNG")
    return ViamImage(buf.getvalue(), mime)


class FakeCamera(Camera):
    def __init__(self, name: str, image: ViamImage):
        super().__init__(name)
        self.image = image
        self.calls = 0

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        self.calls += 1
        return [NamedImage("color", self.image.data, self.image.mime_type)], ResponseMetadata()

    async def get_point_cloud(self, *, extra=None, timeout=None, **kwargs):
        raise NotImplementedError

    async def get_properties(self, *, timeout=None, **kwargs):
        raise NotImplementedError


class FakeMLModel(MLModel):
    """Returns a fixed tensor and records inputs; input [1,3,H,W] float32."""

    def __init__(self, name: str, output: np.ndarray, shape, label_path: Optional[str] = None, dtype="float32", output_name="output0"):
        super().__init__(name)
        self.output = output
        self.shape = list(shape)
        self.label_path = label_path
        self.dtype = dtype
        self.output_name = output_name
        self.inputs: List[Dict[str, np.ndarray]] = []

    async def infer(self, input_tensors, *, extra=None, timeout=None):
        self.inputs.append(input_tensors)
        return {self.output_name: self.output}

    async def metadata(self, *, extra=None, timeout=None):
        out = TensorInfo(name=self.output_name, data_type="float32", shape=list(self.output.shape))
        if self.label_path:
            out.extra.CopyFrom(dict_to_struct({"labels": self.label_path}))
        return Metadata(name="fake", type="fake", input_info=[TensorInfo(name="images", data_type=self.dtype, shape=self.shape)], output_info=[out])


def v8_output(rows, C):
    out = np.zeros((1, 4 + C, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cid, s) in enumerate(rows):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cid, i] = s
    return out


def deps(ml, cam=None):
    d = {MLModel.get_resource_name(ml.name): ml}
    if cam is not None:
        d[Camera.get_resource_name(cam.name)] = cam
    return d


def test_validate_config(labels_file):
    assert YoloVision.validate_config(make_config(mlmodel_name="m", camera_name="c")) == (["m"], ["c"])
    assert YoloVision.validate_config(make_config(mlmodel_name="m")) == (["m"], [])
    with pytest.raises(ValueError, match="mlmodel_name"):
        YoloVision.validate_config(make_config())
    for bad in (
        {"yolo_version": "v9"}, {"input_size": [640]}, {"confidence_threshold": 2}, {"iou_threshold": -1},
        {"max_detections": 0}, {"class_filter": "cat"}, {"label_path": "/nope/labels.txt"}, {"input_format": "HWC"},
    ):
        with pytest.raises(ValueError):
            YoloConfig({"mlmodel_name": "m", **bad})
    cfg = YoloConfig({"mlmodel_name": "m", "input_size": [320.0, 640.0], "class_filter": ["cat"], "label_path": labels_file})
    assert cfg.input_size == (320, 640) and cfg.class_filter == {"cat"}


async def test_detections_with_letterbox_inversion(labels_file):
    # Model input 64x64; image 128x64 (wide) -> scale 0.5, new size 64x32, pad_y 16.
    # cat box in letterbox coords: cx=32, cy=32, w=20, h=10 -> xyxy [22,27,42,37] -> orig [(22)/0.5, (27-16)/0.5, ...] = [44,22,84,42]
    out = v8_output([(32, 32, 20, 10, 0, 0.9), (33, 32, 20, 10, 0, 0.5), (10, 40, 8, 8, 1, 0.8), (50, 20, 8, 8, 2, 0.1)], C=3)
    ml = FakeMLModel("m", out, [1, 3, 64, 64], label_path=labels_file)
    cam = FakeCamera("cam", make_image(128, 64))
    v = YoloVision.new(make_config(mlmodel_name="m", camera_name="cam"), deps(ml, cam))
    dets = await v.get_detections_from_camera("")
    assert [(d.class_name, d.x_min, d.y_min, d.x_max, d.y_max) for d in dets] == [
        ("cat", 44, 22, 84, 42),
        ("dog", 12, 40, 28, 56),
    ]
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[0].x_min_normalized == pytest.approx(44 / 128) and dets[0].y_max_normalized == pytest.approx(42 / 64)
    sent = ml.inputs[0]["images"]
    assert sent.shape == (1, 3, 64, 64) and sent.dtype == np.float32 and sent.max() <= 1.0
    # Padding rows are 114/255 in the normalized tensor.
    assert sent[0, 0, 0, 0] == pytest.approx(114 / 255)
    assert cam.calls == 1


async def test_input_format_raw_pixels_and_class_filter(labels_file):
    out = v8_output([(32, 32, 20, 10, 0, 0.9), (10, 40, 8, 8, 1, 0.8)], C=3)
    ml = FakeMLModel("m", out, [1, -1, -1, 3], label_path=labels_file, dtype="uint8")
    v = YoloVision.new(make_config(mlmodel_name="m", input_format="NHWC_u8_rgb", input_size=[64, 64], class_filter=["dog"]), deps(ml))
    dets = await v.get_detections(make_image(64, 64))
    assert [d.class_name for d in dets] == ["dog"]
    sent = ml.inputs[0]["images"]
    assert sent.shape == (1, 64, 64, 3) and sent.dtype == np.uint8


async def test_v5_auto_detection_and_no_labels():
    C = 3
    out = np.zeros((1, 12, 5 + C), dtype=np.float32)  # 12 anchors > 5 + C channels
    out[0, 0, :5] = [32, 32, 10, 10, 0.9]
    out[0, 0, 5 + 2] = 0.9
    ml = FakeMLModel("m", out, [1, 3, 64, 64])
    v = YoloVision.new(make_config(mlmodel_name="m"), deps(ml))
    dets = await v.get_detections(make_image(64, 64))
    assert len(dets) == 1 and dets[0].class_name == "2" and dets[0].confidence == pytest.approx(0.81)
    info = await v.do_command({"command": "model_info"})
    assert info["version"] == "v5" and info["labels_count"] == 0


async def test_ambiguous_version_errors(labels_file):
    out = np.zeros((1, 8, 7), dtype=np.float32)  # 3 labels: neither 4+C (7) on dim1 nor 5+C (8) on dim2
    ml = FakeMLModel("m", out, [1, 3, 64, 64], label_path=labels_file)
    v = YoloVision.new(make_config(mlmodel_name="m"), deps(ml))
    with pytest.raises(ValueError, match="ambiguous"):
        await v.get_detections(make_image(64, 64))


async def test_dynamic_input_requires_input_size():
    ml = FakeMLModel("m", np.zeros((1, 7, 4), dtype=np.float32), [1, 3, -1, -1])
    v = YoloVision.new(make_config(mlmodel_name="m"), deps(ml))
    with pytest.raises(ValueError, match="input_size"):
        await v.get_detections(make_image(64, 64))


async def test_capture_all_properties_and_unsupported(labels_file):
    out = v8_output([(32, 32, 20, 10, 0, 0.9)], C=3)
    ml = FakeMLModel("m", out, [1, 3, 64, 64], label_path=labels_file)
    cam = FakeCamera("cam", make_image(64, 64, CameraMimeType.PNG))
    v = YoloVision.new(make_config(mlmodel_name="m", camera_name="cam"), deps(ml, cam))
    res = await v.capture_all_from_camera("cam", return_image=True, return_detections=True)
    assert res.image is not None and res.image.mime_type == CameraMimeType.PNG
    assert len(res.detections) == 1
    res = await v.capture_all_from_camera("cam")
    assert res.image is None and not res.detections
    props = await v.get_properties()
    assert props.detections_supported and not props.classifications_supported and not props.object_point_clouds_supported
    assert props.default_camera == "cam"
    with pytest.raises(NotSupportedError):
        await v.get_classifications(make_image(8, 8), 1)
    with pytest.raises(NotSupportedError):
        await v.get_classifications_from_camera("cam", 1)
    with pytest.raises(NotSupportedError):
        await v.get_object_point_clouds("cam")
    with pytest.raises(ValueError, match="not a configured dependency"):
        await v.get_detections_from_camera("other")


async def test_missing_dependency():
    with pytest.raises(ValueError, match="not found in dependencies"):
        YoloVision.new(make_config(mlmodel_name="m"), {})


async def test_end_to_end_with_openvino_mlmodel(fake_yolo_ir, labels_file, tmp_path):
    """erh:openvino:mlmodel + erh:openvino:yolo on a synthetic YOLOv8-shaped OpenVINO model."""
    ml = OpenVINOMLModel.new(
        ComponentConfig(name="m", api="rdk:service:mlmodel", model="erh:openvino:mlmodel",
                        attributes=dict_to_struct({"model_path": fake_yolo_ir, "label_path": labels_file, "device": "CPU", "cache_dir": str(tmp_path)})),
        {},
    )
    try:
        cam = FakeCamera("cam", make_image(96, 64))  # scale 2/3, new 64x43, pad_y 10
        v = YoloVision.new(make_config(mlmodel_name="m", camera_name="cam"), deps(ml, cam))
        dets = await v.get_detections_from_camera("cam")
        names = [d.class_name for d in dets]
        assert names == ["cat", "dog"]
        cat = dets[0]
        # cat xyxy in letterbox coords [22, 22, 42, 42] -> orig x = 22 * 1.5 = 33, y = (22 - 10) * 1.5 = 18 (+/- 1 px)
        assert abs(cat.x_min - 33) <= 1 and abs(cat.y_min - 18) <= 1 and abs(cat.x_max - 63) <= 1 and abs(cat.y_max - 48) <= 1
        info = await v.do_command({"command": "model_info"})
        assert info["version"] == "v8" and info["labels_count"] == 3 and info["input_size"] == [64, 64]
        stats = await ml.do_command({"command": "stats"})
        assert stats["total_calls"] == 1
    finally:
        await ml.close()


async def test_yolo_forwards_diagnostics(fake_yolo_ir, labels_file, tmp_path):
    ml = OpenVINOMLModel.new(
        ComponentConfig(name="m2", api="rdk:service:mlmodel", model="erh:openvino:mlmodel",
                        attributes=dict_to_struct({"model_path": fake_yolo_ir, "device": "CPU", "cache_dir": str(tmp_path)})),
        {},
    )
    try:
        v = YoloVision.new(make_config(mlmodel_name="m2"), deps(ml))
        assert (await v.do_command({"command": "list_devices"}))["openvino_version"]
        assert (await v.do_command({"command": "get_compiled_properties"}))["requested_device"] == "CPU"
        with pytest.raises(ValueError, match="unknown command"):
            await v.do_command({"command": "nope"})
    finally:
        await ml.close()
