"""Opt-in integration test with a real YOLOv8n ONNX export.

Set YOLOV8N_ONNX=/path/to/yolov8n.onnx (see `make fetch-yolov8n`). Skipped otherwise, so CI stays hermetic.
The reference expectation is deliberately loose: a synthetic image with a large solid rectangle is not a
COCO object, so we only check that the whole pipeline runs and returns well-formed, in-bounds boxes.
"""

import io
import os

import pytest
from PIL import Image, ImageDraw
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata
from viam.services.mlmodel import MLModel
from viam.utils import dict_to_struct

from models.openvino_mlmodel import OpenVINOMLModel
from models.yolo_vision import YoloVision

MODEL = os.environ.get("YOLOV8N_ONNX")
pytestmark = pytest.mark.skipif(not MODEL or not os.path.isfile(MODEL or ""), reason="set YOLOV8N_ONNX to a yolov8n.onnx file")

COCO80 = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench bird "
    "cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball "
    "kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple "
    "sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse "
    "remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()


class OneShotCamera(Camera):
    def __init__(self, name, data):
        super().__init__(name)
        self.data = data

    async def get_images(self, *, filter_source_names=None, extra=None, timeout=None, **kwargs):
        return [NamedImage("color", self.data, CameraMimeType.JPEG)], ResponseMetadata()

    async def get_point_cloud(self, **kwargs):
        raise NotImplementedError

    async def get_properties(self, **kwargs):
        raise NotImplementedError


@pytest.fixture
def labels(tmp_path):
    p = tmp_path / "coco.txt"
    p.write_text("\n".join(COCO80) + "\n")
    return str(p)


async def test_yolov8n_end_to_end(labels, tmp_path):
    ml = OpenVINOMLModel.new(
        ComponentConfig(name="m", api="rdk:service:mlmodel", model="erh:openvino:mlmodel",
                        attributes=dict_to_struct({"model_path": MODEL, "label_path": labels, "device": "CPU",
                                                   "cache_dir": str(tmp_path / "cache")})),
        {},
    )
    try:
        md = await ml.metadata()
        assert [t.name for t in md.input_info] == ["images"]
        assert list(md.input_info[0].shape) == [1, 3, 640, 640]
        img = Image.new("RGB", (800, 450), (200, 200, 200))
        ImageDraw.Draw(img).rectangle([100, 100, 400, 400], fill=(20, 20, 20))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cam = OneShotCamera("cam", buf.getvalue())
        vision = YoloVision.new(
            ComponentConfig(name="v", api="rdk:service:vision", model="erh:openvino:yolo",
                            attributes=dict_to_struct({"mlmodel_name": "m", "camera_name": "cam", "confidence_threshold": 0.05})),
            {MLModel.get_resource_name("m"): ml, Camera.get_resource_name("cam"): cam},
        )
        dets = await vision.get_detections_from_camera("cam")
        info = await vision.do_command({"command": "model_info"})
        assert info["version"] == "v8" and info["labels_count"] == 80
        for d in dets:
            assert 0 <= d.x_min <= d.x_max <= 800 and 0 <= d.y_min <= d.y_max <= 450
            assert d.class_name in COCO80 and 0 < d.confidence <= 1
        bench = await ml.do_command({"command": "benchmark", "iterations": 5, "warmup": 2})
        assert bench["p50_ms"] > 0
    finally:
        await ml.close()
