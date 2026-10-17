import io

import numpy as np
import pytest
from PIL import Image
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.services.mlmodel import Metadata, MLModel, TensorInfo
from viam.utils import dict_to_struct

from models.detr_decode import IMAGENET_MEAN, decode, pick_outputs, preprocess, resize_bilinear
from models.detr_vision import RfDetrConfig, RfDetrVision


def logit(p):
    return float(np.log(p / (1 - p)))


def detr_outputs(rows, Q=5, C=4):
    """rows: (query, cx, cy, w, h, class_slot, prob). Every other logit is very negative."""
    boxes = np.zeros((1, Q, 4), dtype=np.float32)
    logits = np.full((1, Q, C), -20.0, dtype=np.float32)
    for q, cx, cy, w, h, c, p in rows:
        boxes[0, q] = [cx, cy, w, h]
        logits[0, q, c] = logit(p)
    return boxes, logits


class FakeDetrModel(MLModel):
    def __init__(self, name, boxes, logits, input_shape=(1, 3, 32, 32), dtype="float32", label_path=None):
        super().__init__(name)
        self.boxes, self.logits = boxes, logits
        self.input_shape, self.dtype, self.label_path = list(input_shape), dtype, label_path
        self.inputs = []

    async def infer(self, input_tensors, *, extra=None, timeout=None):
        self.inputs.append(input_tensors)
        return {"dets": self.boxes, "labels": self.logits}

    async def metadata(self, *, extra=None, timeout=None):
        dets = TensorInfo(name="dets", data_type="float32", shape=list(self.boxes.shape))
        labels = TensorInfo(name="labels", data_type="float32", shape=list(self.logits.shape))
        if self.label_path:
            labels.extra.CopyFrom(dict_to_struct({"labels": self.label_path}))
        return Metadata(name="fake", type="fake", input_info=[TensorInfo(name="input", data_type=self.dtype, shape=self.input_shape)],
                        output_info=[dets, labels])


@pytest.fixture
def slot_labels(tmp_path):
    # Slot 0 is unused, as in the COCO-pretrained RF-DETR checkpoints, which index classes by COCO category id.
    p = tmp_path / "labels.txt"
    p.write_text("N/A\nperson\nbicycle\ncar\n")
    return str(p)


def make_service(ml, **attrs):
    cfg = ComponentConfig(name="det", api="rdk:service:vision", model="erh:openvino:rf-detr",
                          attributes=dict_to_struct({"mlmodel_name": ml.name, **attrs}))
    return RfDetrVision.new(cfg, {MLModel.get_resource_name(ml.name): ml})


def png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, format="PNG")
    return ViamImage(buf.getvalue(), CameraMimeType.PNG)


def test_resize_bilinear_half_pixel_without_antialiasing():
    img = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    np.testing.assert_array_equal(resize_bilinear(img, (4, 4)), img)
    # Halving with half-pixel centers samples between pixel pairs: the mean of each 2x2 block.
    np.testing.assert_allclose(resize_bilinear(img, (2, 2))[..., 0], [[2.5, 4.5], [10.5, 12.5]])
    # Upscaling holds the edges instead of extrapolating.
    up = resize_bilinear(np.array([[[0.0], [10.0]]], dtype=np.float32), (1, 4))[0, :, 0]
    np.testing.assert_allclose(up, [0, 2.5, 7.5, 10])
    assert resize_bilinear(np.zeros((3, 5, 3), dtype=np.uint8), (6, 7)).dtype == np.float32


def test_preprocess_stretches_and_imagenet_normalizes():
    rgb = np.empty((10, 30, 3), dtype=np.uint8)
    rgb[:] = np.round(IMAGENET_MEAN * 255).astype(np.uint8)
    x = preprocess(rgb, (8, 8))
    assert x.shape == (1, 3, 8, 8) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(x, 0, atol=0.01)
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess(np.zeros((4, 4), dtype=np.uint8), (8, 8))


def test_decode_topk_over_query_class_pairs():
    boxes, logits = detr_outputs([(0, 0.5, 0.5, 0.2, 0.4, 1, 0.9), (3, 0.25, 0.75, 0.5, 0.5, 3, 0.6)])
    logits[0, 0, 2] = logit(0.7)  # the same query also scores as class 2
    xyxy, scores, cls = decode(boxes[0], logits[0], 0.5, 10, (100, 200))
    assert cls.tolist() == [1, 2, 3]
    np.testing.assert_allclose(scores, [0.9, 0.7, 0.6], rtol=1e-5)
    np.testing.assert_allclose(xyxy[0], [80, 30, 120, 70])  # normalized cxcywh scaled to W=200, H=100
    np.testing.assert_allclose(xyxy[2], [0, 50, 100, 100])  # clipped to the image
    assert decode(boxes[0], logits[0], 0.5, 2, (100, 200))[2].tolist() == [1, 2]
    assert len(decode(boxes[0], logits[0], 0.95, 10, (100, 200))[0]) == 0


def test_pick_outputs_by_name_or_shape():
    boxes, logits = detr_outputs([], Q=3, C=2)
    b, lg = pick_outputs({"labels": logits, "dets": boxes})
    assert b.shape == (3, 4) and lg.shape == (3, 2)
    b, lg = pick_outputs({"a": logits, "b": boxes, "masks": np.zeros((1, 3, 8, 8))})
    assert b.shape == (3, 4) and lg.shape == (3, 2)
    with pytest.raises(ValueError, match="expected DETR outputs"):
        pick_outputs({"output0": np.zeros((1, 84, 8400))})
    with pytest.raises(ValueError, match="same queries"):
        pick_outputs({"dets": boxes, "labels": np.zeros((1, 4, 2))})


def test_config():
    assert RfDetrVision.validate_config(ComponentConfig(attributes=dict_to_struct({"mlmodel_name": "m", "camera_name": "c"}))) == (["m"], ["c"])
    cfg = RfDetrConfig({"mlmodel_name": "m"})
    assert cfg.confidence_threshold == 0.5 and not cfg.raw_pixels
    assert RfDetrConfig({"mlmodel_name": "m", "input_format": "NHWC_u8_rgb"}).raw_pixels
    for bad, msg in (({"yolo_version": "v8"}, "do not apply"), ({"iou_threshold": 0.5}, "do not apply"),
                     ({"input_format": "NCHW_f32_rgb_norm"}, "must be one of")):
        with pytest.raises(ValueError, match=msg):
            RfDetrConfig({"mlmodel_name": "m", **bad})


async def test_detections_in_original_pixels(slot_labels):
    boxes, logits = detr_outputs([(0, 0.5, 0.5, 0.2, 0.4, 1, 0.9), (1, 0.25, 0.25, 0.1, 0.1, 3, 0.4)])
    ml = FakeDetrModel("m", boxes, logits, label_path=slot_labels)
    v = make_service(ml)
    dets = await v.get_detections(png(200, 100))
    assert [d.class_name for d in dets] == ["person"]  # 0.4 is under the 0.5 default threshold
    d = dets[0]
    assert (d.x_min, d.y_min, d.x_max, d.y_max) == (80, 30, 120, 70)  # stretched, not letterboxed
    assert d.x_min_normalized == pytest.approx(0.4) and d.y_max_normalized == pytest.approx(0.7)
    sent = ml.inputs[0]["input"]
    assert sent.shape == (1, 3, 32, 32) and sent.dtype == np.float32
    info = await v.do_command({"command": "model_info"})
    assert info["input_size"] == [32, 32] and "version" not in info


async def test_class_filter_applies_before_max_detections(slot_labels):
    rows = [(q, 0.5, 0.5, 0.1, 0.1, 1, 0.9 - q * 0.01) for q in range(4)] + [(4, 0.5, 0.5, 0.1, 0.1, 3, 0.6)]
    ml = FakeDetrModel("m", *detr_outputs(rows), label_path=slot_labels)
    v = make_service(ml, class_filter=["car"], max_detections=1)
    assert [d.class_name for d in await v.get_detections(png(64, 64))] == ["car"]
    v = make_service(ml, max_detections=2)
    assert [d.class_name for d in await v.get_detections(png(64, 64))] == ["person", "person"]


async def test_raw_pixels_for_on_device_preprocess(slot_labels):
    # An mlmodel with `preprocess` and resize reports a dynamic uint8 NHWC input.
    ml = FakeDetrModel("m", *detr_outputs([(0, 0.5, 0.5, 0.2, 0.4, 1, 0.9)]), input_shape=(1, -1, -1, 3), dtype="uint8",
                       label_path=slot_labels)
    v = make_service(ml, input_format="NHWC_u8_rgb")
    dets = await v.get_detections(png(200, 100))
    assert len(dets) == 1 and (dets[0].x_min, dets[0].x_max) == (80, 120)
    sent = ml.inputs[0]["input"]
    assert sent.shape == (1, 100, 200, 3) and sent.dtype == np.uint8
    assert (await v.do_command({"command": "model_info"}))["input_size"] is None
    with pytest.raises(ValueError, match="set input_size"):
        await make_service(ml).get_detections(png(200, 100))
