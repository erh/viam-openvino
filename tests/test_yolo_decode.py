import numpy as np
import pytest

from models.yolo_decode import InputFormat, decode, detect_version, letterbox, nms, to_model_input, unletterbox


def _v8_output(rows, C):
    """rows: list of (cx, cy, w, h, class_id, score) -> [1, 4+C, N]"""
    out = np.zeros((1, 4 + C, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cid, s) in enumerate(rows):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cid, i] = s
    return out


def _v5_output(rows, C):
    """rows: list of (cx, cy, w, h, obj, class_id, cls_score) -> [1, N, 5+C]"""
    out = np.zeros((1, len(rows), 5 + C), dtype=np.float32)
    for i, (cx, cy, w, h, obj, cid, s) in enumerate(rows):
        out[0, i, :4] = [cx, cy, w, h]
        out[0, i, 4] = obj
        out[0, i, 5 + cid] = s
    return out


def test_decode_v8():
    out = _v8_output([(100, 100, 50, 20, 1, 0.9), (10, 10, 4, 4, 0, 0.1)], C=3)
    boxes, scores, cids = decode(out, "v8", 0.25)
    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [75, 90, 125, 110])
    assert scores[0] == pytest.approx(0.9) and cids[0] == 1


def test_decode_v11_alias():
    out = _v8_output([(100, 100, 50, 20, 2, 0.7)], C=3)
    boxes, scores, cids = decode(out, "v11", 0.25)
    assert cids[0] == 2 and scores[0] == pytest.approx(0.7)


def test_decode_v5_multiplies_objectness():
    out = _v5_output([(50, 60, 10, 20, 0.5, 2, 0.8), (5, 5, 2, 2, 0.9, 0, 0.2)], C=3)
    boxes, scores, cids = decode(out, "v5", 0.25)
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [45, 50, 55, 70])
    assert scores[0] == pytest.approx(0.4) and cids[0] == 2


def test_decode_rejects_batch():
    with pytest.raises(ValueError, match="batch size 1"):
        decode(np.zeros((2, 7, 3), dtype=np.float32), "v8", 0.1)


@pytest.mark.parametrize("shape,C,expected", [
    ((1, 84, 8400), 80, "v8"),
    ((1, 25200, 85), 80, "v5"),
    ((1, 84, 8400), None, "v8"),
    ((1, 25200, 85), None, "v5"),
    ((1, 7, 4), 3, "v8"),
    ((1, 4, 8), 3, "v5"),
])
def test_detect_version(shape, C, expected):
    assert detect_version(shape, C) == expected


def test_detect_version_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        detect_version((1, 84, 8400), 10)
    with pytest.raises(ValueError, match="ambiguous"):
        detect_version((1, 8, 8), None)


def test_nms_suppresses_same_class_keeps_other_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.85, 0.3], dtype=np.float32)
    cids = np.array([0, 0, 1, 0])
    keep = nms(boxes, scores, cids, 0.45, 100)
    assert keep.tolist() == [0, 2, 3]


def test_nms_max_detections():
    boxes = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(5)], dtype=np.float32)
    scores = np.linspace(0.5, 0.9, 5).astype(np.float32)
    keep = nms(boxes, scores, np.zeros(5, dtype=np.int64), 0.5, 2)
    assert keep.tolist() == [4, 3]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64), 0.5, 10).size == 0


def test_letterbox_non_square_and_inversion():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    canvas, info = letterbox(img, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.6)
    assert (info.new_w, info.new_h) == (640, 480)
    assert (info.pad_x, info.pad_y) == (0, 80)
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[80, 0].tolist() == [0, 0, 0]
    # A box in letterboxed space maps back to original pixels.
    boxes = np.array([[160, 240, 320, 400], [-10, 70, 1000, 1000]], dtype=np.float32)
    orig = unletterbox(boxes, info)
    np.testing.assert_allclose(orig[0], [100, 100, 200, 200])
    np.testing.assert_allclose(orig[1], [0, 0, 400, 300])  # clamped


def test_letterbox_tall_image():
    img = np.zeros((400, 200, 3), dtype=np.uint8)
    canvas, info = letterbox(img, (320, 320))
    assert (info.new_w, info.new_h) == (160, 320)
    assert (info.pad_x, info.pad_y) == (80, 0)


def test_to_model_input_formats():
    canvas = np.zeros((4, 4, 3), dtype=np.uint8)
    canvas[..., 0] = 255  # red
    t = to_model_input(canvas, InputFormat.parse("NCHW_f32_rgb_norm"))
    assert t.shape == (1, 3, 4, 4) and t.dtype == np.float32 and t[0, 0, 0, 0] == 1.0 and t[0, 2, 0, 0] == 0.0
    t = to_model_input(canvas, InputFormat.parse("NHWC_u8_bgr"))
    assert t.shape == (1, 4, 4, 3) and t.dtype == np.uint8 and t[0, 0, 0, 2] == 255 and t[0, 0, 0, 0] == 0
    t = to_model_input(canvas, InputFormat.parse("NCHW_f16_rgb"))
    assert t.dtype == np.float16 and t[0, 0, 0, 0] == 255


def test_input_format_invalid():
    with pytest.raises(ValueError):
        InputFormat.parse("CHW_f32")
    with pytest.raises(ValueError, match="u8 cannot be normalized"):
        InputFormat.parse("NHWC_u8_rgb_norm")
