"""Pure-numpy YOLO helpers: letterboxing, output decoding, NMS, and mapping
boxes back to original image coordinates.  No Viam or OpenVINO imports so
this is trivially unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

YOLO_VERSIONS = ("v5", "v8", "v11", "auto")
INPUT_FORMAT_RE = re.compile(r"^(NCHW|NHWC)_(u8|f32|f16)_(rgb|bgr)(_norm)?$", re.IGNORECASE)


@dataclass(frozen=True)
class LetterboxInfo:
    orig_w: int
    orig_h: int
    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    out_w: int
    out_h: int


@dataclass(frozen=True)
class InputFormat:
    layout: str  # NCHW or NHWC
    dtype: str  # u8, f32, f16
    color: str  # rgb or bgr
    normalize: bool

    @classmethod
    def parse(cls, spec: str) -> "InputFormat":
        m = INPUT_FORMAT_RE.match(spec.strip())
        if not m:
            raise ValueError(
                f"input_format '{spec}' is invalid; expected <NCHW|NHWC>_<u8|f32|f16>_<rgb|bgr>[_norm], "
                "e.g. NCHW_f32_rgb_norm or NHWC_u8_rgb"
            )
        layout, dtype, color, norm = m.groups()
        fmt = cls(layout.upper(), dtype.lower(), color.lower(), norm is not None)
        if fmt.dtype == "u8" and fmt.normalize:
            raise ValueError(f"input_format '{spec}': u8 cannot be normalized; drop the _norm suffix")
        return fmt


def letterbox(image: np.ndarray, size: Tuple[int, int], pad_value: int = 114) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize ``image`` (H, W, 3 uint8) to fit inside ``size`` (H, W) keeping aspect
    ratio, padding the remainder with ``pad_value``.  Returns (canvas, info)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"letterbox expects an HxWx3 image, got shape {list(image.shape)}")
    out_h, out_w = int(size[0]), int(size[1])
    orig_h, orig_w = image.shape[:2]
    scale = min(out_w / orig_w, out_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    if (new_w, new_h) != (orig_w, orig_h):
        resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    else:
        resized = image
    canvas = np.full((out_h, out_w, 3), pad_value, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, LetterboxInfo(orig_w, orig_h, scale, pad_x, pad_y, new_w, new_h, out_w, out_h)


def to_model_input(canvas: np.ndarray, fmt: InputFormat) -> np.ndarray:
    """Convert an HxWx3 uint8 RGB canvas to the tensor layout/dtype the model wants (with batch dim)."""
    arr = canvas
    if fmt.color == "bgr":
        arr = arr[..., ::-1]
    if fmt.dtype == "u8":
        out = arr.astype(np.uint8)
    else:
        out = arr.astype(np.float32)
        if fmt.normalize:
            out = out / 255.0
        if fmt.dtype == "f16":
            out = out.astype(np.float16)
    if fmt.layout == "NCHW":
        out = np.transpose(out, (2, 0, 1))
    return np.ascontiguousarray(out[np.newaxis, ...])


def detect_version(output_shape: Sequence[int], num_classes: Optional[int]) -> str:
    """Pick 'v5' or 'v8' (v11 shares the v8 layout) from an output shape."""
    shape = [int(d) for d in output_shape]
    if len(shape) == 3:
        _, a, b = shape
    elif len(shape) == 2:
        a, b = shape
    else:
        raise ValueError(f"cannot infer YOLO version from output shape {shape}; expected [1, 4+C, N] or [1, N, 5+C]")
    if num_classes:
        v8 = a == 4 + num_classes
        v5 = b == 5 + num_classes
        if v8 and not v5:
            return "v8"
        if v5 and not v8:
            return "v5"
        raise ValueError(
            f"ambiguous YOLO output shape {shape} for {num_classes} labels: neither/both of dim1 == 4+C ({4 + num_classes}) "
            f"and dim2 == 5+C ({5 + num_classes}) match; set yolo_version explicitly"
        )
    # No labels: channel axis is the small one. v8 is [1, 4+C, N], v5 is [1, N, 5+C].
    if a < b:
        return "v8"
    if b < a:
        return "v5"
    raise ValueError(f"ambiguous YOLO output shape {shape}; set yolo_version explicitly or provide labels")


def decode(output: np.ndarray, version: str, conf_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLO output into (boxes xyxy [N,4], scores [N], class_ids [N]) in
    letterboxed-image coordinates, keeping only candidates above ``conf_threshold``."""
    out = np.asarray(output, dtype=np.float32)
    if out.ndim == 3:
        if out.shape[0] != 1:
            raise ValueError(f"only batch size 1 is supported, got output shape {list(out.shape)}")
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"unexpected YOLO output rank {out.ndim} with shape {list(output.shape)}")
    version = "v8" if version == "v11" else version
    if version == "v8":
        preds = out.T  # [N, 4+C]
        if preds.shape[1] < 5:
            raise ValueError(f"v8 output must have at least 5 channels, got shape {list(output.shape)}")
        class_scores = preds[:, 4:]
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
    elif version == "v5":
        preds = out  # [N, 5+C]
        if preds.shape[1] < 6:
            raise ValueError(f"v5 output must have at least 6 channels, got shape {list(output.shape)}")
        class_scores = preds[:, 5:]
        obj = preds[:, 4]
        class_ids = class_scores.argmax(axis=1)
        scores = obj * class_scores[np.arange(len(class_ids)), class_ids]
    else:
        raise ValueError(f"unknown yolo_version '{version}'")
    keep = scores >= conf_threshold
    preds, scores, class_ids = preds[keep], scores[keep], class_ids[keep]
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, scores, class_ids.astype(np.int64)


def nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float, max_detections: int) -> np.ndarray:
    """Class-aware greedy NMS. Returns indices to keep, ordered by descending score."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    # Offset boxes per class so different classes never overlap.
    offset = (boxes.max() + 1.0) if boxes.size else 1.0
    shifted = boxes + class_ids[:, None].astype(np.float32) * offset
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = shifted[:, 0], shifted[:, 1], shifted[:, 2], shifted[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_detections:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def unletterbox(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from letterboxed coordinates back to original pixels, clamped to bounds."""
    if len(boxes) == 0:
        return boxes.reshape(0, 4)
    out = boxes.astype(np.float32).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - info.pad_x) / info.scale
    out[:, [1, 3]] = (out[:, [1, 3]] - info.pad_y) / info.scale
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, info.orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, info.orig_h)
    return out
