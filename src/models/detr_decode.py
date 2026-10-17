"""Pure-numpy RF-DETR helpers: preprocessing and output decoding. No Viam or OpenVINO imports so this is
trivially unit-testable.

RF-DETR (and DETR-style detectors generally) predict a fixed set of queries, each with a normalized ``cx, cy, w, h``
box and one logit per class slot. There is no NMS: each object gets one query. Decoding is a sigmoid over the logits,
then the top-scoring (query, class) pairs.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_axis_weights(src: int, dst: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices and weights for bilinear resampling along one axis with half-pixel centers."""
    pos = (np.arange(dst, dtype=np.float32) + 0.5) * (src / dst) - 0.5
    pos = np.clip(pos, 0, src - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, src - 1)
    return lo, hi, (pos - lo).astype(np.float32)


def resize_bilinear(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an HxWxC image to ``size`` (H, W) as float32: bilinear, half-pixel centers, no antialiasing.

    This matches ``torchvision.transforms.functional.resize(..., antialias=False)``, which RF-DETR uses at training
    and inference. PIL's bilinear filter antialiases when downscaling and shifts the scores. Rows are resampled
    first, on the source dtype, so only the output-height rows are ever converted to float."""
    h, w = image.shape[:2]
    out_h, out_w = int(size[0]), int(size[1])
    y0, y1, wy = _resize_axis_weights(h, out_h)
    x0, x1, wx = _resize_axis_weights(w, out_w)
    wy = wy[:, None, None]
    rows = image[y0].astype(np.float32) * (1 - wy) + image[y1].astype(np.float32) * wy
    wx = wx[None, :, None]
    return rows[:, x0] * (1 - wx) + rows[:, x1] * wx


def preprocess(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """HxWx3 uint8 RGB -> [1, 3, H, W] float32, stretched to ``size`` (no letterbox) and ImageNet-normalized."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 image, got shape {list(rgb.shape)}")
    img = resize_bilinear(rgb, size)
    img = (img - 255.0 * IMAGENET_MEAN) * (1.0 / (255.0 * IMAGENET_STD))
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis].astype(np.float32))


def pick_outputs(outputs: Mapping[str, np.ndarray], output_names: Sequence[str] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """Find the (boxes [Q, 4], logits [Q, C]) pair among the model outputs.

    Uses the names RF-DETR's exporter gives them (``dets`` and ``labels``) when present, otherwise the rank-3 outputs
    with matching query counts, taking the one whose last dimension is 4 as the boxes."""
    if "dets" in outputs and "labels" in outputs:
        boxes, logits = np.asarray(outputs["dets"]), np.asarray(outputs["labels"])
    else:
        names = list(output_names) + [n for n in outputs if n not in output_names]
        rank3 = [np.asarray(outputs[n]) for n in names if n in outputs and np.ndim(outputs[n]) == 3]
        box_like = [a for a in rank3 if a.shape[-1] == 4]
        others = [a for a in rank3 if a.shape[-1] != 4]
        if not box_like or not others:
            raise ValueError(
                "expected DETR outputs [1, Q, 4] boxes and [1, Q, C] class logits, got "
                f"{[(k, list(np.shape(v))) for k, v in outputs.items()]}"
            )
        boxes, logits = box_like[0], others[0]
    if boxes.ndim != 3 or logits.ndim != 3 or boxes.shape[0] != 1 or logits.shape[0] != 1:
        raise ValueError(f"only batch size 1 is supported, got boxes {list(boxes.shape)} and logits {list(logits.shape)}")
    if boxes.shape[1] != logits.shape[1] or boxes.shape[2] != 4:
        raise ValueError(f"boxes {list(boxes.shape)} and logits {list(logits.shape)} do not describe the same queries")
    return boxes[0].astype(np.float32), logits[0].astype(np.float32)


def decode(
    boxes: np.ndarray,
    logits: np.ndarray,
    conf_threshold: float,
    max_detections: int,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's queries into (boxes xyxy [N, 4] in pixels of ``image_size`` (H, W), scores [N], class ids [N]).

    Every (query, class) pair is a candidate, as in RF-DETR's own post-processing, so one query can yield more than one
    class. Results are sorted by descending score."""
    h, w = image_size
    scores = 1.0 / (1.0 + np.exp(-logits))
    flat = scores.ravel()
    keep = np.nonzero(flat >= conf_threshold)[0]
    keep = keep[np.argsort(-flat[keep], kind="stable")][:max_detections]
    query, class_ids = np.divmod(keep, scores.shape[1])
    cx, cy, bw, bh = boxes[query].T
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    return xyxy.astype(np.float32), flat[keep].astype(np.float32), class_ids.astype(np.int64)
