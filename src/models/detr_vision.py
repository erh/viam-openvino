"""``erh:openvino:rf-detr`` - a vision service that decodes RF-DETR outputs from any ``mlmodel`` service and returns
Viam detections. Camera handling, labels, model metadata and the Vision API come from ``erh:openvino:yolo``; only
preprocessing and decoding differ."""

from __future__ import annotations

from typing import Any, ClassVar, List, Mapping, Optional

import numpy as np
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.proto.service.vision import Detection
from viam.resource.types import Model
from viam.utils import ValueTypes

from .detr_decode import decode, pick_outputs, preprocess
from .yolo_decode import LetterboxInfo
from .yolo_vision import YoloConfig, YoloVision

YOLO_ONLY_ATTRIBUTES = ("yolo_version", "iou_threshold")
# Normalized float tensors at the model resolution, or raw pixels for an mlmodel with on-device `preprocess`.
INPUT_FORMATS = ("NCHW_f32_rgb", "NHWC_u8_rgb")


class RfDetrConfig(YoloConfig):
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        attrs = dict(attrs or {})
        unsupported = [k for k in YOLO_ONLY_ATTRIBUTES if k in attrs]
        if unsupported:
            raise ValueError(f"{', '.join(unsupported)} do not apply to RF-DETR models; remove them")
        fmt = attrs.setdefault("input_format", INPUT_FORMATS[0])
        if fmt not in INPUT_FORMATS:
            raise ValueError(f"input_format '{fmt}' must be one of {', '.join(INPUT_FORMATS)} for RF-DETR models")
        attrs.setdefault("confidence_threshold", 0.5)
        super().__init__(attrs)
        self.raw_pixels = fmt == "NHWC_u8_rgb"


class RfDetrVision(YoloVision):
    MODEL: ClassVar[Model] = "erh:openvino:rf-detr"
    CONFIG_CLASS: ClassVar[type] = RfDetrConfig

    async def _detect(self, image: ViamImage) -> List[Detection]:
        assert self.cfg is not None and self.mlmodel is not None
        cfg = self.cfg
        info = await self._get_model_info()
        rgb = np.asarray(viam_to_pil_image(image).convert("RGB"))
        tensor = np.ascontiguousarray(rgb[np.newaxis]) if cfg.raw_pixels else preprocess(rgb, info["input_size"])
        outputs = await self.mlmodel.infer({info["input_name"]: tensor})
        boxes, logits = pick_outputs(outputs, info["output_names"])
        h, w = rgb.shape[:2]
        # With a class filter, cap after filtering so other classes cannot crowd out the wanted ones.
        limit = cfg.max_detections if cfg.class_filter is None else logits.size
        xyxy, scores, class_ids = decode(boxes, logits, cfg.confidence_threshold, limit, (h, w))
        # Boxes are already in original pixels; an identity letterbox lets _to_detection normalize them.
        lb = LetterboxInfo(orig_w=w, orig_h=h, scale=1.0, pad_x=0, pad_y=0, new_w=w, new_h=h, out_w=w, out_h=h)
        dets: List[Detection] = []
        for box, score, cid in zip(xyxy, scores, class_ids):
            name = self._label(info, cid)
            if cfg.class_filter is not None and name not in cfg.class_filter:
                continue
            dets.append(self._to_detection(box, score, name, lb))
            if len(dets) >= cfg.max_detections:
                break
        return dets

    def _input_size_required(self) -> bool:
        # With raw pixels the mlmodel resizes on the device, so any image size is sent as is.
        return not self.cfg.raw_pixels  # type: ignore[union-attr]

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        result = await super().do_command(command, timeout=timeout, **kwargs)
        if command.get("command") == "model_info":
            result = dict(result)
            result.pop("version", None)
        return result
