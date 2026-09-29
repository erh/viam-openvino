"""``erh:openvino:yolo`` - a vision service that decodes YOLO v5/v8/v11 outputs
from any ``mlmodel`` service and returns Viam detections."""

from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components.camera import Camera
from viam.errors import NotSupportedError
from viam.logging import getLogger
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.mlmodel import MLModel
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes, struct_to_dict

from . import registry
from .config import load_labels
from .yolo_decode import (
    YOLO_VERSIONS,
    InputFormat,
    decode,
    detect_version,
    letterbox,
    nms,
    to_model_input,
    unletterbox,
)

LOGGER = getLogger(__name__)


class YoloConfig:
    def __init__(self, attrs: Mapping[str, Any]) -> None:
        attrs = dict(attrs or {})
        name = attrs.get("mlmodel_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("mlmodel_name is required and must name an mlmodel service")
        self.mlmodel_name: str = name.strip()

        cam = attrs.get("camera_name")
        if cam is not None and (not isinstance(cam, str) or not cam.strip()):
            raise ValueError("camera_name must be a non-empty string when set")
        self.camera_name: Optional[str] = cam.strip() if cam else None

        version = attrs.get("yolo_version", "auto")
        if not isinstance(version, str) or version.lower() not in YOLO_VERSIONS:
            raise ValueError(f"yolo_version '{version}' must be one of {', '.join(YOLO_VERSIONS)}")
        self.yolo_version: str = version.lower()

        size = attrs.get("input_size")
        self.input_size: Optional[Tuple[int, int]] = None
        if size is not None:
            if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
                isinstance(d, bool) or not isinstance(d, (int, float)) or int(d) != d or int(d) <= 0 for d in size
            ):
                raise ValueError("input_size must be [height, width] with positive integers")
            self.input_size = (int(size[0]), int(size[1]))

        self.confidence_threshold = _float_attr(attrs, "confidence_threshold", 0.25, 0.0, 1.0)
        self.iou_threshold = _float_attr(attrs, "iou_threshold", 0.45, 0.0, 1.0)
        max_det = attrs.get("max_detections", 100)
        if isinstance(max_det, bool) or not isinstance(max_det, (int, float)) or int(max_det) != max_det or int(max_det) < 1:
            raise ValueError("max_detections must be a positive integer")
        self.max_detections = int(max_det)

        cf = attrs.get("class_filter")
        if cf is not None and (not isinstance(cf, (list, tuple)) or any(not isinstance(c, str) for c in cf)):
            raise ValueError("class_filter must be a list of label strings")
        self.class_filter: Optional[set] = set(cf) if cf else None

        lp = attrs.get("label_path")
        if lp is not None:
            if not isinstance(lp, str) or not lp.strip():
                raise ValueError("label_path must be a non-empty string when set")
            lp = os.path.expanduser(lp.strip())
            if not os.path.isfile(lp):
                raise ValueError(f"label_path '{lp}' does not exist")
        self.label_path: Optional[str] = lp

        self.input_format = InputFormat.parse(str(attrs.get("input_format", "NCHW_f32_rgb_norm")))


def _float_attr(attrs: Mapping[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = attrs.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
        raise ValueError(f"{key} must be a number between {lo} and {hi}")
    return float(v)


class YoloVision(Vision, EasyResource):
    MODEL: ClassVar[Model] = "erh:openvino:yolo"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.cfg: Optional[YoloConfig] = None
        self.mlmodel: Optional[MLModel] = None
        self.cameras: Dict[str, Camera] = {}
        self.logger = getattr(self, "logger", None) or LOGGER
        self._model_info: Optional[Dict[str, Any]] = None
        self._info_lock = asyncio.Lock()

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = cls(config.name)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        cfg = YoloConfig(struct_to_dict(config.attributes))
        required = [cfg.mlmodel_name]
        optional = [cfg.camera_name] if cfg.camera_name else []
        return required, optional

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        cfg = YoloConfig(struct_to_dict(config.attributes))
        ml = dependencies.get(MLModel.get_resource_name(cfg.mlmodel_name))
        if ml is None:
            raise ValueError(f"mlmodel '{cfg.mlmodel_name}' was not found in dependencies")
        self.mlmodel = ml  # type: ignore[assignment]
        self.cameras = {}
        for rn, res in dependencies.items():
            if rn.subtype == "camera":
                self.cameras[rn.name] = res  # type: ignore[assignment]
        self.cfg = cfg
        self._model_info = None

    async def close(self) -> None:
        self._model_info = None

    # ---------------------------------------------------------- model info
    async def _get_model_info(self) -> Dict[str, Any]:
        if self._model_info is not None:
            return self._model_info
        async with self._info_lock:
            if self._model_info is not None:
                return self._model_info
            assert self.mlmodel is not None and self.cfg is not None
            md = await self.mlmodel.metadata()
            if len(md.input_info) != 1:
                raise ValueError(
                    f"mlmodel '{self.cfg.mlmodel_name}' has {len(md.input_info)} inputs; YOLO models must have exactly one"
                )
            inp = md.input_info[0]
            input_size = self.cfg.input_size
            if input_size is None:
                input_size = _spatial_from_shape(list(inp.shape), self.cfg.input_format.layout)
            if input_size is None:
                raise ValueError(
                    f"could not determine the model input size from metadata shape {list(inp.shape)}; set input_size"
                )
            label_path = self.cfg.label_path
            if label_path is None:
                for out in md.output_info:
                    extra = struct_to_dict(out.extra) if out.HasField("extra") else {}
                    lp = extra.get("labels")
                    if isinstance(lp, str) and lp:
                        label_path = lp
                        break
            labels: List[str] = []
            if label_path:
                if os.path.isfile(label_path):
                    labels = load_labels(label_path)
                else:
                    self.logger.warning("labels file '%s' from mlmodel metadata does not exist; using class indices", label_path)
            self._model_info = {
                "input_name": inp.name,
                "input_size": input_size,
                "output_names": [o.name for o in md.output_info],
                "output_shapes": {o.name: list(o.shape) for o in md.output_info},
                "labels": labels,
                "version": None,  # resolved on first inference
            }
            self.logger.info(
                "yolo '%s': mlmodel input '%s' size %s, %d labels, outputs %s",
                self.name, inp.name, input_size, len(labels), self._model_info["output_names"],
            )
            return self._model_info

    # ------------------------------------------------------------ detection
    async def _detect(self, image: ViamImage) -> List[Detection]:
        assert self.cfg is not None and self.mlmodel is not None
        cfg = self.cfg
        info = await self._get_model_info()
        pil = viam_to_pil_image(image).convert("RGB")
        rgb = np.asarray(pil)
        canvas, lb = letterbox(rgb, info["input_size"])
        tensor = to_model_input(canvas, cfg.input_format)
        outputs = await self.mlmodel.infer({info["input_name"]: tensor})
        raw = _pick_detection_output(outputs, info["output_names"])
        version = info["version"]
        if version is None:
            version = cfg.yolo_version
            if version == "auto":
                version = detect_version(raw.shape, len(info["labels"]) or None)
                self.logger.info("yolo '%s': auto-detected YOLO %s from output shape %s", self.name, version, list(raw.shape))
            info["version"] = version
        boxes, scores, class_ids = decode(raw, version, cfg.confidence_threshold)
        keep = nms(boxes, scores, class_ids, cfg.iou_threshold, cfg.max_detections)
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        boxes = unletterbox(boxes, lb)
        labels = info["labels"]
        dets: List[Detection] = []
        for box, score, cid in zip(boxes, scores, class_ids):
            name = labels[cid] if 0 <= cid < len(labels) else str(int(cid))
            if cfg.class_filter is not None and name not in cfg.class_filter:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append(
                Detection(
                    x_min=int(round(x1)), y_min=int(round(y1)), x_max=int(round(x2)), y_max=int(round(y2)),
                    x_min_normalized=x1 / lb.orig_w, y_min_normalized=y1 / lb.orig_h,
                    x_max_normalized=x2 / lb.orig_w, y_max_normalized=y2 / lb.orig_h,
                    confidence=float(score), class_name=name,
                )
            )
        return dets

    async def _get_camera_image(self, camera_name: str) -> ViamImage:
        assert self.cfg is not None
        name = camera_name or self.cfg.camera_name
        if not name:
            raise ValueError(f"vision '{self.name}': no camera_name given and no default camera configured")
        cam = self.cameras.get(name)
        if cam is None:
            raise ValueError(
                f"vision '{self.name}': camera '{name}' is not a configured dependency; set camera_name to '{name}'"
            )
        images, _ = await cam.get_images()
        if not images:
            raise RuntimeError(f"camera '{name}' returned no images")
        first = images[0]
        return ViamImage(first.data, first.mime_type)

    # ------------------------------------------------------------ vision API
    async def get_detections(self, image: ViamImage, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[Detection]:
        return await self._detect(image)

    async def get_detections_from_camera(self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[Detection]:
        image = await self._get_camera_image(camera_name)
        return await self._detect(image)

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        if not (return_image or return_detections):
            return result
        image = await self._get_camera_image(camera_name)
        if return_image:
            result.image = image
        if return_detections:
            result.detections = await self._detect(image)
        return result

    async def get_classifications(self, image: ViamImage, count: int, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[Classification]:
        raise NotSupportedError(f"vision '{self.name}' (erh:openvino:yolo) does not support classifications")

    async def get_classifications_from_camera(self, camera_name: str, count: int, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[Classification]:
        raise NotSupportedError(f"vision '{self.name}' (erh:openvino:yolo) does not support classifications")

    async def get_object_point_clouds(self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> List[PointCloudObject]:
        raise NotSupportedError(f"vision '{self.name}' (erh:openvino:yolo) does not support object point clouds")

    async def get_properties(self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
            default_camera=(self.cfg.camera_name or "") if self.cfg else "",
        )

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "model_info":
            info = dict(await self._get_model_info())
            info["input_size"] = list(info["input_size"])
            info["labels_count"] = len(info.pop("labels"))
            info["version"] = info["version"] or self.cfg.yolo_version  # type: ignore[union-attr]
            return info
        if cmd in registry.DIAGNOSTIC_COMMANDS:
            # The mlmodel API has no DoCommand RPC; forward diagnostics to an in-process erh:openvino:mlmodel.
            return await registry.run_diagnostic(command, self.cfg.mlmodel_name if self.cfg else None)
        raise ValueError(f"unknown command '{cmd}'; expected model_info or one of {', '.join(registry.DIAGNOSTIC_COMMANDS)}")


def _spatial_from_shape(shape: List[int], layout: str) -> Optional[Tuple[int, int]]:
    if len(shape) != 4:
        return None
    if layout == "NCHW":
        h, w = shape[2], shape[3]
    else:
        h, w = shape[1], shape[2]
    if h > 0 and w > 0:
        return (int(h), int(w))
    return None


def _pick_detection_output(outputs: Mapping[str, Any], output_names: Sequence[str]) -> np.ndarray:
    """Pick the detection tensor: the first output whose rank is 2 or 3 (skips e.g. seg protos)."""
    candidates = []
    for name in list(output_names) + [n for n in outputs if n not in output_names]:
        if name in outputs:
            arr = np.asarray(outputs[name])
            if arr.ndim in (2, 3):
                candidates.append((name, arr))
    if not candidates:
        raise ValueError(f"no rank-2/3 YOLO output found in mlmodel outputs {[(k, list(np.shape(v))) for k, v in outputs.items()]}")
    return candidates[0][1]
