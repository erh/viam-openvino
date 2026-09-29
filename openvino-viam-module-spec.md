# Spec: OpenVINO ML Model Module for Viam

**Status:** Draft for implementation
**Audience:** Implementing engineer / coding agent
**Deliverable:** A Viam registry module that runs ML models on Intel hardware (CPU, iGPU, Arc GPU, NPU) via the OpenVINO runtime.

---

## 1. Purpose

Viam machines built on Intel x86 hardware (industrial PCs, NUCs, Core Ultra edge boxes, Atom/N-series boards) currently fall back to generic CPU runtimes (`tflite_cpu`, `onnx-cpu`) and leave the integrated GPU and NPU unused. This module makes those machines first-class ML targets by implementing Viam's `mlmodel` service API on top of OpenVINO.

Because it implements the standard `mlmodel` API, the stock `vision:mlmodel` service and anything else that consumes `Infer`/`Metadata` must work with it with no changes. This module is a runtime, not a vision library. The one exception is YOLO decoding (Section 6), which fills a known gap.

---

## 2. Scope

### In scope
1. An `mlmodel` service model backed by OpenVINO (Section 4). **Required.**
2. A `vision` service model that decodes YOLO outputs (Section 6). **Required, phase 2.**
3. On-device preprocessing via OpenVINO `PrePostProcessor`.
4. Compiled-model caching.
5. Device discovery, diagnostics, and benchmarking via `DoCommand`.
6. Packaging for the Viam registry, with CI builds.

### Out of scope (non-goals)
- Model training or fine-tuning.
- Quantization tooling (NNCF) and offline conversion pipelines. Users bring an already-quantized model if they want INT8.
- Bundling kernel drivers (Intel compute runtime, Level Zero, NPU driver). The module detects and reports missing drivers; it does not install them.
- General-purpose postprocessing for arbitrary architectures beyond the YOLO families listed in Section 6.

---

## 3. Model identifiers (proposed)

| Service | Model triplet | Notes |
|---|---|---|
| `rdk:service:mlmodel` | `viam:openvino:mlmodel` | Core runtime |
| `rdk:service:vision` | `viam:openvino:yolo` | YOLO decode + NMS on top of any `mlmodel` service |

Namespace is a placeholder. Confirm with the owner before publishing.

---

## 4. `mlmodel` service: `viam:openvino:mlmodel`

### 4.1 Configuration attributes

| Attribute | Type | Default | Required | Description |
|---|---|---|---|---|
| `model_path` | string | — | yes | Path to the model. Supports registry package variables, e.g. `${packages.ml_model.foo}/model.onnx`. Accepted formats: OpenVINO IR (`.xml` with sibling `.bin`), ONNX (`.onnx`), TFLite (`.tflite`), TensorFlow SavedModel (directory or `.pb`), PaddlePaddle (`.pdmodel`). Detect by extension; anything else is a validation error. |
| `label_path` | string | none | no | Path to a newline-delimited labels file. Surfaced through metadata (Section 4.3). |
| `device` | string | `"AUTO"` | no | OpenVINO device string: `CPU`, `GPU`, `GPU.0`, `GPU.1`, `NPU`, `AUTO`, `AUTO:GPU,CPU`, `HETERO:GPU,CPU`, `MULTI:...`. Passed through to `compile_model`. |
| `performance_hint` | string | `"LATENCY"` | no | `LATENCY`, `THROUGHPUT`, or `CUMULATIVE_THROUGHPUT`. |
| `num_requests` | int | derived | no | Number of async infer requests in the pool. If unset, use `OPTIMAL_NUMBER_OF_INFER_REQUESTS` from the compiled model. |
| `inference_precision` | string | device default | no | `f32`, `f16`, `bf16`. Maps to `INFERENCE_PRECISION_HINT`. |
| `cache_dir` | string | `$VIAM_MODULE_DATA/ov_cache` | no | Directory for compiled-blob cache (`CACHE_DIR`). Create it if missing. Set to `""` to disable. |
| `num_threads` | int | runtime default | no | CPU only. `INFERENCE_NUM_THREADS`. |
| `input_shape` | object `{name: [int,...]}` | model shape | no | Reshape inputs with dynamic or undesired shapes before compiling, e.g. `{"images": [1, 3, 640, 640]}`. |
| `preprocess` | object | none | no | See Section 4.4. If absent, inputs are fed as-is. |
| `extra_config` | object | `{}` | no | Arbitrary OpenVINO properties passed to `compile_model`. Escape hatch for advanced users; log them at startup. |

**Validation** (fail at config validation time with a clear message, not at first inference):
- `model_path` is required and must exist.
- For `.xml`, the sibling `.bin` must exist.
- `device` must parse. Do not require it to be present at validation time (drivers may load later), but log a warning if `Core().available_devices` does not include it.
- `performance_hint` and `inference_precision` must be in the allowed sets.
- `preprocess` fields must be internally consistent (Section 4.4).

**Reconfigure:** support in-place reconfiguration. Rebuilding the compiled model on any attribute change is acceptable. The cache should make this cheap after the first compile.

### 4.2 `Infer`

- Input: map of tensor name → ndarray. Output: map of tensor name → ndarray.
- Input names must match `Metadata.input_info[].name`. On mismatch, return an error listing the expected names.
- If the model has exactly one input and the caller supplies exactly one tensor under a different name, accept it and log once at debug level. Some callers use `image` generically.
- Convert dtypes when safe (e.g. uint8 → float32 when preprocessing is configured to handle scaling). Reject unsafe conversions with a clear error.
- Use a pool of `AsyncInferQueue` / infer requests of size `num_requests`, so concurrent `Infer` calls (e.g. multiple vision services or cameras sharing one mlmodel) run in parallel instead of serializing on one request.
- Must be safe to call concurrently.
- Output tensors must be copied out of OpenVINO-owned memory before returning.
- Record per-call latency (Section 7).

### 4.3 `Metadata`

Return:
- `name`: the model file stem.
- `type`: `"openvino"` (or a value consistent with other Viam mlmodel runtimes; check the reference implementations).
- `description`: model path, device actually in use, and the OpenVINO version.
- `input_info` / `output_info`: one entry per tensor with name, dtype, and shape. Use `-1` for dynamic dimensions.
- **Input info must describe what the caller sends, not what the model sees internally.** If preprocessing is configured (e.g. caller sends `uint8 [1,H,W,3]` RGB, model expects `float32 [1,3,640,640]`), report the caller-facing shape and dtype.
- **Labels:** if `label_path` is set, attach it to output tensors the same way the existing `tflite_cpu` and `onnx-cpu` modules do, so `vision:mlmodel` picks it up. As far as I know this is via the `extra` field on the output `TensorInfo` (e.g. `extra: {"labels": "<path>"}`). **Verify against the current reference module source before implementing.**

### 4.4 Preprocessing (`preprocess` attribute)

Implement with OpenVINO's `PrePostProcessor`, so preprocessing is compiled into the graph and runs on the target device.

```json
"preprocess": {
  "input_name": "images",
  "tensor_element_type": "u8",
  "tensor_layout": "NHWC",
  "tensor_color_format": "RGB",
  "model_layout": "NCHW",
  "model_color_format": "RGB",
  "resize": "linear",
  "mean": [0, 0, 0],
  "scale": [255, 255, 255]
}
```

| Field | Meaning |
|---|---|
| `input_name` | Which model input to attach to. Optional if the model has one input. |
| `tensor_element_type` | dtype the caller will send (`u8`, `f32`). |
| `tensor_layout` | Layout the caller will send. |
| `tensor_color_format` | `RGB` or `BGR`. |
| `model_layout` | Layout the model expects. |
| `model_color_format` | Color order the model expects. A mismatch triggers a conversion step. |
| `resize` | `none`, `linear`, `nearest`, `cubic`. When not `none`, the caller-facing spatial dims are dynamic (`-1`). |
| `mean`, `scale` | Per-channel; applied as `(x - mean) / scale`. |

Letterboxing is **not** handled here. It belongs to the YOLO vision model (Section 6), which needs the padding offsets to map boxes back.

---

## 5. `DoCommand` (on the mlmodel service)

All commands take a JSON map with a `"command"` key.

| Command | Args | Returns |
|---|---|---|
| `list_devices` | — | `{"devices": [{"name": "GPU.0", "full_name": "...", "type": "...", "capabilities": [...]}], "openvino_version": "..."}` |
| `get_compiled_properties` | — | Effective compiled-model properties: execution devices actually selected (important for `AUTO`), precision, number of streams/requests, cache hit or miss. |
| `benchmark` | `iterations` (int, default 100), `warmup` (int, default 10) | `{"mean_ms", "p50_ms", "p90_ms", "p99_ms", "throughput_fps", "device"}` using random inputs of the caller-facing shape. For dynamic dims, require an `input_shape` arg. |
| `stats` | — | Rolling latency stats and call counts since start (Section 7). |

---

## 6. Vision service: `viam:openvino:yolo`

**Why:** YOLO exports output raw tensors that need decoding and NMS. The stock `vision:mlmodel` path does not cover this well, and YOLO is the first thing most users will try.

### 6.1 Dependencies and config

| Attribute | Type | Default | Required | Description |
|---|---|---|---|---|
| `mlmodel_name` | string | — | yes | Name of an `mlmodel` service. Should work with any mlmodel implementation, not just this one. |
| `camera_name` | string | none | no | Default camera for `*FromCamera` calls. |
| `yolo_version` | string | `"auto"` | no | `v5`, `v8`, `v11`, or `auto` (infer from output shape). |
| `input_size` | [int, int] | from metadata | no | Model input H, W. |
| `confidence_threshold` | float | 0.25 | no | |
| `iou_threshold` | float | 0.45 | no | NMS IoU threshold. |
| `max_detections` | int | 100 | no | |
| `class_filter` | [string] | all | no | Only return these labels. |
| `label_path` | string | from mlmodel metadata | no | Overrides the labels from metadata. |
| `input_format` | string | `"NCHW_f32_rgb_norm"` | no | What to send to the mlmodel. If the mlmodel has on-device preprocessing configured, set to `NHWC_u8_rgb` to send raw pixels. |

### 6.2 Behavior

1. Get an image (from the request or `camera_name`).
2. Letterbox to `input_size` (keep aspect ratio, pad with 114), recording scale and padding.
3. Convert to `input_format` and call `Infer` on the dependency.
4. Decode:
   - **v8/v11:** output `[1, 4 + C, N]`. Boxes are `cx, cy, w, h`; class scores follow. No objectness.
   - **v5:** output `[1, N, 5 + C]`. `cx, cy, w, h, obj`, then class scores. Score = `obj * class_score`.
   - `auto`: pick based on which axis equals `4 + C` or `5 + C` given the label count. Fail clearly if ambiguous.
5. Filter by confidence, run class-aware NMS, cap at `max_detections`.
6. Undo letterbox to original image pixel coordinates; clamp to bounds.
7. Return Viam `Detection` objects (`x_min, y_min, x_max, y_max, confidence, class_name`).

### 6.3 Vision API coverage

- `GetDetections`, `GetDetectionsFromCamera`: implemented.
- `CaptureAllFromCamera`: implemented (image + detections).
- `GetProperties`: detections supported; classifications and point clouds not.
- `GetClassifications*`, `GetObjectPointClouds`: return an unimplemented error.

---

## 7. Observability

- On startup, log at info level: OpenVINO version, available devices, requested device, device(s) actually selected after compile, whether the compile was a cache hit, and compile time.
- If the requested device is `GPU` or `NPU` and it is not in `available_devices`, log an **error** naming the likely missing driver (Intel compute runtime / Level Zero for GPU; the Intel NPU driver for NPU). With `AUTO`, log a **warning** and continue on whatever `AUTO` picks. Never silently fall back from an explicitly requested accelerator. That is a hard failure.
- Track rolling inference latency (mean, p50, p90, p99 over the last N calls) and total calls. Expose via `DoCommand stats`.

---

## 8. Error handling

- Errors returned to callers must be actionable: include the tensor name, expected vs. received shape/dtype, or the device name.
- A failure to compile on the configured device fails the resource build with the OpenVINO error message included.
- A corrupt cache entry must not brick the module. On a cache load failure, delete the entry, recompile, and log a warning.

---

## 9. Implementation guidance

- **Language:** Python using the `openvino` pip wheel and the current `viam-sdk`. Scaffold with `viam module generate` so the entrypoint, `meta.json`, and build setup match current conventions. C++ is a possible later optimization; don't start there.
- **Reference implementations:** read the existing `tflite_cpu` and `onnx-cpu` mlmodel modules before writing the metadata and labels code, and match their conventions exactly so `vision:mlmodel` behaves identically.
- **Blocking work:** OpenVINO compile and infer calls block. Run them off the asyncio event loop (thread pool or OpenVINO's async API with futures).
- **Dependencies:** pin the `openvino` version in requirements. Record it in the metadata description.

---

## 10. Packaging and platforms

| Platform | Priority | Notes |
|---|---|---|
| linux/amd64 | P0 | CPU, Intel iGPU, Arc, NPU (Core Ultra). |
| linux/arm64 | P1 | CPU plugin only. Useful as a fallback runtime; not a headline target. |
| windows/amd64 | P2 | Only if the Viam module system supports it for this module type. |

- Build in CI on tag and upload to the registry.
- The README must document host driver prerequisites for GPU and NPU with install commands for Ubuntu 22.04 and 24.04, and tell users to run `DoCommand list_devices` to confirm.

---

## 11. Testing

### Unit
- Config validation: each invalid case in 4.1 produces the expected error.
- Metadata reports caller-facing shapes and dtypes when preprocessing is configured.
- YOLO decode for v5 and v8 using fixed fixture tensors with known expected boxes, including letterbox inversion for non-square images.
- NMS correctness (overlapping same-class boxes suppressed; different-class boxes kept).

### Integration (CPU, runs in CI)
- Load the same small model as ONNX and as IR; verify outputs match within tolerance.
- End to end: `viam:openvino:mlmodel` + stock `vision:mlmodel` on a MobileNet/SSD-style detector returns detections on a fixture image.
- End to end: `viam:openvino:mlmodel` + `viam:openvino:yolo` with YOLOv8n on a fixture image returns the expected classes with boxes within a pixel tolerance of a reference run.
- Concurrent `Infer` from several tasks completes without errors and with results matching serial runs.
- Reconfigure with a changed attribute rebuilds; the second startup is a cache hit.

### Hardware (manual, documented results)
- On an Intel iGPU machine and a Core Ultra machine: run `benchmark` on CPU, GPU, NPU, and AUTO with YOLOv8n and record results in the README.
- With GPU drivers removed: explicit `device: GPU` fails with the driver message; `AUTO` runs on CPU with a warning.

---

## 12. Acceptance criteria

1. A user can add the module from the registry, point it at an ONNX YOLOv8n model, add `viam:openvino:yolo`, and get correct detections from a camera **with no config beyond `model_path`, `label_path`, `mlmodel_name`, and `camera_name`.**
2. The same model works unchanged with the stock `vision:mlmodel` service for architectures whose outputs it already supports.
3. On a supported Intel iGPU, `device: GPU` shows a measurable speedup over `device: CPU` in the `benchmark` command, and the second startup is served from the cache.
4. Missing drivers produce a clear, specific log message, never a silent CPU fallback for an explicitly requested accelerator.
5. All tests in Section 11 pass in CI; hardware results are recorded in the README.

---

## 13. Open questions for the owner

1. Final namespace and model names (Section 3).
2. Should the YOLO vision model live in this module or ship as a separate, runtime-agnostic module so it also serves `onnx-cpu` and Triton users? (It is written to be runtime-agnostic either way.)
3. Is Windows support needed for any current customers?
4. Should segmentation (YOLO-seg) and pose models be in scope for a later phase?
