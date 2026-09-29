# viam-openvino: OpenVINO module for Viam

Module ID: `erh:viam-openvino` (source: <https://github.com/erh/viam-openvino>).

Run ML models on Intel hardware (CPU, integrated GPU, Arc GPU, and the NPU in Core Ultra chips) from a
Viam machine, using the [OpenVINO](https://docs.openvino.ai/) runtime.

This module provides two models:

| Model | API | Purpose |
|---|---|---|
| [`erh:openvino:mlmodel`](#viamopenvinomlmodel) | `rdk:service:mlmodel` | Runs ONNX, OpenVINO IR, TFLite, TensorFlow and PaddlePaddle models. Works with the stock `vision:mlmodel` service and anything else that speaks the `mlmodel` API. |
| [`erh:openvino:yolo`](#viamopenvinoyolo) | `rdk:service:vision` | Decodes YOLO v5 / v8 / v11 detection outputs (letterbox, decode, NMS) on top of any `mlmodel` service, not only this one. |
| [`erh:openvino:diagnostics`](#viamopenvinodiagnostics) | `rdk:service:generic` | `list_devices`, `get_compiled_properties`, `benchmark` and `stats` via `DoCommand`. |

Supported platforms: `linux/amd64` (CPU, Intel iGPU, Arc, NPU), `windows/amd64` (CPU, Intel iGPU, Arc, NPU),
`linux/arm64` (CPU only, as a fallback runtime). macOS is supported for development from source only.

## Quick start: YOLOv8 detections from a camera

1. Upload a YOLOv8 export (`yolov8n.onnx`) and its labels file (one class name per line) to the Viam registry as an
   ML model package, or copy them to the machine.
2. Add the module to your machine and configure the two services:

```json
{
  "services": [
    {
      "name": "ov",
      "api": "rdk:service:mlmodel",
      "model": "erh:openvino:mlmodel",
      "attributes": {
        "model_path": "${packages.ml_model.yolov8n}/yolov8n.onnx",
        "label_path": "${packages.ml_model.yolov8n}/labels.txt"
      }
    },
    {
      "name": "detector",
      "api": "rdk:service:vision",
      "model": "erh:openvino:yolo",
      "attributes": {
        "mlmodel_name": "ov",
        "camera_name": "cam"
      }
    }
  ]
}
```

That is the whole configuration. `device` defaults to `AUTO`, which picks the best available accelerator (GPU or NPU
when the drivers are installed, otherwise CPU). Call `GetDetectionsFromCamera` on `detector` to get boxes.

To confirm which devices OpenVINO can see on the machine, add a `erh:openvino:diagnostics` service (or use the
`detector` above) and run `DoCommand`:

```json
{"command": "list_devices"}
```

## `erh:openvino:mlmodel`

### Attributes

| Attribute | Type | Default | Required | Description |
|---|---|---|---|---|
| `model_path` | string | | yes | Path to the model. Supports package variables such as `${packages.ml_model.foo}/model.onnx`. Formats are detected by extension: OpenVINO IR (`.xml` with a sibling `.bin`), ONNX (`.onnx`), TFLite (`.tflite`), TensorFlow (`.pb` or a SavedModel directory), PaddlePaddle (`.pdmodel`). |
| `label_path` | string | | no | Newline-delimited labels file. Exposed through metadata so `vision:mlmodel` and `erh:openvino:yolo` pick it up. |
| `device` | string | `"AUTO"` | no | OpenVINO device: `CPU`, `GPU`, `GPU.0`, `GPU.1`, `NPU`, `AUTO`, `AUTO:GPU,CPU`, `HETERO:GPU,CPU`, `MULTI:GPU,CPU`. |
| `performance_hint` | string | `"LATENCY"` | no | `LATENCY`, `THROUGHPUT` or `CUMULATIVE_THROUGHPUT`. |
| `num_requests` | int | optimal for device | no | Size of the infer-request pool. Concurrent `Infer` calls run in parallel up to this number. |
| `inference_precision` | string | device default | no | `f32`, `f16` or `bf16`. |
| `cache_dir` | string | `$VIAM_MODULE_DATA/ov_cache` | no | Compiled-model cache directory. The second start on the same device is served from the cache, which matters most for GPU and NPU compiles. Set to `""` to disable. |
| `num_threads` | int | runtime default | no | CPU inference threads. Only applies when `device` is exactly `CPU`. |
| `input_shape` | object | model shape | no | Reshape inputs before compiling, for example `{"images": [1, 3, 640, 640]}` for a model exported with dynamic dimensions. |
| `preprocess` | object | none | no | On-device preprocessing, see below. |
| `extra_config` | object | `{}` | no | Arbitrary OpenVINO properties passed to `compile_model`. Logged at startup. |

**How `AUTO` picks a device.** OpenVINO's own `AUTO` only considers GPU and CPU and ignores the NPU. This module
therefore expands a bare `"device": "AUTO"` into an explicit priority list over the devices actually present:
GPU(s) first (widest op support), then NPU, then CPU, for example `AUTO:GPU,NPU,CPU` on a Core Ultra machine or
`AUTO:NPU,CPU` on a box without a usable GPU. OpenVINO's `AUTO` then tries the candidates in that order and falls
back to the next one if a compile fails. The startup log shows the expansion, and `get_compiled_properties` reports it
as `compile_device`. To force a different order write it yourself, e.g. `"device": "AUTO:NPU,GPU,CPU"`, or name a
single device such as `"NPU"`.

`AUTO` also starts serving requests on the CPU while the accelerator compiles, then moves over. During that window
`execution_devices` reads `["(CPU)"]`, parentheses meaning "temporary", and the module logs when inference has moved
to the accelerator. Set `"extra_config": {"ENABLE_STARTUP_FALLBACK": "NO"}` to wait for the accelerator instead. The
NPU needs static input shapes; use `input_shape` for models exported with dynamic dimensions.

Validation happens when the config is saved, not at first inference: a missing model, a missing `.bin`, a bad device
string, an unknown hint or precision, or an inconsistent `preprocess` block all produce an error naming the attribute.

Changing any attribute rebuilds the compiled model. With the cache enabled this is fast after the first compile.

### Preprocessing

By default tensors are fed to the model as-is, so the caller has to send exactly what the model expects (typically
`float32 [1,3,H,W]`). With `preprocess`, resizing, layout and color conversion, dtype conversion and normalization are
compiled into the graph by OpenVINO's `PrePostProcessor` and run on the target device. The caller can then send raw
`uint8` pixels:

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
| `input_name` | Model input to attach to. Optional when the model has a single input. |
| `tensor_element_type` | dtype the caller sends: `u8`, `f32`, `f16`, and other OpenVINO element types. |
| `tensor_layout` / `model_layout` | Layout the caller sends and the layout the model expects, for example `NHWC` and `NCHW`. |
| `tensor_color_format` / `model_color_format` | `RGB` or `BGR`. A mismatch adds a conversion step. |
| `resize` | `none`, `linear`, `nearest` or `cubic`. When set, the caller-facing spatial dimensions become dynamic (`-1`) and any image size is accepted. |
| `mean` / `scale` | Per-channel, applied as `(x - mean) / scale`. |

Metadata always describes what the caller sends. With the block above, `Metadata` reports the input as
`uint8 [1, -1, -1, 3]`, even though the model itself consumes `float32 [1, 3, 640, 640]`.

Letterboxing is not part of preprocessing. `erh:openvino:yolo` does it, because it needs the padding offsets to map
boxes back to the original image.

### Metadata

`Metadata` returns the model file stem as `name`, `"openvino"` as `type`, and a description with the model path, the
requested device, the devices actually selected after compile, and the OpenVINO version. Each input and output has a
name, dtype and shape (`-1` for dynamic dimensions). When `label_path` is set, every output tensor carries
`extra: {"labels": "<path>"}`, the same convention as the `tflite_cpu` and `onnx-cpu` modules, so `vision:mlmodel`
finds the labels without extra configuration.

### Infer

- Input names must match the metadata. If they do not, the error lists the expected names. A model with a single
  input accepts a single tensor under any name (some callers send `image` generically).
- Safe dtype conversions are done for you (`uint8` to `float32`, `float64` to `float32`). Unsafe ones such as
  `float32` to `uint8` are rejected with an error naming the tensor and both dtypes.
- A batch dimension of 1 is added when the caller omits it.
- Calls are safe to make concurrently. Each call takes an infer request from the pool, so several vision services or
  cameras sharing one mlmodel run in parallel instead of queueing on a single request.
- Output tensors are copied out of OpenVINO memory before being returned.

### DoCommand

The `mlmodel` gRPC API only has `Infer`, `Metadata` and `GetStatus`, so clients cannot call `DoCommand` on an mlmodel
service directly. The commands below are therefore served by [`erh:openvino:diagnostics`](#viamopenvinodiagnostics)
and by the `DoCommand` of [`erh:openvino:yolo`](#viamopenvinoyolo), both of which forward to the mlmodel running in
the same module process. All commands take a JSON object with a `command` key.

| Command | Arguments | Returns |
|---|---|---|
| `list_devices` | | `{"devices": [{"name": "GPU", "full_name": "Intel(R) Iris(R) Xe Graphics", "type": "integrated", "capabilities": ["FP32", "FP16", "INT8", ...]}], "openvino_version": "..."}` |
| `get_compiled_properties` | | Requested device, device string actually compiled for, execution devices selected (important with `AUTO`), cache hit or miss, compile time, request pool size, and every property the compiled model reports (precision hint, streams, and per-device properties under `AUTO`). |
| `benchmark` | `iterations` (default 100), `warmup` (default 10), `input_shape` (required when an input has dynamic dims, e.g. `{"images": [1, 640, 640, 3]}`) | `mean_ms`, `p50_ms`, `p90_ms`, `p99_ms`, `min_ms`, `max_ms`, `throughput_fps`, `device`, `execution_devices`. Uses random inputs of the caller-facing shape and dtype. |
| `stats` | | Rolling latency over the last 1000 calls (`mean_ms`, `p50_ms`, `p90_ms`, `p99_ms`) plus `total_calls` and `total_errors` since the service started. |

### Logging and failure behavior

At startup the module logs the OpenVINO version, the available devices, the requested device, the devices actually
selected, whether the compile was served from the cache, and the compile time.

- An explicit `GPU` or `NPU` that is not available is a hard failure. The resource does not start, and the log names
  the missing driver. The module never silently falls back to CPU from an explicitly requested accelerator.
- With `AUTO` (or `AUTO:NPU,GPU,CPU`) a missing accelerator logs a warning and the model runs on whatever is available.
- A compile failure fails the resource with the OpenVINO error message.
- A corrupt cache entry is deleted and the model is recompiled, with a warning.

## `erh:openvino:yolo`

A vision service that turns raw YOLO outputs into Viam detections. It depends on an `mlmodel` service and works with
any implementation of that API, not only `erh:openvino:mlmodel`.

### Attributes

| Attribute | Type | Default | Required | Description |
|---|---|---|---|---|
| `mlmodel_name` | string | | yes | Name of the `mlmodel` service to call. |
| `camera_name` | string | | no | Default camera for `GetDetectionsFromCamera` and `CaptureAllFromCamera`. |
| `yolo_version` | string | `"auto"` | no | `v5`, `v8`, `v11` or `auto`. `auto` picks by output shape and label count and fails clearly if it cannot decide. |
| `input_size` | `[H, W]` | from metadata | no | Model input size. Only needed when the mlmodel reports dynamic spatial dimensions. |
| `confidence_threshold` | float | `0.25` | no | |
| `iou_threshold` | float | `0.45` | no | NMS IoU threshold. |
| `max_detections` | int | `100` | no | |
| `class_filter` | `[string]` | all | no | Only return these labels. |
| `label_path` | string | from mlmodel metadata | no | Overrides the labels the mlmodel advertises. |
| `input_format` | string | `"NCHW_f32_rgb_norm"` | no | Tensor sent to the mlmodel: `<NCHW|NHWC>_<u8|f32|f16>_<rgb|bgr>[_norm]`. Use `NHWC_u8_rgb` when the mlmodel has on-device `preprocess` configured, so raw pixels go over the wire. |

### Behavior

1. Get the image from the request or the camera.
2. Letterbox to the model input size (aspect ratio preserved, padding value 114).
3. Convert to `input_format` and call `Infer`.
4. Decode: v8/v11 outputs are `[1, 4 + C, N]` with `cx, cy, w, h` and class scores; v5 outputs are `[1, N, 5 + C]`
   with an objectness term that multiplies the class score.
5. Filter by confidence, run class-aware NMS, cap at `max_detections`.
6. Map boxes back to original pixel coordinates and clamp to the image.

`GetDetections`, `GetDetectionsFromCamera` and `CaptureAllFromCamera` (image plus detections) are implemented.
`GetProperties` reports detections only. Classification and point cloud calls return an unimplemented error.

`DoCommand {"command": "model_info"}` reports the input name and size, output names and shapes, label count, and the
YOLO version in use. The mlmodel diagnostics commands (`list_devices`, `get_compiled_properties`, `benchmark`, `stats`)
are also accepted and forwarded to the configured `mlmodel_name` when it is a `erh:openvino:mlmodel` in this module.

## `erh:openvino:diagnostics`

A generic service whose only job is `DoCommand`. Configure it next to an mlmodel:

```json
{
  "name": "ov-diag",
  "api": "rdk:service:generic",
  "model": "erh:openvino:diagnostics",
  "attributes": {"mlmodel_name": "ov"}
}
```

| Attribute | Type | Required | Description |
|---|---|---|---|
| `mlmodel_name` | string | no | The `erh:openvino:mlmodel` to inspect. `list_devices` works without it. A `mlmodel_name` key in the command overrides it per call. |

It accepts every command in the [DoCommand table](#docommand). Example from the Python SDK:

```python
diag = Generic.from_robot(machine, "ov-diag")
print(await diag.do_command({"command": "list_devices"}))
print(await diag.do_command({"command": "benchmark", "iterations": 200}))
```

## Host prerequisites for GPU and NPU

The module bundles the OpenVINO runtime and its device plugins, but not kernel drivers or the user-space compute
runtime. Install those on the host, restart viam-server, and confirm with `DoCommand {"command": "list_devices"}` on a
`erh:openvino:diagnostics` service. CPU inference needs nothing extra.

### Intel GPU (integrated Iris Xe / UHD, Arc)

Ubuntu 24.04:

```sh
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list
sudo apt-get update
sudo apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo
```

Ubuntu 22.04:

```sh
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt-get update
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero clinfo
```

On either release, the packaged `intel-opencl-icd` from Ubuntu's own repositories also works for older iGPUs:
`sudo apt-get install -y intel-opencl-icd`.

If viam-server runs as a non-root user, add it to the `render` and `video` groups:
`sudo usermod -aG render,video <user>`. Verify with `clinfo -l`, which should list an Intel platform and device.

### Intel NPU (Core Ultra)

Needs Linux kernel 6.8 or newer (the `intel_vpu` module, in stock Ubuntu 24.04 and the 22.04 HWE kernel) and the
user-space driver from <https://github.com/intel/linux-npu-driver/releases>. Download the `.deb` packages for your
Ubuntu release from the latest release page, then:

```sh
sudo apt-get install -y libtbb12
sudo dpkg -i intel-driver-compiler-npu_*.deb intel-fw-npu_*.deb intel-level-zero-npu_*.deb
# Level Zero loader, if not already present
sudo apt-get install -y libze1 || sudo dpkg -i level-zero_*.deb
# Let non-root users reach the device
sudo bash -c "echo 'SUBSYSTEM==\"accel\", KERNEL==\"accel*\", GROUP=\"render\", MODE=\"0660\"' > /etc/udev/rules.d/10-intel-vpu.rules"
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=accel
sudo usermod -aG render <user>
```

Confirm with `ls /dev/accel/` (expect `accel0`) and `list_devices` (expect an `NPU` entry). After a driver upgrade,
clear the module's cache directory so blobs are recompiled for the new driver.

### Windows

Install the current Intel graphics driver from <https://www.intel.com/content/www/us/en/download-center/home.html>
(or through Intel Driver & Support Assistant). It provides the OpenCL and Level Zero runtimes that the GPU plugin needs.
On Core Ultra machines the same driver package installs the NPU driver; Device Manager should show
"Intel(R) AI Boost" under Neural processors. Reboot, restart viam-server, and confirm with `list_devices`.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Log: `device 'GPU' cannot be used: 'GPU' is not in available devices ['CPU']` | Compute runtime missing or the user cannot open `/dev/dri/renderD*` | Install the packages above and fix group membership. |
| Log: `'NPU' is not in available devices` | NPU driver missing, kernel too old, or `/dev/accel/accel0` not accessible (Linux); driver not installed (Windows) | Install the NPU driver. On Linux check `dmesg \| grep -i vpu`. |
| `AUTO` runs on CPU although a GPU exists | Same as above; `AUTO` warns and continues | Check the startup warning and `list_devices`. |
| First start on GPU/NPU is slow | Model compilation | Expected once. Keep `cache_dir` enabled; the next start is a cache hit. |
| `input tensor names do not match the model` | Caller uses a different input name on a multi-input model | Use the names from `Metadata`. |

## Benchmarks

`scripts/benchmark_devices.py` compares CPU, GPU, NPU and AUTO on the machine it runs on, without viam-server. For
each device it does a cold compile, a second compile that should hit the cache, a serial latency run and a concurrent
throughput run, and it keeps going when a device is missing or fails to compile. Without `--model` it downloads a
public sample from the ONNX model zoo (MobileNetV2 by default, 14 MB) into `~/.cache/viam-openvino/models` on first
use; `--sample resnet50` and `--sample ssd-mobilenetv1` are also available. Dynamic dimensions such as the batch axis
are set to 1 unless overridden, because the NPU needs static shapes. It needs the same venv as the tests:

```sh
make venv
make bench                                                   # MobileNetV2 sample, Markdown table + bench-results.json
make bench MODEL=/path/to/yolov8n.onnx
.venv/bin/python scripts/benchmark_devices.py --sample resnet50 --devices CPU GPU NPU AUTO --iterations 200
.venv/bin/python scripts/benchmark_devices.py --model m.onnx --precision f16 --hint THROUGHPUT --concurrency 4
.venv/bin/python scripts/benchmark_devices.py --model m.onnx --input-shape images=1,3,640,640   # dynamic exports
```

Pass `--verbose` to see the module's own startup log lines (driver hints, execution devices, cache status). Paste
the Markdown table into the section below. Results with YOLOv8n (640x640, ONNX) are recorded as they are measured; the
maintainers have not yet run this on Intel GPU or NPU hardware.

| Machine | Device | mean ms | p50 ms | p99 ms | FPS | Cache hit on restart |
|---|---|---|---|---|---|---|
| (Intel iGPU machine) | CPU | | | | | |
| (Intel iGPU machine) | GPU | | | | | |
| (Core Ultra machine) | CPU | | | | | |
| (Core Ultra machine) | GPU | | | | | |
| (Core Ultra machine) | NPU | | | | | |
| (Core Ultra machine) | AUTO | | | | | |

## Development

```sh
make venv        # python3 -m venv .venv && pip install -r requirements-dev.txt
make test        # pytest (CPU only; runs in CI)
make build       # ./setup.sh && ./build.sh -> dist/archive.tar.gz, same as the Viam cloud build
```

OpenVINO publishes wheels for Python 3.9 through 3.13. If your default `python3` is newer, point the scripts at a
supported interpreter: `PYTHON=python3.12 ./setup.sh`. On macOS the frozen binary does not build (the OpenVINO wheel's
TBB libraries cannot be relinked by PyInstaller), so develop and run from the venv there; it is not a release target.

The test suite covers config validation, caller-facing metadata under preprocessing, YOLO v5/v8 decoding and NMS with
fixture tensors, letterbox inversion for non-square images, IR-versus-ONNX output equality, concurrent inference,
cache hits on reload, and an end-to-end run of `erh:openvino:mlmodel` plus `erh:openvino:yolo` on a synthetic
YOLOv8-shaped model with known boxes. A real YOLOv8n test runs when `YOLOV8N_ONNX` points at an ONNX export
(`make fetch-yolov8n` downloads one from `YOLOV8N_ONNX_URL`).

To run the module against a local viam-server without the registry, use the executable path
`dist/main` (or `.venv/bin/python src/main.py` during development) as a local module in the machine config.

Releases: pushing a tag such as `v0.1.0` runs `.github/workflows/deploy.yml`. Linux builds (`linux/amd64`,
`linux/arm64`, the architectures listed in `meta.json`) run in Viam's cloud build through the Viam build action. The
Windows build runs natively on a GitHub `windows-latest` runner (PyInstaller cannot cross-compile from the Linux cloud
build containers), is smoke-tested with `dist/main.exe --selftest`, and is uploaded as `windows/amd64` for the same
version with the viam CLI. Both jobs need the `viam_key_id` and `viam_key_value` repository secrets.
