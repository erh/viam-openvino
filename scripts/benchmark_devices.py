#!/usr/bin/env python3
"""Benchmark a model on CPU, GPU, NPU and AUTO with the module's own engine.

Runs without viam-server. For each device it compiles the model twice (cold compile, then a second
compile that should hit the cache), runs a warm-up, then times ``--iterations`` serial inferences and
a concurrent run with ``--concurrency`` threads (or the device's optimal request count).

Without --model it downloads a small public ONNX sample (MobileNetV2 from the ONNX model zoo) into
~/.cache/viam-openvino/models the first time and reuses it afterwards. Dynamic dimensions such as a
batch axis are set to 1 unless overridden with --input-shape.

Examples (from the repo root, after ./setup.sh or make venv):

    .venv/bin/python scripts/benchmark_devices.py                      # MobileNetV2 sample on CPU/GPU/NPU/AUTO
    .venv/bin/python scripts/benchmark_devices.py --sample resnet50
    .venv/bin/python scripts/benchmark_devices.py --model /path/to/yolov8n.onnx
    .venv/bin/python scripts/benchmark_devices.py --model model.xml --devices CPU GPU NPU AUTO --iterations 200
    .venv/bin/python scripts/benchmark_devices.py --model model.onnx --input-shape images=1,3,640,640 --precision f16
    .venv/bin/python scripts/benchmark_devices.py --model model.onnx --markdown >> README.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.modules["openvino_telemetry"] = None  # type: ignore[assignment]  # same opt-out as src/main.py

import openvino as ov  # noqa: E402

from models.config import MLModelConfig, parse_device  # noqa: E402
from models.engine import OpenVINOEngine, list_devices  # noqa: E402

DEFAULT_DEVICES = ["CPU", "GPU", "NPU", "AUTO"]

# Public sample models (ONNX model zoo, served through GitHub LFS). All have a dynamic batch axis.
SAMPLE_MODELS = {
    "mobilenetv2": (
        "mobilenetv2-12.onnx",
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    ),
    "resnet50": (
        "resnet50-v2-7.onnx",
        "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx",
    ),
    "ssd-mobilenetv1": (
        "ssd_mobilenet_v1_12.onnx",
        "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx",
    ),
}
DEFAULT_SAMPLE = "mobilenetv2"
SAMPLE_INPUT_SHAPES = {"ssd-mobilenetv1": {"image_tensor:0": [1, 640, 640, 3]}}


def sample_model_path(name: str, model_dir: str) -> str:
    """Return the local path of a sample model, downloading it if it is not there yet."""
    filename, url = SAMPLE_MODELS[name]
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    import urllib.request

    print(f"downloading sample model {name} from {url} ...", flush=True)
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(f"could not download {url}: {e}\nDownload it manually and pass --model {path}") from e
    print(f"saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def parse_shape_args(values: List[str]) -> Dict[str, List[int]]:
    shapes: Dict[str, List[int]] = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--input-shape expects name=d0,d1,..., got '{v}'")
        name, dims = v.split("=", 1)
        shapes[name] = [int(d) for d in dims.split(",")]
    return shapes


def random_feeds(engine: OpenVINOEngine, shapes: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    feeds: Dict[str, np.ndarray] = {}
    for spec in engine.inputs:
        shape = shapes.get(spec.name, list(spec.shape))
        if any(d < 0 for d in shape):
            raise ValueError(f"input '{spec.name}' has dynamic shape {spec.shape}; pass --input-shape {spec.name}=...")
        dtype = spec.np_dtype or np.float32
        if np.issubdtype(dtype, np.integer):
            feeds[spec.name] = rng.integers(0, 256, size=shape).astype(dtype)
        else:
            feeds[spec.name] = rng.random(size=shape, dtype=np.float32).astype(dtype)
    return feeds


def resolve_input_shapes(args) -> Dict[str, List[int]]:
    """Explicit --input-shape values, plus every remaining dynamic dimension of the model set to 1."""
    shapes = dict(parse_shape_args(args.input_shape))
    if args.sample_shapes:
        for k, v in args.sample_shapes.items():
            shapes.setdefault(k, v)
    core = ov.Core()
    model = core.read_model(args.model)
    for i, inp in enumerate(model.inputs):
        try:
            name = inp.get_any_name()
        except Exception:
            name = f"input{i}"
        if name in shapes:
            continue
        ps = inp.get_partial_shape()
        if ps.rank.is_dynamic:
            continue
        dims = [(-1 if d.is_dynamic else d.get_length()) for d in ps]
        if any(d < 0 for d in dims):
            fixed = [1 if d < 0 else d for d in dims]
            print(f"input '{name}' has dynamic shape {dims}; using {fixed} (override with --input-shape {name}=...)")
            shapes[name] = fixed
    return shapes


def make_engine(args, device: str, cache_dir: str) -> OpenVINOEngine:
    attrs: Dict[str, Any] = {
        "model_path": args.model,
        "device": device,
        "performance_hint": args.hint,
        "cache_dir": cache_dir,
    }
    if args.precision:
        attrs["inference_precision"] = args.precision
    if args.num_requests:
        attrs["num_requests"] = args.num_requests
    if args.resolved_shapes:
        attrs["input_shape"] = args.resolved_shapes
    engine = OpenVINOEngine(MLModelConfig.from_attributes(attrs), logging.getLogger("bench"))
    engine.load()
    return engine


def bench_device(args, device: str, cache_root: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"device": device}
    cache_dir = os.path.join(cache_root, device.replace(":", "_").replace(",", "_"))
    shutil.rmtree(cache_dir, ignore_errors=True)

    engine = make_engine(args, device, cache_dir)  # cold compile
    result["execution_devices"] = engine.execution_devices
    result["compile_cold_ms"] = engine.compile_time_ms
    result["cold_from_cache"] = engine.loaded_from_cache
    result["num_requests"] = engine.num_requests
    engine.close()

    engine = make_engine(args, device, cache_dir)  # should be a cache hit
    result["compile_cached_ms"] = engine.compile_time_ms
    result["cache_hit"] = engine.loaded_from_cache
    try:
        props = engine.compiled_properties().get("properties", {})
        for key in ("INFERENCE_PRECISION_HINT", "NUM_STREAMS", "OPTIMAL_NUMBER_OF_INFER_REQUESTS"):
            if key in props:
                result[key.lower()] = props[key]
        dev_props = props.get("DEVICE_PROPERTIES")
        if isinstance(dev_props, dict):
            result["device_properties"] = dev_props
    except Exception:
        pass

    feeds = random_feeds(engine, args.resolved_shapes)
    result["input_shape"] = {k: list(v.shape) for k, v in feeds.items()}
    try:
        for _ in range(args.warmup):
            engine.infer(feeds)
        lat = []
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            t = time.perf_counter()
            engine.infer(feeds)
            lat.append((time.perf_counter() - t) * 1000)
        wall = time.perf_counter() - t0
        arr = np.asarray(lat)
        result.update({
            "serial_mean_ms": float(arr.mean()),
            "serial_p50_ms": float(np.percentile(arr, 50)),
            "serial_p90_ms": float(np.percentile(arr, 90)),
            "serial_p99_ms": float(np.percentile(arr, 99)),
            "serial_fps": args.iterations / wall,
        })

        workers = args.concurrency or engine.num_requests
        if workers > 1:
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda _: engine.infer(feeds), range(args.iterations)))
            wall = time.perf_counter() - t0
            result["concurrent_workers"] = workers
            result["concurrent_fps"] = args.iterations / wall
        else:
            result["concurrent_workers"] = 1
            result["concurrent_fps"] = result["serial_fps"]
    finally:
        engine.close()
    return result


def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(rows: List[Dict[str, Any]], markdown: bool) -> None:
    cols = [
        ("device", "Device"), ("execution_devices", "Runs on"), ("serial_mean_ms", "mean ms"), ("serial_p50_ms", "p50 ms"),
        ("serial_p99_ms", "p99 ms"), ("serial_fps", "FPS serial"), ("concurrent_fps", "FPS concurrent"),
        ("concurrent_workers", "workers"), ("compile_cold_ms", "compile ms"), ("compile_cached_ms", "cached compile ms"),
        ("cache_hit", "cache hit"), ("error", "error"),
    ]
    table = []
    for r in rows:
        line = []
        for key, _ in cols:
            v = r.get(key)
            if isinstance(v, list):
                v = ",".join(map(str, v))
            line.append(fmt(v))
        table.append(line)
    headers = [h for _, h in cols]
    if markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for line in table:
            print("| " + " | ".join(line) + " |")
        return
    widths = [max(len(h), *(len(line[i]) for line in table)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for line in table:
        print("  ".join(c.ljust(w) for c, w in zip(line, widths)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model path (.onnx, .xml, .tflite, ...); default: a downloaded sample")
    ap.add_argument("--sample", default=DEFAULT_SAMPLE, choices=sorted(SAMPLE_MODELS), help=f"public sample model to use when --model is not given (default: {DEFAULT_SAMPLE})")
    ap.add_argument("--model-dir", default=os.environ.get("VIAM_OPENVINO_MODEL_DIR", os.path.expanduser("~/.cache/viam-openvino/models")), help="where sample models are downloaded")
    ap.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES, help=f"devices to test (default: {' '.join(DEFAULT_DEVICES)})")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"])
    ap.add_argument("--precision", default=None, choices=["f32", "f16", "bf16"], help="inference_precision for every device")
    ap.add_argument("--num-requests", type=int, default=None, help="infer request pool size (default: device optimal)")
    ap.add_argument("--concurrency", type=int, default=None, help="threads for the concurrent run (default: pool size)")
    ap.add_argument("--input-shape", nargs="*", default=[], help="name=d0,d1,... for dynamic inputs, e.g. images=1,3,640,640")
    ap.add_argument("--cache-dir", default=None, help="cache root (default: a temp dir, deleted afterwards)")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table for the README")
    ap.add_argument("--json", dest="json_out", default=None, help="also write full results to this JSON file")
    ap.add_argument("--all", action="store_true", help="also try devices that are not listed as available")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.sample_shapes = None
    if args.model is None:
        args.model = sample_model_path(args.sample, args.model_dir)
        args.sample_shapes = SAMPLE_INPUT_SHAPES.get(args.sample)
    elif not os.path.exists(args.model):
        raise SystemExit(f"model '{args.model}' does not exist")
    args.resolved_shapes = resolve_input_shapes(args)
    info = list_devices()
    available = [d["name"] for d in info["devices"]]
    print(f"OpenVINO {info['openvino_version']} on {platform.node()} ({platform.machine()}, {platform.system()} {platform.release()})")
    for d in info["devices"]:
        print(f"  {d['name']:<8} {d.get('full_name', '')}  [{', '.join(d.get('capabilities', []))}]")
    print(f"model: {args.model}  hint={args.hint} precision={args.precision or 'default'} iterations={args.iterations}\n")

    tmp_cache = None
    cache_root = args.cache_dir
    if cache_root is None:
        tmp_cache = tempfile.mkdtemp(prefix="ov-bench-")
        cache_root = tmp_cache

    rows: List[Dict[str, Any]] = []
    try:
        for device in args.devices:
            parsed = parse_device(device)
            missing = [d for d in parsed.devices if d not in available and d.split(".")[0] not in available]
            if missing and not parsed.is_auto and not args.all:
                print(f"[{device}] skipped: {missing} not in available devices {available} (drivers missing?)")
                rows.append({"device": device, "error": f"unavailable: {','.join(missing)}"})
                continue
            print(f"[{device}] compiling and benchmarking...", flush=True)
            try:
                r = bench_device(args, device, cache_root)
                rows.append(r)
                print(
                    f"[{device}] runs on {r['execution_devices']}: mean {r['serial_mean_ms']:.2f} ms, p99 {r['serial_p99_ms']:.2f} ms, "
                    f"{r['serial_fps']:.1f} fps serial, {r['concurrent_fps']:.1f} fps with {r['concurrent_workers']} workers; "
                    f"compile {r['compile_cold_ms']:.0f} ms cold / {r['compile_cached_ms']:.0f} ms cached (hit={r['cache_hit']})"
                )
            except Exception as e:  # keep going so one broken device does not hide the others
                msg = str(e).splitlines()[0][:200]
                print(f"[{device}] FAILED: {msg}")
                rows.append({"device": device, "error": msg})
    finally:
        if tmp_cache:
            shutil.rmtree(tmp_cache, ignore_errors=True)

    print()
    print_table(rows, args.markdown)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"host": platform.uname()._asdict(), "openvino": info, "args": vars(args), "results": rows}, f, indent=2, default=str)
        print(f"\nfull results written to {args.json_out}")
    return 0 if any("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
