"""Behavior under concurrent load: CPU work off the event loop, the in-process mlmodel fast path, and reconfigure
while calls are queued."""

import asyncio
import time

import numpy as np
from test_mlmodel_resource import make_config as mlmodel_config
from test_yolo_vision import FakeCamera, FakeMLModel, deps, make_config, make_image, v8_output

import models.yolo_vision as yolo_vision
from models.openvino_mlmodel import OpenVINOMLModel


async def test_preprocessing_does_not_block_event_loop(monkeypatch, labels_file):
    real_letterbox = yolo_vision.letterbox

    def slow_letterbox(*args, **kwargs):
        time.sleep(0.3)
        return real_letterbox(*args, **kwargs)

    monkeypatch.setattr(yolo_vision, "letterbox", slow_letterbox)
    ml = FakeMLModel("m", v8_output([(32, 32, 20, 20, 0, 0.9)], C=3), [1, 3, 64, 64], label_path=labels_file)
    v = yolo_vision.YoloVision.new(make_config(mlmodel_name="m", camera_name="cam"), deps(ml, FakeCamera("cam", make_image(64, 64))))

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    dets = await v.get_detections_from_camera("")
    t.cancel()
    assert [d.class_name for d in dets] == ["cat"]
    assert ticks >= 10, f"event loop was blocked during preprocessing ({ticks} ticks in 0.3 s)"


class RemoteOnlyMLModel(FakeMLModel):
    async def infer(self, input_tensors, *, extra=None, timeout=None):
        raise AssertionError("the gRPC dependency was called although the mlmodel runs in this process")


async def test_in_process_mlmodel_skips_grpc(fake_yolo_ir, labels_file, tmp_path):
    ml = OpenVINOMLModel.new(
        mlmodel_config(name="local", model_path=fake_yolo_ir, label_path=labels_file, device="CPU", cache_dir=str(tmp_path)), {}
    )
    try:
        client = RemoteOnlyMLModel("local", np.zeros((1, 7, 4), np.float32), [1, 3, 64, 64], label_path=labels_file)
        v = yolo_vision.YoloVision.new(make_config(mlmodel_name="local", camera_name="cam"), deps(client, FakeCamera("cam", make_image(64, 64))))
        dets = await v.get_detections_from_camera("")
        assert [d.class_name for d in dets] == ["cat", "dog"]
        assert (await ml.do_command({"command": "stats"}))["total_calls"] == 1
    finally:
        await ml.close()


async def test_queued_infers_survive_reconfigure(tiny_ir, tmp_path):
    cfg = dict(model_path=tiny_ir, device="CPU", cache_dir=str(tmp_path), num_requests=1)
    r = OpenVINOMLModel.new(mlmodel_config(**cfg), {})
    try:
        real_infer = r.engine.infer

        def slow_infer(feeds):
            out = real_infer(feeds)
            time.sleep(0.05)  # keep the single worker busy so the other calls stay queued across the reconfigure
            return out

        r.engine.infer = slow_infer
        x = np.zeros((1, 3, 8, 8), np.float32)
        calls = [asyncio.create_task(r.infer({"images": x})) for _ in range(5)]
        await asyncio.sleep(0.01)  # first call running, the rest queued on the single worker
        r.reconfigure(mlmodel_config(**cfg, performance_hint="THROUGHPUT"), {})
        outs = await asyncio.gather(*calls)
        assert all(o["output0"].shape == (1, 4, 8, 8) for o in outs)
    finally:
        await r.close()
