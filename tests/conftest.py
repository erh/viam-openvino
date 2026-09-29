import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import openvino as ov  # noqa: E402
import openvino.opset13 as ops  # noqa: E402


def _tiny_weights():
    rng = np.random.default_rng(1234)
    return rng.standard_normal((4, 3, 3, 3)).astype(np.float32)


def build_tiny_ov_model(weights: np.ndarray, input_name: str = "images", output_name: str = "output0") -> ov.Model:
    p = ops.parameter([1, 3, 8, 8], ov.Type.f32, name=input_name)
    w = ops.constant(weights)
    c = ops.convolution(p, w, [1, 1], [1, 1], [1, 1], [1, 1])
    r = ops.relu(c)
    r.output(0).get_tensor().set_names({output_name})
    return ov.Model([r], [p], "tiny")


def build_tiny_onnx(weights: np.ndarray, path: str, input_name: str = "images", output_name: str = "output0") -> str:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(weights, name="W")
    conv = helper.make_node("Conv", [input_name, "W"], ["conv_out"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    relu = helper.make_node("Relu", ["conv_out"], [output_name])
    graph = helper.make_graph(
        [conv, relu], "tiny",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


@pytest.fixture(scope="session")
def tiny_weights():
    return _tiny_weights()


@pytest.fixture(scope="session")
def tiny_ir(tmp_path_factory, tiny_weights):
    d = tmp_path_factory.mktemp("ir")
    xml = str(d / "tiny.xml")
    # Keep f32 weights: save_model compresses to f16 by default, which makes IR and ONNX outputs differ.
    ov.save_model(build_tiny_ov_model(tiny_weights), xml, compress_to_fp16=False)
    return xml


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory, tiny_weights):
    d = tmp_path_factory.mktemp("onnx")
    return build_tiny_onnx(tiny_weights, str(d / "tiny.onnx"))


@pytest.fixture(scope="session")
def labels_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("labels")
    p = d / "labels.txt"
    p.write_text("cat\ndog\nbird\n")
    return str(p)


def build_fake_yolo_model(output: np.ndarray, input_hw=(64, 64), input_name="images", output_name="output0") -> ov.Model:
    """A model with a YOLO-shaped input that always returns ``output`` (plus 0 * sum(input))."""
    p = ops.parameter([1, 3, input_hw[0], input_hw[1]], ov.Type.f32, name=input_name)
    s = ops.reduce_sum(p, ops.constant(np.array([0, 1, 2, 3], dtype=np.int64)), keep_dims=False)
    z = ops.multiply(s, ops.constant(np.float32(0.0)))
    out = ops.add(ops.constant(output.astype(np.float32)), z)
    out.output(0).get_tensor().set_names({output_name})
    return ov.Model([out], [p], "fake_yolo")


@pytest.fixture(scope="session")
def fake_yolo_ir(tmp_path_factory):
    """v8-style [1, 4+C, N] output with C=3 classes and 4 anchors in a 64x64 letterboxed frame.

    Anchor 0: cat box   cx=32 cy=32 w=20 h=20 score 0.9
    Anchor 1: cat box   cx=33 cy=33 w=20 h=20 score 0.6   (suppressed by NMS)
    Anchor 2: dog box   cx=10 cy=50 w=8  h=8  score 0.8
    Anchor 3: bird      low score 0.1                    (below threshold)
    """
    C, N = 3, 4
    out = np.zeros((1, 4 + C, N), dtype=np.float32)
    out[0, :4, 0] = [32, 32, 20, 20]
    out[0, 4 + 0, 0] = 0.9
    out[0, :4, 1] = [33, 33, 20, 20]
    out[0, 4 + 0, 1] = 0.6
    out[0, :4, 2] = [10, 50, 8, 8]
    out[0, 4 + 1, 2] = 0.8
    out[0, :4, 3] = [50, 10, 8, 8]
    out[0, 4 + 2, 3] = 0.1
    d = tmp_path_factory.mktemp("fake_yolo")
    xml = str(d / "fake_yolo.xml")
    ov.save_model(build_fake_yolo_model(out), xml)
    return xml
