import torch
import logging
from functools import reduce


class BaseConverter:
    """
    NOTE:
        - Quantize of onnxruntime using operators corresponding to each of Ops: (IntegerOps, QLinearOps, QDQOps)
        https://github.com/microsoft/onnxruntime/blob/48647bc7d788699b5eef35bcaea35f320263cb22/onnxruntime/python/tools/quantization/registry.py#L29

        - So if you want to try to both of onnxruntime and optimum, you should config same operators for each of them
    """

    def __init__(self, dummy_inputs, model, input_names, output_names, dynamic_axes, use_gpu: bool = False):
        self.dummy_inputs = dummy_inputs
        self.model = model
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def compose(*functions):
        return reduce(lambda f, g: lambda x: g(f(x)), functions)

    @property
    def mapping_func(self):
        return {
            "ONNX": self.onnx_only,
            "quantized_optimized_ONNX": self.compose(
                self.onnx_only, self.apply_optimized, self.apply_quantized)
        }

    def onnx_only(self, saved_path):
        torch.onnx.export(
            self.model,
            self.dummy_inputs,
            saved_path,
            verbose=True,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes
        )
        self.logger.info(f"EXPORT model to ONNX - DONE at {saved_path}")
        return saved_path

    def convert(self, option, saved_path):
        converter = self.mapping_func[option]
        return converter(saved_path)

    def apply_optimized(self, saved_path: str):
        raise "This func need implementation"

    def apply_quantized(self, saved_path: str):
        raise "This func need implementation"
