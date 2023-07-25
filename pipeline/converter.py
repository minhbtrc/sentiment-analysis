from typing import Tuple, Any
import logging

from common.config import SentimentConfig, ONNXOptions
from onnx_converter import ONNXRuntimeConverter, OptimumConverter

CONVERTER = {
    "ONNXRUNTIME": ONNXRuntimeConverter,
    "Optimum": OptimumConverter
}


class ONNXConverter:
    def __init__(self, model, mode, config: SentimentConfig = None):
        self.config = config if config is not None else SentimentConfig()
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.onnx_func = CONVERTER[mode]

    @property
    def onnx_config(self):
        class OnnxConfig:
            input_keys = ["input_ids", "attention_mask"]
            output_keys = ["logits"]
            dynamic_axes = {
                "input_ids": {
                    0: "batch_size",
                    1: "hidden_dim"
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "hidden_dim"
                }
            }

        return OnnxConfig

    @staticmethod
    def __valid_options(option: str):
        if option not in ONNXOptions.keys():
            raise "ONNX option must be in `ONNXOptions`"

    def convert(self, option: str = None, dummy_inputs: Tuple[Any] = None, use_gpu: bool = False):
        self.__valid_options(option)

        converter = self.onnx_func(
            model=self.model,
            dummy_inputs=dummy_inputs,
            input_names=self.onnx_config.input_keys,
            output_names=self.onnx_config.output_keys,
            dynamic_axes=self.onnx_config.dynamic_axes,
            use_gpu=use_gpu
        )
        converter.convert(option=option, saved_path=self.config.onnx_path)

        self.logger.info(f"EXPORT {option} model to ONNX - DONE")
