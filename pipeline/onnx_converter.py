from typing import Tuple, Any
import torch
import logging
from functools import reduce

from onnxruntime.transformers.optimizer import optimize_model
from transformers.convert_graph_to_onnx import quantize

from common.config import SentimentConfig, ONNXOptions


class ONNXFunc:
    def __init__(self, dummy_inputs, model, input_names, output_names, dynamic_axes):
        self.dummy_inputs = dummy_inputs
        self.model = model
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def compose(*functions):
        return reduce(lambda f, g: lambda x: g(f(x)), functions)

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

    def apply_optimized_func(self, saved_path: str):
        optimized_model = optimize_model(saved_path)
        new_onnx_path = saved_path.replace(".onnx", "_optimized.onnx")
        optimized_model.save_model_to_file(new_onnx_path)
        self.logger.info(f"EXPORT model to ONNX + Optimized - DONE at {new_onnx_path}")
        return new_onnx_path

    def apply_quantized_func(self, saved_path: str):
        from pathlib import Path
        quantized_model_path = quantize(Path(saved_path))
        self.logger.info(f"EXPORT model to ONNX + Optimized + Quantized - DONE at {quantized_model_path}")
        return quantized_model_path

    @property
    def mapping_func(self):
        return {
            "ONNX": self.onnx_only,
            "optimized_ONNX": self.compose(self.onnx_only, self.apply_optimized_func),
            "quantized_optimized_ONNX": self.compose(
                self.onnx_only, self.apply_optimized_func, self.apply_quantized_func)
        }

    def convert(self, option, saved_path):
        converter = self.mapping_func[option]
        return converter(saved_path)


class ONNXConverter:
    def __init__(self, tokenizer, model, config: SentimentConfig = None):
        self.config = config if config is not None else SentimentConfig()
        self.tokenizer = tokenizer
        self.model = model
        self.logger = self.logger = logging.getLogger(self.__class__.__name__)
        self.onnx_func = ONNXFunc()

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

    def convert(self, option: str = None, dummy_inputs: Tuple[Any] = None):
        self.__valid_options(option)

        converter = ONNXFunc(
            model=self.model,
            dummy_inputs=dummy_inputs,
            input_names=self.onnx_config.input_keys,
            output_names=self.onnx_config.output_keys,
            dynamic_axes=self.onnx_config.dynamic_axes
        )
        converter.convert(option=option, saved_path=self.config.onnx_path)

        self.logger.info(f"EXPORT {option} model to ONNX - DONE")
