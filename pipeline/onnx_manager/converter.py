from typing import Tuple, Any

import onnx

from common.base import BaseModel
from common.config import SentimentConfig, ONNXOptions
from pipeline.model_manager.model import SentimentModel
from pipeline.onnx_manager import ONNXRuntimeConverter, OptimumConverter

CONVERTER = {
    "ONNXRUNTIME": ONNXRuntimeConverter,
    "Optimum": OptimumConverter
}


class ONNXConverter(BaseModel):
    def __init__(self, model: SentimentModel = None, mode: str = None, config: SentimentConfig = None):
        super(ONNXConverter, self).__init__(config=config)
        self.model = model
        self.onnx_func = CONVERTER.get(mode, ONNXRuntimeConverter)

    def add_model(self, model: SentimentModel):
        self.model = model

    @property
    def onnx_config(self):
        return self.model.onnx_config

    @staticmethod
    def __valid_options(option: str):
        if option not in ONNXOptions.keys():
            raise "ONNX option must be in `ONNXOptions`"

    def check_onnx_model(self, path):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        self.logger.info(f"CHECK exported model to ONNX - DONE")

    def convert(self, option: str = None, dummy_inputs: Tuple[Any] = None, use_gpu: bool = False):
        self.__valid_options(option)

        converter = self.onnx_func(
            model=self.model.model,
            dummy_inputs=dummy_inputs,
            input_names=self.onnx_config.input_keys,
            output_names=self.onnx_config.output_keys,
            dynamic_axes=self.onnx_config.dynamic_axes,
            use_gpu=use_gpu
        )
        converter.convert(option=option, saved_path=self.config.onnx_path)

        self.logger.info(f"EXPORT {option} model to ONNX - DONE")
