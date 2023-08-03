from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.optimizer import optimize_model
from transformers.convert_graph_to_onnx import quantize

from pipeline.onnx_manager.converters.base_converter import BaseConverter


class ONNXRuntimeConverter(BaseConverter):
    """
        Using technique to optimize onnx model using ONNXRuntime package
        """

    def __init__(self, dummy_inputs, model, input_names, output_names, dynamic_axes, use_gpu: bool = False):
        super().__init__(
            dummy_inputs=dummy_inputs,
            model=model,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, use_gpu=use_gpu
        )

    def apply_optimized(self, saved_path: str):
        """
        Note from original optimize_model function: "If your model is intended for GPU inference only (especially float16 or mixed precision model),
        it is recommended to set use_gpu to be True, otherwise the model is not optimized for GPU inference."

        :param saved_path:
        :return:
        """
        # if opt_level == 1 => onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # elif opt_level == 2 => onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # else => onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        optimized_model = optimize_model(saved_path, use_gpu=self.use_gpu, opt_level=99, verbose=True)
        new_onnx_path = saved_path.replace(".onnx", "_optimized.onnx")
        optimized_model.save_model_to_file(new_onnx_path)
        self.logger.info(f"EXPORT model to ONNX + Optimized - DONE at {new_onnx_path}")
        return new_onnx_path

    def apply_quantized(self, saved_path: str):
        quantized_model_path = quantize(Path(saved_path))
        # saved_path = Path(saved_path)
        # quantized_model_path = Path(saved_path.replace(".onnx", "_quantized.onnx"))
        # quantize_dynamic(saved_path, quantized_model_path, weight_type=QuantType.QInt8, optimize_model=True)
        self.logger.info(f"EXPORT model to ONNX + Quantized - DONE at {quantized_model_path}")
        return quantized_model_path


