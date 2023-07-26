from pipeline.onnx_manager.converters.base_converter import BaseConverter


class OptimumConverter(BaseConverter):
    """
    Using technique to optimize onnx model using Optimum package
    """

    def __init__(self, dummy_inputs, model, input_names, output_names, dynamic_axes, use_gpu: bool = False, **kwargs):
        super().__init__(
            dummy_inputs=dummy_inputs,
            model=model,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, use_gpu=use_gpu
        )

    def apply_optimized(self, saved_path: str):
        raise "This func need implementation"

    def apply_quantized(self, saved_path: str):
        raise "This func need implementation"
