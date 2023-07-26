from transformers import BloomForSequenceClassification


class SentimentModelWithBloom(BloomForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

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
