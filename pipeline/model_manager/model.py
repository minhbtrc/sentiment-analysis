import torch
from typing import Any, Dict

from transformers import RobertaForSequenceClassification
from peft import (
    LoraConfig,
    get_peft_model
)
from common.config import SentimentConfig
from common.base import BaseModel
from pipeline.model_manager import (
    SentimentModelWithLinear,
    SentimentModelWithLSTM,
    SentimentModelWithBloom
)
from pipeline.model_manager import LORA_TARGET_MODULES, LORA_TASK_TYPE

MODEL = {
    "roberta_linear": SentimentModelWithLinear,
    "roberta_lstm": SentimentModelWithLSTM,
    "bloom": SentimentModelWithBloom
}


class SentimentModel(BaseModel):
    def __init__(self, config: SentimentConfig = None):
        super(SentimentModel, self).__init__(config=config)
        self._model_class = None
        self.model = None
        self._use_lora = None

    @property
    def use_lora(self):
        return self._use_lora

    @use_lora.setter
    def use_lora(self, _use: bool):
        self._use_lora = _use

    @property
    def model_class(self):
        return MODEL[self._model_class]

    @property
    def model_class_str(self):
        return self._model_class

    @model_class.setter
    def model_class(self, _model_class: str = None):
        if _model_class is not None:
            self._model_class = _model_class

    def load_pretrained(self, checkpoint_path):
        self.model = self.model_class.from_pretrained(checkpoint_path, num_labels=self.config.model_num_labels)

    def init(self, use_lora: bool = False, model_class: str = None):
        self.model_class = model_class
        self.logger.info(f"Loading model from {self.model_class}-{self.config.pretrained_path} checkpoint")
        self.model = self.model_class.from_pretrained(
            self.config.pretrained_path,
            num_labels=self.config.model_num_labels
        )
        self.model.use_cache = False
        self.model.to(self.config.training_device)
        self.use_lora = use_lora
        if use_lora:
            self.apply_lora()

    def apply_lora(self):
        # This target modules value base on name in base model definition
        target_modules = LORA_TARGET_MODULES.get(self.model_class_str.split("_")[0], [])
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=LORA_TASK_TYPE.get(self.model_class_str.split("_")[0], ""),
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.logger.info("============APPLY LORA DONE==============")

    def load_lora_adapter(self, adapter_path, adapter_name: str = "default"):
        self.logger.info(f"==============LOADING_LORA_ADAPTER==============")
        if not self.use_lora:
            self.logger.warn("Model is not trained with lora")
        self.model.load_adapter(model_id=adapter_path, adapter_name=adapter_name)

    @property
    def onnx_config(self):
        return self.model.onnx_config

    def get_dummy_inputs(self, _dummy_inputs: Dict[str, Any]):
        return tuple(_dummy_inputs[i] for i in self.onnx_config.input_keys)
