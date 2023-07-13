import torch
import onnx
import os
import json
import logging

import regex as re
from evaluate import load
from transformers.trainer_utils import EvalPrediction
from transformers import RobertaForSequenceClassification, AutoTokenizer, Trainer, SchedulerType
from transformers.training_args import TrainingArguments, OptimizerNames
from transformers.trainer_utils import IntervalStrategy

from common.config import SentimentConfig
from dataset.dataset import SentimentDataset


class SentimentTrainer:
    def __init__(self, config: SentimentConfig = None):
        self.config = config if config is not None else SentimentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = RobertaForSequenceClassification.from_pretrained(self.config.pretrained_path,
                                                                      num_labels=self.config.model_num_labels)
        self.model.to(self.config.training_device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_path)
        self.metrics = load("f1")
        self.train_dataset, self.eval_dataset = self.load_dataset()
        self.trainer = None
        self.init_trainer()

    def load_dataset(self):
        train_dataset = SentimentDataset(config=self.config, mode="train", tokenizer=self.tokenizer)
        eval_dataset = SentimentDataset(config=self.config, mode="eval", tokenizer=self.tokenizer)
        return train_dataset, eval_dataset

    def collator(self, data):
        batch = {}
        _keys = data[0].keys()
        for key in _keys:
            batch[key] = torch.vstack([ele[key] for ele in data]).to(self.config.training_device)

        return batch

    def compute_metrics(self, eval_predictions: EvalPrediction):
        predictions, labels = eval_predictions
        predictions = torch.from_numpy(predictions).softmax(dim=-1).argmax(dim=-1)
        result = self.metrics.compute(predictions=predictions, references=torch.from_numpy(labels).argmax(dim=-1),
                                      average="weighted")
        return result

    def init_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.config.training_output_dir,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_strategy=IntervalStrategy.STEPS,
            per_device_train_batch_size=self.config.training_batch_size,
            per_device_eval_batch_size=self.config.training_batch_size,
            learning_rate=self.config.training_learning_rate,
            weight_decay=self.config.training_weight_decay,
            save_total_limit=self.config.training_save_total_limit,
            num_train_epochs=self.config.training_num_epochs,
            gradient_accumulation_steps=self.config.training_gradient_accumulation_steps,
            eval_steps=self.config.training_eval_steps,
            fp16=self.config.training_device == "cuda",
            push_to_hub=False,
            dataloader_num_workers=0,
            logging_dir=self.config.training_logging_dir,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=self.config.training_logging_steps,
            save_steps=self.config.training_save_steps,
            logging_first_step=False,
            optim=OptimizerNames.ADAMW_TORCH,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.training_metrics,
            warmup_ratio=self.config.training_warm_up_ratio,
            lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
            dataloader_pin_memory=False
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )

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

    def get_best_checkpoint(self, mode: str = "best"):
        _re_checkpoint = re.compile(r"^checkpoint\-(\d+)$")
        folder_checkpoint = self.config.training_output_dir
        best_checkpoint_until_now = ""

        if mode not in ["best", "last"]:
            self.logger.error(f'>>>`checkpoint_type` must be in ["best", "last"], not {mode}<<<')

        if _re_checkpoint.search(folder_checkpoint.split("/")[-1]):
            return folder_checkpoint
        checkpoints = [path for path in os.listdir(folder_checkpoint) if _re_checkpoint.search(path) is not None and \
                       os.path.isdir(os.path.join(folder_checkpoint, path))]
        while True:
            if len(checkpoints) == 0:
                return folder_checkpoint
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            if mode == "last":
                return os.path.join(folder_checkpoint, last_checkpoint)
            state_of_last_checkpoint = os.path.join(folder_checkpoint, last_checkpoint) + "/trainer_state.json"
            if not os.path.isfile(state_of_last_checkpoint):
                checkpoints.remove(last_checkpoint)
                continue
            best_checkpoint_until_now = json.load(open(state_of_last_checkpoint, "r", encoding="utf8"))[
                "best_model_checkpoint"]
            break
        return best_checkpoint_until_now if best_checkpoint_until_now is not None else folder_checkpoint

    def export_model_to_onnx(self, mode: str = None):
        def get_dummy_inputs(example_text):
            _dummy_inputs = self.tokenizer(example_text, return_tensors="pt", padding="max_length", max_length=256,
                                           truncation=True)
            _dummy_inputs = tuple(_dummy_inputs[i] for i in self.onnx_config.input_keys)
            return _dummy_inputs

        def check_onnx_model():
            model = onnx.load(self.config.onnx_path)
            onnx.checker.check_model(model)
            self.logger.info(f"CHECK exported model to ONNX - DONE")

        if mode is None:
            mode = "best"
        checkpoint_path = self.get_best_checkpoint(mode=mode)
        onnx_model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
        dummy_inputs = get_dummy_inputs("Tôi là một kỹ sư AI")
        torch.onnx.export(
            onnx_model,
            dummy_inputs,
            self.config.onnx_path,
            verbose=True,
            input_names=self.onnx_config.input_keys,
            output_names=self.onnx_config.output_keys,
            dynamic_axes=self.onnx_config.dynamic_axes
        )
        self.logger.info(f"EXPORT {mode} model to ONNX - DONE")
        check_onnx_model()

    def train(self, export_last=False, export_mode: str = "last"):
        self.trainer.train()
        if export_last:
            self.export_model_to_onnx(export_mode)

    def eval(self):
        print(self.trainer.evaluate(eval_dataset=self.eval_dataset))


if __name__ == "__main__":
    trainer = SentimentTrainer()
    trainer.train()
