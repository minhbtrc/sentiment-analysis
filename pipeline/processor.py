import torch
import os
import json

import regex as re
from evaluate import load
from transformers import AutoTokenizer, Trainer, SchedulerType
from transformers.training_args import TrainingArguments, OptimizerNames
from transformers.trainer_utils import IntervalStrategy, EvalPrediction

from common.config import SentimentConfig, ONNXOptions
from common.base import BaseModel
from pipeline.dataset_manager.dataset import SentimentDataset
from pipeline.model_manager.model import SentimentModel
from pipeline.onnx_manager.converter import ONNXConverter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class SentimentProcessor(BaseModel):
    def __init__(self, config: SentimentConfig = None, model_class: str = None):
        super(SentimentProcessor, self).__init__(config=config)

        self.converter = ONNXConverter(mode="ONNXRUNTIME", config=self.config)
        self.model = SentimentModel(config=self.config)
        self.model.model_class = model_class
        self.tokenizer = None
        self.metrics = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

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

    def dynamic_collator(self, data):
        batch_items = [item["input_ids"] for item in data]
        batch = self.tokenizer(batch_items, padding=True, max_length=self.config.model_input_max_length,
                               truncation=True, return_tensors="pt").to(self.config.training_device)
        batch["labels"] = torch.vstack([ele["labels"] for ele in data]).to(self.config.training_device)
        return batch

    def compute_metrics(self, eval_predictions: EvalPrediction):
        predictions, labels = eval_predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = torch.from_numpy(predictions).softmax(dim=-1).argmax(dim=-1)
        labels = torch.from_numpy(labels).argmax(dim=-1)

        output = {
            "precision": self.metrics["precision"].compute(predictions=predictions,
                                                           references=labels,
                                                           average="weighted")["precision"],
            "recall": self.metrics["recall"].compute(predictions=predictions,
                                                     references=labels,
                                                     average="weighted")["recall"],
            "f1": self.metrics["f1"].compute(predictions=predictions,
                                             references=labels,
                                             average="weighted")["f1"],
            "accuracy": self.metrics["accuracy"].compute(predictions=predictions,
                                                         references=labels)["accuracy"]
        }
        return output

    def init(self, use_lora: bool = False):
        self.metrics = {
            "precision": load("precision"),
            "recall": load("recall"),
            "f1": load("f1"),
            "accuracy": load("accuracy")
        }
        self.model.init(use_lora=use_lora)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_path)
        self.train_dataset, self.eval_dataset = self.load_dataset()
        self.init_trainer()

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
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.dynamic_collator,
            compute_metrics=self.compute_metrics
        )

    @property
    def best_checkpoint(self):
        _re_checkpoint = re.compile(r"^checkpoint\-(\d+)$")
        folder_checkpoint = self.config.training_output_dir
        best_checkpoint_until_now = ""

        if _re_checkpoint.search(folder_checkpoint.split("/")[-1]):
            return folder_checkpoint
        checkpoints = [path for path in os.listdir(folder_checkpoint) if _re_checkpoint.search(path) is not None and \
                       os.path.isdir(os.path.join(folder_checkpoint, path))]
        while True:
            if len(checkpoints) == 0:
                return folder_checkpoint
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))

            state_of_last_checkpoint = os.path.join(folder_checkpoint, last_checkpoint) + "/trainer_state.json"
            if not os.path.isfile(state_of_last_checkpoint):
                checkpoints.remove(last_checkpoint)
                continue
            best_checkpoint_until_now = json.load(open(state_of_last_checkpoint, "r", encoding="utf8"))[
                "best_model_checkpoint"]
            break
        return best_checkpoint_until_now if best_checkpoint_until_now is not None else folder_checkpoint

    def export_model_to_onnx(self, checkpoint_path: str = None, use_gpu: bool = False, adapter_path: str = None):
        def _get_dummy_inputs(example_text):
            _dummy_inputs = self.tokenizer(example_text, return_tensors="pt", padding=True)
            _dummy_inputs = self.model.get_dummy_inputs(_dummy_inputs)
            return _dummy_inputs

        ckpt_path = checkpoint_path if checkpoint_path is not None else self.best_checkpoint
        self.load_trained_model(checkpoint_path=ckpt_path, adapter_path=adapter_path)
        dummy_inputs = _get_dummy_inputs(example_text="Tôi là một kỹ_sư AI")
        self.converter.add_model(model=self.model)
        self.converter.convert(option=ONNXOptions.quantized_optimized_ONNX, dummy_inputs=dummy_inputs, use_gpu=use_gpu)

        self.logger.info(f"EXPORT model to ONNX - DONE")

    def train(self, export_last=False, export_mode: str = "last"):
        self.trainer.train()
        if export_last:
            # Default when convert to onnx, use_gpu is based on training_device
            self.export_model_to_onnx(export_mode, use_gpu=self.config.training_device)

    def eval(self):
        self.logger.info(self.trainer.evaluate(eval_dataset=self.eval_dataset))

    def load_trained_model(self, checkpoint_path: str = None, use_lora: bool = None, adapter_path: str = None):
        self.model.load_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model.use_lora = use_lora
        if self.model.use_lora and adapter_path is not None:
            self.model.load_lora_adapter(adapter_path=adapter_path)


if __name__ == "__main__":
    trainer = SentimentProcessor(model_class="bloom")
    trainer.init(use_lora=True)
    trainer.train()
    # trainer.export_model_to_onnx(checkpoint_path=trainer.config.pretrained_path,
    #                              adapter_path=f"{trainer.config.training_output_dir}/checkpoint-8000")
