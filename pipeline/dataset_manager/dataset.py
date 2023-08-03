import os
import json

import torch
from collections import Counter
import random
from torch.utils.data import Dataset

from pipeline.dataset_manager.config import SentimentId2Tag

random.seed(2023)


class SentimentDataset(Dataset):
    def __init__(self, config, mode, tokenizer, num_folds: int = None):
        super().__init__()
        self.config = config
        self.num_folds = num_folds
        self.tokenizer = tokenizer
        self.original_data = self.load_dataset(mode=mode)
        if num_folds is None:
            self.data = self.original_data
        else:
            self.data = None
        print(Counter([e[1] for e in self.data]))
        print(f"Load {len(self.data)} examples for {mode} dataset")

    @staticmethod
    def generate_prompt(sentence):
        return f"""Classify the sentiment of the following Vietnamese sentence into one of three classes: neutral, negative, positive.\n### Input: {sentence}"""

    @staticmethod
    def generate_label(label_id):
        return SentimentId2Tag.get_tag(label_id)

    def apply_fold(self, fold_id):
        if fold_id >= self.num_folds:
            raise f"`num_folds` must larger than `fold_id`"
        total_len = len(self.original_data)
        base_indices = total_len // self.num_folds
        fold_eval = self.original_data[fold_id * base_indices:(fold_id + 1) * base_indices]
        self.data = self.original_data[:fold_id * base_indices] + self.original_data[(fold_id + 1) * base_indices:]
        eval_data = SentimentDataset(config=self.config, mode=None, tokenizer=self.tokenizer)
        eval_data.data = fold_eval
        return eval_data

    def load_dataset(self, mode: str = None):
        dataset = []
        if mode is None:
            return dataset
        for ff in os.listdir(self.config.dataset_path):
            if not os.path.isdir(f"{self.config.dataset_path}/{ff}"):
                continue
            for f in os.listdir(f"{self.config.dataset_path}/{ff}"):
                if f.endswith(".json") and f.startswith(mode):
                    dataset += json.load(open(f"{self.config.dataset_path}/{ff}/{f}"))["data"]
        random.shuffle(dataset)
        return dataset

    def new__getitem__(self, idx):
        data_item = self.data[idx]
        labels = torch.zeros(self.config.model_num_labels)
        labels[data_item[-1]] = 1
        text = data_item[0].lower().replace("_", " ")
        segmented_text = self.generate_prompt(text, self.generate_label(data_item[-1]))
        return {
            "input_ids": segmented_text,
            "labels": labels
        }

    def __getitem__(self, idx):
        data_item = self.data[idx]
        labels = torch.zeros(self.config.model_num_labels)
        labels[data_item[-1]] = 1
        tokenized_text = self.tokenizer(data_item[0], padding="max_length", truncation=True,
                                        max_length=self.config.model_input_max_length, return_tensors="pt")
        return {
            **tokenized_text,
            "labels": labels
        }

    def __len__(self):
        return len(self.data)
