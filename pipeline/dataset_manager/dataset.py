import os
import json
import torch
from collections import Counter
import random
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, config, mode, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        dataset = []
        for ff in os.listdir(self.config.dataset_path):
            for f in os.listdir(f"{self.config.dataset_path}/{ff}"):
                if f.endswith(".json") and f.startswith(mode):
                    dataset += self.load_dataset(path=f"{self.config.dataset_path}/{ff}/{f}")
        random.shuffle(dataset)
        self.data = dataset
        if mode == "eval":
            self.data = self.data[:5000]
        print(Counter([e[1] for e in self.data]))
        print(f"Load {len(self.data)} examples for {mode} dataset")
        self.mapping_label = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }

    @staticmethod
    def generate_prompt(sentence):
        return f"""Classify the sentiment of the following Vietnamese sentence into one of three classes: neutral, negative, positive.\n### Input: {sentence}"""

    def generate_label(self, label_id):
        return self.mapping_label[label_id]

    @staticmethod
    def load_dataset(path: str):
        data = json.load(open(path, "r", encoding="utf8"))
        return data["data"]

    def __getitem__(self, idx):
        data_item = self.data[idx]
        labels = torch.zeros(self.config.model_num_labels)
        labels[data_item[-1]] = 1
        text = data_item[0].lower().replace("_", " ")
        segmented_text = self.generate_prompt(text)
        return {
            "input_ids": segmented_text,
            "labels": labels
        }

    def __len__(self):
        return len(self.data)
