import os
import json
import logging
import requests

from typing import Dict, Optional
from tqdm import tqdm

from pipeline.dataset_manager.config import DatasetConfig, SentimentId2Tag
from pipeline.dataset_manager.pre_process import Cleaner


class Labeler:
    def __init__(self, mode: str, config: DatasetConfig):
        self.config = config if config is not None else DatasetConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.mode = mode

    @staticmethod
    def _generate_prompt(sentence: str):
        return "Classify the following Vietnamese sentence into one of three labels positive, " \
               "negative and neutral for sentiment analysis problem, please return only label. " \
               "Example: sentence: \"mẫu này mặc nóng ko shop?\" is neutral.\n " \
               "sentence: \"Dạ không, em hỏi sản phẩm mã XXXX ạ\" is neutral.\n " \
               "sentence: \"shop rep mình lâu thế, mình không mua nữa, làm ăn kì cục\" is negative.\n " \
               "sentence: \"cảm ơn shop nhé, shop tư vấn nhiệt tình, sản phẩm đẹp nữa\" is positive. \n " \
               f"\"{sentence}\""

    def call_api(self, sentence):
        def openai():
            payload = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": self._generate_prompt(sentence)
                    }
                ]
            })
            headers = {
                '': '',
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config.openai_key}'
            }
            response = requests.request("POST", self.config.openai_url, headers=headers, data=payload).json()
            return response["choices"][0]["message"]["content"]

        def deployed_model():
            payload = json.dumps({
                "index": "minhbtc",
                "data": {
                    "message": sentence,
                    "class": "XXXXXX",
                    "storage_id": "XXXXXX",
                    "channel_id": "XXXXXX"
                }
            })
            headers = {
                'Content-Type': 'application/json'
            }
            response = requests.request("POST", self.config.deployed_model_url, headers=headers, data=payload).json()
            return response

        if self.mode == "trained_model":
            return deployed_model()
        return openai()

    def label(self, sentence: str):
        return self.call_api(sentence)


class DatasetProcessor:
    def __init__(self, config: DatasetConfig = None):
        self.config = config if config is not None else DatasetConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cleaner = Cleaner()
        self.labeler = Labeler(mode="trained_model", config=self.config)
        self.raw_dataset: Optional[Dict] = None
        self.init()

    def init(self):
        self.load_raw_dataset()

    def load_raw_dataset(self):
        dataset = []
        all_files = os.listdir(self.config.dataset_path)
        for f in all_files:
            if not f.endswith(".json"):
                continue
            dataset += json.load(open(f"{self.config.dataset_path}/{f}", "r", encoding="utf8"))["data"]
        self.raw_dataset = [[self.cleaner.text_process(message[0]), message[1]] for message in dataset]
        self.logger.info(f"Loaded {len(self.raw_dataset)} raw messages")

    def process(self):
        output = []
        for sent in tqdm(self.raw_dataset):
            label = self.labeler.label(sent[0])
            processed_text = label["text"][0]
            label = label["output"][0]["tag"] if label["output"][0]["score"] > 0.98 else "XXXXXX"
            output.append([processed_text, sent[1], SentimentId2Tag.get_id(label),
                           bool(sent[1] == SentimentId2Tag.get_id(label))])
        output_path = self.config.dataset_path.replace("raw", "processed")
        json.dump({"data": output}, open(f"{output_path}/processed_data.json", "w", encoding="utf8"), indent=4,
                  ensure_ascii=False)


if __name__ == "__main__":
    labeler = DatasetProcessor()
    labeler.process()
