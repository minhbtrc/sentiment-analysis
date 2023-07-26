from common.base import SingletonClass
from pipeline.dataset_manager.constants import *
from common.common_keys import *


class SentimentId2Tag:
    tag_mapping = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }
    id_mapping = {
        v: k
        for k, v in tag_mapping.items()
    }

    @classmethod
    def get_tag(cls, idx: int):
        return cls.tag_mapping.get(idx)

    @classmethod
    def get_id(cls, tag: str):
        return cls.id_mapping.get(tag)


class DatasetConfig(SingletonClass):
    def __init__(
            self,
            dataset_path: str = None,
            checkpoint_path: str = None,
            openai_url: str = None,
            deployed_model_url: str = None,
            openai_key: str = None,
    ):
        self.dataset_path = dataset_path if dataset_path is not None else DATASET_PATH
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else f"{CURRENT_PATH}/pipeline/checkpoint"
        self.openai_url = openai_url if openai_url is not None else os.getenv(OPENAI_URL)
        self.deployed_model_url = deployed_model_url if deployed_model_url is not None else os.getenv(
            DEPLOYED_MODEL_URL)
        self.openai_key = openai_key if openai_key is not None else os.getenv(OPENAI_KEY)
