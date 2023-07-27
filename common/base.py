import logging

from common.config import SentimentConfig, SingletonClass


class BaseModel(SingletonClass):
    def __init__(self, config: SentimentConfig):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config if config is not None else SentimentConfig()

