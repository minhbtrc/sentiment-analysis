import logging

from common.config import SentimentConfig


class SingletonClass(object):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseModel(SingletonClass):
    def __init__(self, config: SentimentConfig):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config if config is not None else SentimentConfig()

