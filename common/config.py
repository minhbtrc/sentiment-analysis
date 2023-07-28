import logging
from typing import List, AnyStr
from logging.handlers import RotatingFileHandler

from common.constants import *
from common.common_keys import *


class ONNXOptions:
    ONNX = "ONNX"
    optimized_ONNX = "optimized_ONNX"
    quantized_optimized_ONNX = "quantized_optimized_ONNX"

    @classmethod
    def keys(cls):
        return [val for attr, val in cls.__dict__.items() if not attr.startswith("__") and isinstance(val, str)]


class SingletonClass(object):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def setup_logging(logging_folder, log_name, log_exception_classes: List[AnyStr] = None, logging_level=logging.INFO):
    if log_exception_classes is None:
        log_exception_classes = []
    os.makedirs(logging_folder, exist_ok=True)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s | %(name)s | [%(levelname)s] | %(lineno)d | %(message)s",
        handlers=[
            RotatingFileHandler(os.path.join(logging_folder, f"{log_name}.log"), encoding="utf8",
                                maxBytes=1024 * 10240, backupCount=20),
            logging.StreamHandler()
        ]
    )
    for exception_class in log_exception_classes:
        logging.getLogger(exception_class).setLevel(logging.WARNING)


class SentimentConfig(SingletonClass):
    def __init__(
            self,
            pretrained_path: str = None,
            onnx_path: str = None,
            dataset_path: str = None,
            training_output_dir: str = None,
            training_learning_rate: float = None,
            training_weight_decay: float = None,
            training_batch_size: int = None,
            training_num_epochs: int = None,
            training_save_total_limit: int = None,
            training_gradient_accumulation_steps: int = None,
            training_eval_steps: int = None,
            training_logging_steps: int = None,
            training_save_steps: int = None,
            training_logging_dir: str = None,
            training_warm_up_ratio: float = None,
            training_device: str = None,
            training_metrics: str = None,
            segmentor_dir: str = None,
            model_input_max_length: int = None,
            model_num_labels: int = None,

            use_lora: bool = None,
            lora_dropout: float = None,
            lora_alpha: float = None,
            lora_rank: int = None,

            logging_folder: str = None,
            log_file: str = None
    ):
        self.pretrained_path = pretrained_path if pretrained_path is not None else os.getenv(PRETRAINED_PATH,
                                                                                             "vinai/phobert-base-v2")
        self.dataset_path = dataset_path if dataset_path is not None else os.getenv(TRAINING_DATASET_PATH,
                                                                                    "dataset/processed")
        self.training_output_dir = training_output_dir if training_output_dir is not None else os.getenv(OUTPUT_DIR,
                                                                                                         "pipeline/output") + "/checkpoint"
        self.onnx_path = onnx_path if onnx_path is not None else f"{self.training_output_dir}/model.onnx"
        self.training_logging_dir = training_logging_dir if training_logging_dir is not None else os.getenv(OUTPUT_DIR,
                                                                                                            "pipeline/output") + "/logging"
        self.segmentor_dir = segmentor_dir if segmentor_dir is not None else "./"
        self.training_learning_rate = training_learning_rate if training_learning_rate is not None else 2e-5
        self.training_weight_decay = training_weight_decay if training_weight_decay is not None else 0.01
        self.training_batch_size = training_batch_size if training_batch_size is not None else 8
        self.training_num_epochs = training_num_epochs if training_num_epochs is not None else 10
        self.training_save_total_limit = training_save_total_limit if training_save_total_limit is not None else 3
        self.training_gradient_accumulation_steps = training_gradient_accumulation_steps if training_gradient_accumulation_steps is not None else 2
        self.training_eval_steps = training_eval_steps if training_eval_steps is not None else 500
        self.training_logging_steps = training_logging_steps if training_logging_steps is not None else 500
        self.training_save_steps = training_save_steps if training_save_steps is not None else 500
        self.training_warm_up_ratio = training_warm_up_ratio if training_warm_up_ratio is not None else 0.1
        self.training_device = training_device if training_device is not None else "cuda"
        self.training_metrics = training_metrics if training_metrics is not None else "f1"
        self.model_input_max_length = model_input_max_length if model_input_max_length is not None else 256
        self.model_num_labels = model_num_labels if model_num_labels is not None else 3

        self.use_lora = use_lora if use_lora is not None else bool(os.getenv(USE_LORA, ""))
        self.lora_dropout = lora_dropout if lora_dropout is not None else float(os.getenv(LORA_DROPOUT, 0.05))
        self.lora_alpha = lora_alpha if lora_alpha is not None else int(os.getenv(LORA_ALPHA, 16))
        self.lora_rank = lora_rank if lora_rank is not None else int(os.getenv(LORA_RANK, 4))

        self.logging_folder = logging_folder if logging_folder is not None else os.getenv(LOG_FOLDER, "logs")
        self.log_file = log_file if log_file is not None else os.getenv(LOG_FILE, "app.log")
        setup_logging(logging_folder=self.logging_folder, log_name=self.log_file)