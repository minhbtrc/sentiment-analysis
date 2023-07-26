LORA_TARGET_MODULES = {
    "roberta": ["query", "value"],
    "bloom": ["query", "value"]
}
LORA_TASK_TYPE = {
    "roberta": "SEQ_CLS",
    "bloom": "SEQ_CLS"
}
