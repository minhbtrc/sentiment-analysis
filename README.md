## Sentiment Analysis using PhoBert+Linear

### Model Architecture
- Try some pretrained model:
  - RobertaForSequenceClassification: `vinai/phobert-base-v2`
  - BloomForSequenceClassification: `bigscience/bloom-560m`
- Architecture: XXXXForSequenceClassification
- Use Trainer of Huggingface to training model

### Dataset
- Dataset: [30K e-commerce reviews](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst)
- Labels:
    - NEG: Negative
    - POS: Positive
    - NEU: Neutral
 
### Optimization
- Use some optimization techniques to optimize ONNX - [Optim](https://github.com/huggingface/notebooks/blob/main/examples/onnx-export.ipynb)
- See: pipeline/onnx_converter.py

### How to run
`PRETRAINED_PATH=bigscience/bloom-560m CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 --master-port=30000 pipeline/trainer.py`