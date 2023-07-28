## Finetune pretrained model for Sentiment analysis

### Model Architecture
- Try some pretrained model:
  - [PhoBERT](https://github.com/VinAIResearch/PhoBERT): `vinai/phobert-base-v2`
  - [Bloom](https://huggingface.co/docs/transformers/model_doc/bloom): `bigscience/bloom-560m`
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
0. Note!!!
- Pass `model_class` to init class SentimentProcessor.
- Pass `n_folds` != None if you want to training with K-fold validation
- If you use Bloom, you should pass `use_lora=True`
1. Export environment variables: `while read LINE; do export "$LINE"; done < .env`
2. Run training: `PRETRAINED_PATH=bigscience/bloom-560m CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 --master-port=30000 pipeline/trainer.py`