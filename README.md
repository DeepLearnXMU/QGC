# Query-Guided Compressor (QGC)
Code for "Retaining Key Information under High Compression Rates: Query-Guided Compressor for LLMs" (ACL 2024)

## Requirements

```
datasets==2.15.0
flash-attn==2.3.3
jsonlines==4.0.0
torch==2.0.0
torchvision==0.15.0
transformers==4.35.0
```

## Instructions

We use an example to show how to use our codes.

### LLMs and Datasets

We use [LongChat-13B](https://huggingface.co/lmsys/longchat-13b-16k) as the target LLM, and use Llama-2-7B to initial the compressor parameters. For datasets, we use open-source QA datasets (NaturalQuestions, TrivialQA, HotpotQA) to train our compressor and evaluate it. All datasets can be downloaded from [this site](https://drive.google.com/drive/folders/1HhwPP6iZUBbAjWeWRkbEPtgXVIRZUz6V?usp=drive_link).

### QGC Training and Inference

```
# train compressor
bash train.sh

# evaluate compressor
bash infer.sh
```
