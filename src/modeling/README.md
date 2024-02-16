# Modeling Folder

This directory contains scripts and resources for customizing, evaluating, fine-tuning, and inferencing with the GPT Big model, focusing on positional interpolation in the transformer architecture.

## Directory Structure

```
modeling/
│
├── eval/
│   ├── ... (Evaluation scripts and resources)
│
├── fine-tune/
│   ├── ... (Scripts and datasets for fine-tuning)
│
├── inference/
│   ├── ... (Scripts for running inferences with the trained model)
│
└── modeling_gpt_bigcode.py.py
    └── ... (Script to modify the GPT Big model architecture for positional interpolation)
```

## Subfolders

### `eval/`

Contains scripts and resources to evaluate the model. Refer to the README within for specific instructions.

### `fine-tune/`

This directory contains scripts and datasets for fine-tuning the model. Make sure to check out the README inside for detailed steps.

### `inference/`

Scripts for running inferences using the trained model. For usage and other details, see the README in the subfolder.

## Script Details

### `modeling_gpt_bigcode.py`

This script is used to modify the GPT Big architecture, focusing on positional interpolation within the transformer architecture. You need to replace this file in the transformer package to use `expand_positional_embeddings` function.

### `submit_finetune.sh`

Submit jobs to AiMOS Sever

