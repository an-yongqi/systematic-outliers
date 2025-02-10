# Training GPT-2 to validate systematic outliers hypotheses

This repository contains the code and instructions to reproduce the experiments in Section 5 of our study on systematic outliers in transformer-based models. The experiments utilize the GPT-2 model and extend its architecture with various biasing mechanisms to analyze their impact on systematic outliers. Our implementation is built on the open-source [nanoGPT](https://github.com/karpathy/nanoGPT) repository and incorporates the [massive activation](https://github.com/locuslab/massive-activations) methodology.

## Model Variants

We provide several variants of the GPT-2 model to isolate the impact of different modifications:

1. **Default Attention (a)**: This serves as the baseline, using the standard attention mechanism without any bias or scaling modifications. [model_default.py](model_default.py)

2. **Explicit Fixed Bias (b)**: Adds a fixed, learnable bias \( \mathbf{v}^{\prime} \) only to the value matrix. This variant helps isolate the effect of fixed biases on the occurrence of systematic outliers. [model_vbias.py](model_vbias.py)

3. **Explicit Context-Aware Bias (c)**: Introduces context-aware bias terms \( \mathbf{k}^{\prime} \) and \( \mathbf{v}^{\prime} \), which vary based on the input sequence, to examine the influence of context-aware bias on systematic outliers. [model_cabias.py](model_cabias.py)

4. **Attention Bias (d)**: Incorporates learnable bias terms \( \mathbf{k}^{\prime} \) and \( \mathbf{v}^{\prime} \) into the key and value matrices. This variant includes both context-aware bias and a scaling factor to observe their effects on systematic outliers. [model_attn_bias.py](model_attn_bias.py)

5. **Explicit Context-Aware Scaling Factor (e)**: Utilizes a learnable scaling factor \( S_c(x) \) that dynamically adjusts the attention weights, which helps investigate how scaling impacts the reduction of systematic outliers. [model_ca_softmax.py](model_ca_softmax.py)

6. **Sigmoid Attention (f)**: Implements a sigmoid-based self-attention mechanism as proposed by Ramapuram et al. (2024) to further explore systematic attention behavior. [model_sigmoid.py](model_sigmoid.py)

## Setup

- **Data**: Follow the instructions [here](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#reproducing-gpt-2) to set up the training and validation data from OpenWebText2 to `systematic-outliers/data/openwebtext`.

  **Note**: Modify the configuration files in the [config](config) directory:
  - Set `out_dir` to the directory containing the downloaded pretrained models.
  - Set `data_dir` to the directories of the prepared OpenWebText2 dataset.

## Evaluation

To evaluate the trained GPT-2 checkpoints, run the following command. This will assess the model's performance, including systematic outlier behavior in attention:

```sh
cd gpt-2

CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_default.py ### GPT-2 default architecture
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_vbias.py ### GPT-2 with fixed value bias
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_cabias.py ### GPT-2 with context-aware bias
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_attn_bias.py ### GPT-2 with attention bias in key and value matrices
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_ca_softmax.py ### GPT-2 with context-aware scaling factor
CUDA_VISIBLE_DEVICES=0 python test.py config/eval_gpt2_sigmoid.py ### GPT-2 with sigmoid-based self-attention
```

## Training

To train the different variants of GPT-2 from scratch, execute the command below. Adjust `CUDA_VISIBLE_DEVICES` to utilize multiple GPUs if needed:

```sh
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_default.py ### GPT-2 default architecture
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_vbias.py ### GPT-2 with fixed value bias
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_cabias.py ### GPT-2 with context-aware bias
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_attn_bias.py ### GPT-2 with attention bias in key and value matrices
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_ca_softmax.py ### GPT-2 with context-aware scaling factor
CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2_sigmoid.py ### GPT-2 with sigmoid-based self-attention
```

Repeat the process for other configuration files to train corresponding variants.

## Analysis

For visualizing the activation magnitudes of intermediate features and obtaining the largest activation magnitudes layer-wise, use the following command:

```sh
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_default.py ### GPT-2 default architecture
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_vbias.py ### GPT-2 with fixed value bias
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_cabias.py ### GPT-2 with context-aware bias
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_attn_bias.py ### GPT-2 with attention bias in key and value matrices
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_ca_softmax.py ### GPT-2 with context-aware scaling factor
CUDA_VISIBLE_DEVICES=0 python analyze.py config/eval_gpt2_sigmoid.py ### GPT-2 with sigmoid-based self-attention
```

---

We hope this documentation helps you in reproducing the experiments and understanding the impact of biases and modifications on systematic outliers in transformer models. For more details, please refer to the paper and relevant code comments.
