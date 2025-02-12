# Installation

## Dependencies
Create an new conda virtual environment
```sh
conda create -n systematic-outliers python=3.9.18 -y
conda activate systematic-outliers
```

Install [Pytorch](https://pytorch.org/)>=2.0.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.15.0 following official instructions. For example:
```sh
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install additional dependencies:
```sh
pip install timm==0.6.7 transformers==4.45.2 accelerate==0.23.0 datasets==2.14.5 matplotlib==3.8.0 wandb==0.16.4 seaborn sentencepiece protobuf
```

## Pretrained Models

- **LLM Models**: To use pretrained LLM models, update the `CACHE_DIR_BASE` variable in the [model_dict.py](lib/model_dict.py) file to point to the directory containing the pretrained model weights.
- **GPT-2**: To replicate the gpt-2 experiments, update the `BASE_PATH` variable in the [base_path.py](gpt-2/config/base_path.py) file to point to the user directory where this code are stored.

## Reference File Structure
The project structure follows the organization below. You can refer to the following structure for model weights and data preparation:
```
systematic-outliers/
│── data/
│   ├── openwebtext/
│── gpt-2/
│   ├── results/
│   ├── ...
│── model_weights/
│   ├── Llama-2-7b-hf/
│   ├── ...
│── ...
```