import os
from base_path import BASE_PATH

model_type = "gpt2_default"

# Extract model variant from model_type
model_variant = model_type.replace("gpt2_", "")

# Adjust output directory dynamically
out_dir = os.path.join(BASE_PATH, "gpt-2/results", model_variant)
data_dir = os.path.join(BASE_PATH, "data/openwebtext")

wandb_log = False
wandb_project = 'systematic-outliers'
wandb_run_name='gpt2-124M-attn-bias-run'
compile=False 

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# print(f"Base Path: {BASE_PATH}")
# print(f"Data Dir: {data_dir}")
# print(f"Model Type: {model_type}")
# print(f"Model Variant: {model_variant}")
# print(f"Output Dir: {out_dir}")