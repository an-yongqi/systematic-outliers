import os
from base_path import BASE_PATH

# Model type
model_type = "gpt2_vbias"

# Model configuration
n_layer = 12
n_head = 12
n_embd = 768
batch_size = 8
eval_iters = 500  # Use more iterations to get a good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
ckpt_iter = 50000

data_dir = os.path.join(BASE_PATH, "data/openwebtext")
save_dir = os.path.join(BASE_PATH, "figures/exp/")

# Extract model variant from model_type
model_variant = model_type.replace("gpt2_", "")

# Adjust output directory dynamically
out_dir = os.path.join(BASE_PATH, "gpt-2/results", model_variant)

# Compilation flag
compile = False

# Print values for verification
print(f"Base Path: {BASE_PATH}")
print(f"Data Dir: {data_dir}")
print(f"Save Dir: {save_dir}")
print(f"Model Type: {model_type}")
print(f"Model Variant: {model_variant}")
print(f"Output Dir: {out_dir}")