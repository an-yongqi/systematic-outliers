#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib

from lib.model_dict import MODEL_TITLE_DICT, MODEL_LAYER_ENABLE_DICT, ACTIVATION_OUTPUT_DICT

#%%
args = SimpleNamespace(
    model="llama2_7b",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/existence"
)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'figure.titlesize': 28
})
matplotlib.rcParams['text.usetex'] = False

#%%
def plot_3d_feat_sub(ax, obj, layer_id, model_name, config, savedir=None):
    num_tokens = len(obj[f"seq"])
    num_channels = obj[f"{layer_id}"].shape[2]
    inp_seq = obj[f"seq"]
    inp_seq = [x if x != "<0x0A>" and x != '\n' else r"\n" for x in inp_seq]
    xdata = np.array([np.linspace(0,num_tokens-1,num_tokens) for i in range(num_channels)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_channels)])
    zdata = obj[f"{layer_id}"][0].abs().numpy().T
    
    heavy_indices = config["heavy_indices"]
    ax.plot_wireframe(xdata[:, heavy_indices], ydata[:, heavy_indices], zdata[:, heavy_indices], rstride=0, color="magenta", linewidth=3)

    # 创建掩码，排除 heavy_indices 部分
    mask = np.ones(num_tokens, dtype=bool)
    mask[heavy_indices] = False
    
    # 绘制非 heavy 部分
    ax.plot_wireframe(xdata[:, mask], ydata[:, mask], zdata[:, mask], rstride=0, color="royalblue", linewidth=2.5)

    ax.set_xticks(np.linspace(0,num_tokens-1,num_tokens), inp_seq, 
                      rotation=50, fontsize=16)

    ax.set_zticks(*config["zticks"], fontsize=15)
    ax.set_yticks(*config["yticks"], fontsize=15, fontweight="heavy")
    for idx in config["heavy_indices"]:
        ax.get_xticklabels()[idx].set_weight("heavy")

    ax.set_title(f"{MODEL_TITLE_DICT[model_name]}, Layer {layer_id+1}", fontsize=18, fontweight="bold", y=1.015)

    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="left",
         rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-4)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-1)
    ax.set_zlim(*config["zlim"])
    
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, f"ao_out_{model_name}.pdf"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()

def plot_3d_feat(obj, layer_id, model_name, config, savedir=None):
    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    
    # for i in range(3):
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_feat_sub(ax, obj, layer_id, model_name, config, savedir)
    
def get_layer_enable_function(model_name):
    func = MODEL_LAYER_ENABLE_DICT.get(model_name)
    if not func:
        raise ValueError(f"No layer enable function found for model: {model_name}")
    return func
    
#%%      
# Load model and tokenizer
model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
print("use device ", device)

#%%
config = ACTIVATION_OUTPUT_DICT.get(args.model, ACTIVATION_OUTPUT_DICT["llama2_7b"])
# 获取适当的启用解码层函数
enable_layer_func = get_layer_enable_function(args.model)

# %%
layer_id = config["layer_id"]
enable_layer_func(layers[layer_id], layer_id)

stats = {}
seq = "Summer is warm. Winter is cold.\n"
valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

with torch.no_grad():
    model(valenc)

seq_decoded = []
for i in range(valenc.shape[1]):
    seq_decoded.append(tokenizer.decode(valenc[0,i].item()))

stats[f"seq"] = seq_decoded
feat_abs = layers[layer_id].feat.abs()
stats[f"{layer_id}"] = feat_abs
plot_3d_feat(stats, layer_id, args.model, config, savedir=args.savedir)

# %%
