#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib
import monkey_patch as mp
import seaborn as sns


args = SimpleNamespace(
    model="llama2_7b_chat",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/mha_relation"
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
# Load model and tokenizer
model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
print("use device ", device)

# %%
def plot_3d_feat_sub(ax, obj, layer_id, model_name, config, act_keys, savedir=None):
    num_tokens = len(obj[f"seq"])
    num_channels = obj[f"{layer_id}"].shape[2]
    inp_seq = obj[f"seq"]
    inp_seq = [x if x != "<0x0A>" else r"\n" for x in inp_seq]
    xdata = np.array([np.linspace(0,num_tokens-1,num_tokens) for i in range(num_channels)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_channels)])
    zdata = obj[f"{layer_id}"][0].numpy().T
        
    heavy_indices = config["heavy_indices"]
    
    ax.plot_wireframe(xdata[:, heavy_indices], ydata[:, heavy_indices], zdata[:, heavy_indices], rstride=0, color="magenta", linewidth=3)

    # 创建掩码，排除 heavy_indices 部分
    mask = np.ones(num_tokens, dtype=bool)
    mask[heavy_indices] = False
    
    # 绘制非 heavy 部分
    ax.plot_wireframe(xdata[:, mask], ydata[:, mask], zdata[:, mask], rstride=0, color="royalblue", linewidth=2)

    ax.set_xticks(np.linspace(0,num_tokens-1,num_tokens), inp_seq, 
                      rotation=50, fontsize=16)
    
    ax.set_zticks(*config["zticks"], fontsize=15)
    if act_keys == "mha_rms_out":
        ax.set_zticks([-10, 0 ,10], [-10, 0 ,10], fontsize=15)
    ax.set_yticks(*config["yticks"], fontsize=15, fontweight="heavy")
        
    for idx in config["heavy_indices"]:
        ax.get_xticklabels()[idx].set_weight("heavy")

    ax.set_title(f"{act_keys}", fontsize=36, fontweight="bold", y=1.015)

    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="left",
        rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-4)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-1)
    
    ax.set_zlim(*config["zlim"])
    if act_keys in ["mha_rms_out"]:
        ax.set_zlim(-12, 12)
    
    if act_keys in ["query", "key"]:
        for y_index in [58, 122]:
            if y_index < num_channels:
                ax.scatter(xdata[y_index], ydata[y_index], zdata[y_index], color='gold', s=100, marker='*', edgecolor='black')
    
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, f"{model_name}_layer_{layer_id+1}_{act_keys}.png"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()

def plot_3d_feat(obj, layer_id, model_name, config, act_keys, savedir=None):
    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_feat_sub(ax, obj, layer_id, model_name, config, act_keys, savedir)


def plot_attn_sub(ax, obj, layer_id, model_name, config, act_keys, savedir=None):
    corr = obj[f"{layer_id}"].numpy()[0].astype("float64")
    num_tokens = len(obj[f"seq"])
    inp_seq = obj["seq"]
    inp_seq = [x if x != "<0x0A>" else r"\n" for x in inp_seq]

    # 绘制热力图，使用自定义的颜色和格式
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, square=True, ax=ax,
                cmap="YlGnBu", cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect": 50},
                xticklabels=inp_seq, yticklabels=inp_seq)
    
    # 设置 x 轴标签竖直显示，y 轴标签水平显示
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", va="center")
    
    ax.set_facecolor("whitesmoke") 
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=18)
    ax.tick_params(left=False, bottom=False)

    # 高亮显示 heavy_indices 对应的标签
    heavy_indices = config["heavy_indices"]
    for idx in heavy_indices:
        ax.get_xticklabels()[idx].set_weight("heavy")
    
    ax.set_title(f"{act_keys}", fontsize=36, fontweight="bold", y=1.015)
    
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, f"{model_name}_layer_{layer_id+1}_{act_keys}.png"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()

def plot_attn(obj, layer_id, model_name, config, act_keys, savedir=None):
    fig = plt.figure(figsize=(8, 4.75))
    fig.tight_layout() 
    plt.subplots_adjust(wspace=0.15)
    
    ax = fig.add_subplot(1, 1, 1)
    plot_attn_sub(ax, obj, layer_id, model_name, config, act_keys, savedir)


#%%
layer_keys = ['mha_rms_in', 'mha_rms_out']
mha_keys = ['query', 'key', 'value']   # , "attn_out", "mha_out", "layer_out"
attn_keys = ["attn_logits", "attn_probs"]
layer_id = 27
head_id = 1

config_layer = {"zlim": (-2400, 2400),
          "yticks": ([1415, 2533], [1415, 2533]),
          "heavy_indices": [0, 3],
          "zticks": ([-2000, 0, 2000], ["-2k", "0", "2k"]),
        }

config_mha = {"zlim": (-12, 12),
          "yticks": ([58, 122], [58, 122]),
          "heavy_indices": [0, 3],
          "zticks": ([-10, 0, 10], ["-10", "0", "10"]),
        }

mp.enable_llama_relation(layers[layer_id], layer_id)

stats = {}
seq = "Summer is warm. Winter is cold.\n"
valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

with torch.no_grad():
    model(valenc)

seq_decoded = []
for i in range(valenc.shape[1]):
    seq_decoded.append(tokenizer.decode(valenc[0, i].item()))   

stats[f"seq"] = seq_decoded

for key in layer_keys:
    feat = getattr(layers[layer_id], key)
    stats[f"{layer_id}"] = feat
    plot_3d_feat(stats, layer_id, args.model, config=config_layer, act_keys=key, savedir=args.savedir)

for key in mha_keys:
    feat = getattr(layers[layer_id].self_attn, key)
    stats[f"{layer_id}"] = feat[:, head_id]
    plot_3d_feat(stats, layer_id, args.model, config=config_mha, act_keys=key, savedir=args.savedir)
    
for key in attn_keys:
    feat = getattr(layers[layer_id].self_attn, key)
    stats[f"{layer_id}"] = feat[:, head_id]
    plot_attn(stats, layer_id, args.model, config=config_layer, act_keys=key, savedir=args.savedir)
# %%