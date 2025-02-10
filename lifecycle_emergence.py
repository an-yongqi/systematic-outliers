#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib
import monkey_patch as mp

from lib.model_dict import WEIGHT_DICT

args = SimpleNamespace(
    model="llama2_7b",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/mlp_relation_start"
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


#%%
def plot_3d_weight_sub(ax, weight, selected_indices, layer_id, model_name, config, weight_key, savedir=None):
    if weight_key != 'down_proj':
        weight = weight.T
    
    num_columns = len(selected_indices)  # 获取列数
    num_rows = weight.shape[0]  # 获取行数
    
    # 动态生成配置
    top_1 = config["top_1"]
    top_1_index = selected_indices.index(top_1)
    
    # 填充省略号
    xticklabels_full = ["···"] * num_columns
    xticklabels_full[0] = '0'
    xticklabels_full[1] = '1'
    xticklabels_full[2] = '2'
    xticklabels_full[top_1_index] = str(selected_indices[top_1_index])
    
    xdata = np.array([np.linspace(0, num_columns - 1, num_columns) for i in range(num_rows)])
    ydata = np.array([np.ones(num_columns) * i for i in range(num_rows)])
    # 处理 -1 情况，将其对应的列设置为 0
    zdata = weight[:, [idx if idx != -1 else 0 for idx in selected_indices]].cpu().numpy()

    # 绘制 heavy 部分
    ax.plot_wireframe(xdata[:, top_1_index][:, np.newaxis], ydata[:, top_1_index][:, np.newaxis], zdata[:, top_1_index][:, np.newaxis], rstride=0, color="magenta", linewidth=3)

    # 创建掩码，排除 heavy_indices 部分和 -1 值
    mask = np.array([True if idx != -1 else False for idx in selected_indices])
    mask[top_1_index] = False  # 保留 heavy 部分
    
    # 绘制非 heavy 部分，跳过 -1 对应的数据
    ax.plot_wireframe(xdata[:, mask], ydata[:, mask], zdata[:, mask], rstride=0, color="forestgreen", linewidth=2)

    # 设置 x 轴标签和样式
    ax.set_xticks(np.linspace(0, num_columns - 1, num_columns))
    ax.set_xticklabels(xticklabels_full, fontsize=16)
    
    # 设置 y 轴的 ticks 和 labels
    if weight_key == 'down_proj':
        ax.set_yticks(*config["yticks"], fontsize=15, fontweight="heavy")
    else:
        ax.set_yticks([], [], fontsize=15)

    # 设置 z 轴的 ticks
    zticks = [-1, 0, 1]  # zticks 从0到最大值的向上取整
    ax.set_zticks(zticks, zticks, fontsize=15)
    
    # 设置 tick 参数
    for i, label in enumerate(ax.get_xticklabels()):
        if i == top_1_index:
            plt.setp(label, rotation=50, ha="right", va="center", rotation_mode="anchor", weight="heavy")
        else:
            plt.setp(label, rotation=-17, ha="center", va="top", rotation_mode="anchor")
    
    ax.set_title(f"{weight_key}", fontsize=36, fontweight="bold", y=1.015)

    plt.setp(ax.get_yticklabels(), ha="left",
         rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-4)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-1)
    
    # 设置 z 轴范围
    ax.set_zlim(-1.5, 1.5)
    
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, f"{model_name}_layer_{layer_id+1}_{weight_key}.png"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()


def plot_3d_weight(weight, layer_id, model_name, config, weight_key, savedir=None):
    selected_indices = config["selected_indices"]

    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_weight_sub(ax, weight, selected_indices, layer_id, model_name, config, weight_key, savedir)

config = WEIGHT_DICT.get(args.model, WEIGHT_DICT["llama2_7b"])
layer_id = config["layer_id"]

weight_keys = ['down_proj', 'gate_proj', 'up_proj']

for key in weight_keys:
    weight = model.model.layers[layer_id].mlp.__getattr__(key).weight.data
    plot_3d_weight(weight, layer_id, args.model, config, key, savedir=args.savedir)

#%%
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
    ax.plot_wireframe(xdata[:, mask], ydata[:, mask], zdata[:, mask], rstride=0, color="royalblue", linewidth=2.5)

    ax.set_xticks(np.linspace(0,num_tokens-1,num_tokens), inp_seq, 
                      rotation=50, fontsize=16)
    
    if key in ["gate_out", "up_out", "act_out", "rms_out"]:
        ax.set_zticks([-50, 0, 50], [-50, 0, 50], fontsize=15)
    else:
        ax.set_zticks(*config["zticks"], fontsize=15)
    if num_channels == 11008:
        ax.set_yticks(*config["yticks"], fontsize=15, fontweight="heavy")
    elif key in ["down_out", "layer_out"]:
        ax.set_yticks([1415, 2533], [1415, 2533], fontsize=15, fontweight="heavy")
    else:
        ax.set_yticks([], [], fontsize=15)
        
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
    if key in ["gate_out", "up_out", "act_out", "rms_out"]:
        ax.set_zlim(-60, 60)
    else:
        ax.set_zlim(*config["zlim"])
    
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

#%%
mlp_keys = ['down_out', 'act_out', 'down_in', 'gate_out', 'up_out']
layer_keys = ['rms_in', 'rms_out', 'layer_out']
layer_id = 1

config = {"zlim": (-2400, 2400),
          "yticks": ([7890], [7890]),
          "heavy_indices": [0, 3],
          "zticks": ([-2000, 0, 2000], ["-2k", "0", "2k"]),
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

for key in mlp_keys:
    feat_abs = getattr(layers[layer_id].mlp, key)
    stats[f"{layer_id}"] = feat_abs

    plot_3d_feat(stats, layer_id, args.model, config=config, act_keys=key, savedir=args.savedir)
    
for key in layer_keys:
    feat_abs = getattr(layers[layer_id], key)
    stats[f"{layer_id}"] = feat_abs

    plot_3d_feat(stats, layer_id, args.model, config=config, act_keys=key, savedir=args.savedir)
# %%