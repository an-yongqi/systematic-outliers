#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib

from lib.model_dict import MODEL_TITLE_DICT, WEIGHT_DICT

args = SimpleNamespace(
    model="mistral_7b_instruct",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/appendix_ft_models"
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
def plot_3d_weight_sub(ax, weight, selected_indices, layer_id, model_name, config, savedir=None):
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
    zdata = weight[:, [idx if idx != -1 else 0 for idx in selected_indices]].abs().cpu().numpy()

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
    ax.set_yticks(*config["yticks"], fontsize=15, fontweight="heavy")

    # 设置 z 轴的 ticks
    if weight.max() < 0.5:
        zticks = [0, 0.5]
    else:
        zticks = [0, int(weight.max().item()) + 1]  # zticks 从0到最大值的向上取整
    ax.set_zticks(zticks, zticks, fontsize=15)
    
    # 设置 tick 参数
    for i, label in enumerate(ax.get_xticklabels()):
        if i == top_1_index:
            plt.setp(label, rotation=50, ha="right", va="center", rotation_mode="anchor", weight="heavy")
        else:
            plt.setp(label, rotation=-17, ha="center", va="top", rotation_mode="anchor")
    
    ax.set_title(f"{MODEL_TITLE_DICT[model_name]}, Layer {layer_id+1}", fontsize=18, fontweight="bold", y=1.015)

    plt.setp(ax.get_yticklabels(), ha="left",
         rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-4)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-1)
    
    # 设置 z 轴范围
    ax.set_zlim(zticks[0], zticks[1] * 1.2)
    
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, f"wo_{model_name}.pdf"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()


def plot_3d_weight(weight, layer_id, model_name, config, savedir=None):
    selected_indices = config["selected_indices"]

    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_weight_sub(ax, weight, selected_indices, layer_id, model_name, config, savedir)

# %%
config = WEIGHT_DICT.get(args.model, WEIGHT_DICT["llama2_7b"])
layer_id = config["layer_id"]
if args.model in ["llama2_7b", "llama2_7b_chat", "llama2_13b", "llama2_13b_chat", "llama3_8b", "mistral_7b", "mistral_7b_instruct"]:
    weight = model.model.layers[layer_id].mlp.down_proj.weight.data
elif args.model in ["phi2"]:
    weight = model.model.layers[layer_id].mlp.fc2.weight.data
elif args.model in ["mpt_7b"]:
    weight = model.transformer.blocks[layer_id].ffn.down_proj.weight.data
elif args.model in ["opt_7b"]:
    weight = model.model.decoder.layers[layer_id].fc2.weight.data
elif args.model in ["falcon_7b"]:
    weight = model.transformer.h[layer_id].mlp.dense_4h_to_h.weight.data
plot_3d_weight(weight, layer_id, args.model, config, savedir=args.savedir)

#%%
# for layer_id in range(len(layers)):
#     weight = model.model.layers[layer_id].mlp.down_proj.weight.data
#     plot_3d_weight(weight, layer_id, args.model, config)
# # %%