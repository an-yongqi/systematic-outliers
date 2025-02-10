#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib
from tqdm import tqdm

from lib.model_dict import MODEL_TITLE_DICT

args = SimpleNamespace(
    model="llama2_7b",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/weight_outliers_pos"
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
# 回答"哪些层"的问题
def plot_layer_ax_sub(ax, data, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey"]

    x_axis = np.arange(data.shape[-1]) + 1
    ax.plot(x_axis, data[0], label="Max Ratio", color=colors[0], 
                     linestyle="-", marker="o", markerfacecolor='none', markersize=5)

    ax.plot(x_axis, data[1], label="Median Ratio", color=colors[1], 
                     linestyle="-", marker="v", markerfacecolor='none', markersize=5)

    ax.scatter(2, data[0][1], color="red", marker="*", s=200, edgecolor='black', label="Layer 2 Highlight")

    ax.set_title(f"{MODEL_TITLE_DICT[model_name]} Layer-wise Analysis", fontsize=18, fontweight="bold")

    num_layers = data.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel("Max/Mean Ratio", fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')

def plot_layer_ax(obj, model_name):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    plot_layer_ax_sub(axs, obj, model_name)
    leg = axs.legend(
        loc='center', bbox_to_anchor=(0.5, -0.25),
        ncol=4, fancybox=True, prop={'size': 14}
    )
    leg.get_frame().set_edgecolor('silver')
    leg.get_frame().set_linewidth(1.0)
    
    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)
        plt.savefig(os.path.join(args.savedir, f"{model_name}_layers.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()
    
def analyze_weight_columns(weight_matrix):
    # Get the max value for each column
    max_vals = torch.max(weight_matrix, axis=0)[0]

    # Get the mean value for each column
    mean_vals = torch.mean(weight_matrix, axis=0)

    # Calculate the max/mean ratio for each column
    max_mean_ratio = max_vals / (mean_vals + 1e-8)  # Avoid division by zero

    # Stack the results and get the index of the max ratio column
    max_ratio_col_index = max_mean_ratio.sort(descending=True)[1][:3]

    return max_mean_ratio, max_ratio_col_index

# %%
layer_stats = np.zeros((5, len(layers)))

for layer_id in tqdm(range(len(layers))):
    weight_abs = model.model.layers[layer_id].mlp.down_proj.weight.data.detach().cpu().abs()
    max_mean_ratio, max_ratio_col_index = analyze_weight_columns(weight_abs)
    
    layer_stats[0, layer_id] = max_mean_ratio.max().item()
    layer_stats[1, layer_id] = torch.median(max_mean_ratio).item()
    layer_stats[2:5, layer_id] = max_ratio_col_index.numpy()[:3]

#%%
plot_layer_ax(layer_stats, args.model)

# %%
# 回答"哪些模块"的问题
def get_max_mean_ratio(module):
    """
    获取指定模块的行和列的max/mean比率的最大值
    """
    weight_abs = module.weight.data.detach().cpu().abs()
    
    # 计算各行和各列的max/mean的比率
    max_mean_row = torch.max(weight_abs, dim=1)[0] / (torch.mean(weight_abs, dim=1) + 1e-8)
    max_mean_col = torch.max(weight_abs, dim=0)[0] / (torch.mean(weight_abs, dim=0) + 1e-8)
    
    # 获取行和列比率中的最大值
    max_mean_ratio = max(torch.max(max_mean_row).item(), torch.max(max_mean_col).item())
    
    return max_mean_ratio

# 画图
def plot_modules_on_single_ax(ax, module_stats, ref_median, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal", "orange", "purple", "red", "dimgrey"]
    markers = ["o", "v", "s", "P", "D", "*", "X"]
    modules = list(module_stats.keys())
    
    x_axis = np.arange(len(layers)) + 1
    for i, module_name in enumerate(modules):
        ax.plot(x_axis, module_stats[module_name], 
                label=f"{module_name} Max/Mean", 
                color=colors[i], 
                linestyle="-", 
                marker=markers[i], 
                markerfacecolor='none', 
                markersize=5)

    ax.plot(x_axis, ref_median, 
            label="Reference Median", 
            color=colors[-1], 
            linestyle=":", 
            marker=markers[-1], 
            markerfacecolor='none', 
            markersize=5)
    
    ax.set_title(f"Max/Mean Ratio Across Modules in {MODEL_TITLE_DICT[model_name]}", fontsize=18, fontweight="bold")
    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel("Max/Mean Ratio", fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')
    ax.legend(loc='upper right', fontsize=12)

def plot_all_modules(module_stats, ref_median, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    
    plot_modules_on_single_ax(ax, module_stats, ref_median, model_name)

    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)
        plt.savefig(os.path.join(args.savedir, f"{model_name}_modules.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()
    
    
# %%
# 初始化字典以存储结果
module_stats = {
    "down_proj": np.zeros(len(layers)),
    "up_proj": np.zeros(len(layers)),
    "gate_proj": np.zeros(len(layers)),
    "q_proj": np.zeros(len(layers)),
    "k_proj": np.zeros(len(layers)),
    "v_proj": np.zeros(len(layers)),
    "o_proj": np.zeros(len(layers)),
}

for layer_id in tqdm(range(len(layers))):
    layer = model.model.layers[layer_id]
    
    # MLP部分
    module_stats["down_proj"][layer_id] = get_max_mean_ratio(layer.mlp.down_proj)
    module_stats["up_proj"][layer_id] = get_max_mean_ratio(layer.mlp.up_proj)
    module_stats["gate_proj"][layer_id] = get_max_mean_ratio(layer.mlp.gate_proj)
    
    # 自注意力部分
    module_stats["q_proj"][layer_id] = get_max_mean_ratio(layer.self_attn.q_proj)
    module_stats["k_proj"][layer_id] = get_max_mean_ratio(layer.self_attn.k_proj)
    module_stats["v_proj"][layer_id] = get_max_mean_ratio(layer.self_attn.v_proj)
    module_stats["o_proj"][layer_id] = get_max_mean_ratio(layer.self_attn.o_proj)

# 计算参考中值
ref_median = np.mean([module_stats[module] for module in module_stats], axis=0)

# %%
plot_all_modules(module_stats, ref_median, args.model)

# %%
def plot_top3_neuron_layered_scatter(layer_stats, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置颜色和标记样式
    colors = ["cornflowerblue", "mediumseagreen", "C4"]
    markers = ["o", "v", "s"]
    
    # 绘制散点图
    for i in range(layer_stats.shape[1]):
        for j in range(3):  # 绘制 Top-1, Top-2, Top-3
            ax.scatter(i + 1, layer_stats[2 + j, i], color=colors[j], marker=markers[j], alpha=0.7, s=100)
    
    # 设置标题和标签
    ax.set_title(f"Top-3 Neuron Index Positions Across Layers in {MODEL_TITLE_DICT[model_name]}", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Layer Index', fontsize=16, labelpad=10)
    ax.set_ylabel('Neuron Index Position', fontsize=16, labelpad=10)
    ax.set_ylim(0, 11008)
    
    # 设置X轴刻度
    num_layers = layer_stats.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label)
    ax.set_xticklabels(xtick_label, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # 添加网格线和图例
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 为图例创建自定义的句柄，以确保颜色匹配
    legend_handles = [plt.Line2D([0], [0], color=color, marker=marker, linestyle='', markersize=10, label=f'Top-{i+1}') 
                      for i, (color, marker) in enumerate(zip(colors, markers))]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=14, fancybox=True)

    # 保存图像
    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)
        plt.savefig(os.path.join(args.savedir, f"{model_name}_neurons.pdf"), bbox_inches="tight", dpi=300)
    
    plt.tight_layout()
    plt.show()
    plt.close()

# 使用layer_stats绘制分层散点图
plot_top3_neuron_layered_scatter(layer_stats, args.model)

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
        plt.savefig(os.path.join(savedir, f"{model_name}_layer_{layer_id+1}_weight_down.pdf"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()


def plot_3d_weight(weight, layer_id, model_name, config, savedir=None):
    selected_indices = config["selected_indices"]

    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_weight_sub(ax, weight, selected_indices, layer_id, model_name, config, savedir)

config1 = {
    "top_1": 7006,
    "yticks": ([1415, 2533], [1415, 2533]),
    "layer_id": 30,
    "selected_indices": [0, 1, 2, -1, -1, 7005, 7006, 7007, -1, -1, 11007],
}
layer_id = config1["layer_id"]
weight = model.model.layers[layer_id].mlp.down_proj.weight.data

plot_3d_weight(weight, layer_id, args.model, config1, savedir=args.savedir)

config2 = {
    "top_1": 3228,
    "yticks": ([1512], [1512]),
    "layer_id": 31,
    "selected_indices": [0, 1, 2, -1, 3227, 3228, 3229, -1, -1, -1, 11007],
}
layer_id = config2["layer_id"]
weight = model.model.layers[layer_id].mlp.down_proj.weight.data

plot_3d_weight(weight, layer_id, args.model, config2, savedir=args.savedir)

# %%
