#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib

from tqdm import tqdm

from lib.model_dict import MODEL_TITLE_DICT, MODEL_ATTN_ENABLE_DICT, ATTN_SCORE_DICT

args = SimpleNamespace(
    model="llama2_7b",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/attention_outliers_pos"
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

def get_layer_enable_function(model_name):
    func = MODEL_ATTN_ENABLE_DICT.get(model_name)
    if not func:
        raise ValueError(f"No layer enable function found for model: {model_name}")
    return func

# %%
# 回答"哪些Head"的问题
def plot_head_layer_ax_sub(ax, data, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "dimgrey"]
    x_axis = np.arange(data.shape[0]) + 1   # [1,...,32]
    
    # 计算每一层中每个head的第0和19个key的累积注意力得分
    fixed_key_scores = data[:, :, [0, 19]]  # (num_layers, num_heads, 2)
    total_scores = np.sum(data, axis=-1)    # (num_layers, num_heads)
    fixed_key_ratio = np.sum(fixed_key_scores, axis=2) / total_scores  # 第0和19 key的累积得分与总得分的比率

    # 计算平均值、最大值、最小值
    mean_fixed_key_ratio = np.mean(fixed_key_ratio, axis=1)
    max_fixed_key_ratio = np.max(fixed_key_ratio, axis=1)
    min_fixed_key_ratio = np.min(fixed_key_ratio, axis=1)

    # 绘制带有最大值和最小值的散点图
    ax.errorbar(x_axis, mean_fixed_key_ratio, yerr=[mean_fixed_key_ratio - min_fixed_key_ratio, max_fixed_key_ratio - mean_fixed_key_ratio], 
                fmt='o', color=colors[0], ecolor=colors[1], elinewidth=2, capsize=4, 
                label='Mean with Min/Max of Attention Outlier Scores')

    # 添加标题、坐标轴标签和刻度
    ax.set_title(f"{MODEL_TITLE_DICT[model_name]} Attention Outlier Ratios per Head Across Layers", fontsize=20, fontweight="bold")
    num_layers = data.shape[0]
    xtick_label = [1, num_layers // 4, num_layers // 2, num_layers * 3 // 4, num_layers]
    ax.set_xticks(xtick_label)
    ax.set_xticklabels(xtick_label, fontsize=16)
    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel('Attention Score Ratio', fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')

def plot_head_layer_ax(data, model_name, save_dir=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)

    plot_head_layer_ax_sub(ax, data, model_name)
    
    leg = ax.legend(
        loc='center', bbox_to_anchor=(0.5, -0.25),
        ncol=3, fancybox=True, prop={'size': 14}
    )
    leg.get_frame().set_edgecolor('silver')
    leg.get_frame().set_linewidth(1.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_heads.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()

#%%
enable_layer_func = get_layer_enable_function(args.model)
for layer_id in tqdm(range(len(layers))):
    enable_layer_func(layers[layer_id], layer_id)

testseq_list = lib.get_data(tokenizer, nsamples=1, seqlen=1024, device=device)
valenc = testseq_list[0]
data = np.zeros((len(layers), 32, 1024))    # (num_layers, num_heads, num_tokens)

with torch.no_grad():
    model(valenc)

for layer_id in tqdm(range(len(layers))):
    attn_logit = layers[layer_id].self_attn.attn_probs  # (1, 32, 1024, 1024)
    data[layer_id] = attn_logit[0].sum(axis=1)  # (32, 1024)

#%%
plot_head_layer_ax(data, args.model, save_dir=args.savedir)

#%%
# 回答"哪些Layer"的问题
def plot_attn_layer_ax_sub(ax, data, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "dimgrey"]
    x_axis = np.arange(data.shape[0]) + 1
    for i in range(2):
        top_values = np.sort(data, axis=-1)[:, -i-1]  # Top-2 values
        ax.plot(x_axis, top_values, label=f"Top {i+1}", color=colors[i], 
                linestyle="-", marker="o", markerfacecolor='none', markersize=5)

    median_values = np.median(data, axis=-1)
    ax.plot(x_axis, median_values, label="Median", color=colors[-1], 
            linestyle="-", marker="v", markerfacecolor='none', markersize=5)

    ax.set_title(f"{MODEL_TITLE_DICT[model_name]} Layer-wise Attention Accumulation", fontsize=18, fontweight="bold")
    num_layers = data.shape[0]
    xtick_label = [1, num_layers // 4, num_layers // 2, num_layers * 3 // 4, num_layers]
    ax.set_xticks(xtick_label)
    ax.set_xticklabels(xtick_label, fontsize=16)
    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel('Attention Scores', fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')

def plot_attn_layer_ax(data, model_name, save_dir=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)

    plot_attn_layer_ax_sub(ax, data, model_name)
    
    leg = ax.legend(
        loc='center', bbox_to_anchor=(0.5, -0.25),
        ncol=3, fancybox=True, prop={'size': 14}
    )
    leg.get_frame().set_edgecolor('silver')
    leg.get_frame().set_linewidth(1.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_layers.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()

#%%
enable_layer_func = get_layer_enable_function(args.model)
for layer_id in tqdm(range(len(layers))):
    enable_layer_func(layers[layer_id], layer_id)

testseq_list = lib.get_data(tokenizer, nsamples=1, seqlen=1024, device=device)
valenc = testseq_list[0]
data = np.zeros((len(layers), 1024))

with torch.no_grad():
    model(valenc)

for layer_id in tqdm(range(len(layers))):
    attn_logit = layers[layer_id].self_attn.attn_probs.mean(axis=1)
    data[layer_id] = attn_logit[0].sum(axis=0)

#%%
plot_attn_layer_ax(data, args.model, save_dir=args.savedir)

# %%
layer_id = 3
enable_layer_func = get_layer_enable_function(args.model)
enable_layer_func(layers[layer_id], layer_id)

testseq_list = lib.get_data(tokenizer, nsamples=100, seqlen=seq_len, device=device)

#%%
stats = {}
stats['counts'] = {'initial_token': 0, 'strong_delimiter': 0, 'weak_semantic': 0, 'other': 0}

for seqid, testseq in tqdm(enumerate(testseq_list)):
    with torch.no_grad():
        model(testseq)
        
    seq_decoded = []
    for i in range(testseq.shape[1]):
        seq_decoded.append(tokenizer.decode(testseq[0,i].item()))
        
    stats[f"seq"] = seq_decoded

    attn_scores = layers[layer_id].self_attn.attn_probs.mean(axis=1)  # 计算平均注意力得分
    attn_sum = attn_scores[0].sum(axis=0)  # 累积注意力得分 [seqlen]
    
    # 筛选出累积注意力得分 >= 100 的 keys
    outlier_mask = attn_sum >= seq_len * 0.1
    outlier_positions = outlier_mask.nonzero(as_tuple=False).squeeze()  # Keys的位置
    
    tokens = testseq[0]  # 当前序列中的 tokens

    strong_delimiters = {".", "<0x0A>", ",", ";", ":", "!", "?", "-", "--", "---", "'", '"', "(", ")", "[", "]", "{", "}", "/", "\\", "|", "*", "&", "^", "%", "$", "#", "@", "~", "`"}
    weak_semantic_tokens = {"", "and", "from", "of", "1", "2", "3", "4", "5", "6", "7", "8", "9", "the", "a", "an", "in", "on", "at", "for", "with", "is", "was", "are", "were", "to", "by", "as", "that", "this", "it", "but", "or", "if", "not"}

    for pos in outlier_positions:
        pos = pos.item()  # Get the integer
        if pos == 0:
            stats['counts']['initial_token'] += 1
        else:
            token_id = tokens[pos].item()
            token_str = tokenizer.decode([token_id]).strip()
            token_str_lower = token_str.lower()
            
            if token_str in strong_delimiters:
                stats['counts']['strong_delimiter'] += 1
            elif token_str_lower in weak_semantic_tokens:
                stats['counts']['weak_semantic'] += 1
            else:
                stats['counts']['other'] += 1

# %%
def plot_attn_outlier_distribution(stats, model_name):
    labels = list(stats['counts'].keys())
    counts = [stats['counts'][label] for label in labels]
    
    # 显示的标签，适当地换行以保持清晰
    display_labels = ['initial\ntoken', 'strong\ndelimiter', 'weak\nsemantic', 'other']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制条形图，显示累积注意力得分 >= 100 的token分布
    ax.bar(display_labels, counts, color=['cornflowerblue', 'mediumseagreen', 'orange', 'dimgrey'])
    
    ax.set_xlabel('Token Types', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_title(f'Token Distribution of Attention Outliers in {MODEL_TITLE_DICT[model_name]}', fontsize=16, fontweight='bold')
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, fontsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.grid(axis='y', color='0.75', linestyle='--')
    
    # 确保保存为PDF
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    plt.savefig(os.path.join(args.savedir, 'attention_outliers_bar_chart.pdf'), bbox_inches="tight", dpi=300)
    plt.show()

# 调用函数绘制条形图
plot_attn_outlier_distribution(stats, args.model)
# %%
