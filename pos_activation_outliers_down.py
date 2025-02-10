#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib

from tqdm import tqdm

from lib.model_dict import MODEL_TITLE_DICT, MODEL_DOWN_ENABLE_DICT, DOWN_INPUT_DICT

args = SimpleNamespace(
    model="llama2_7b",
    seed=1,
    revision="main",
    access_token=None,
    savedir="figures/activation_outliers_pos"
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
    func = MODEL_DOWN_ENABLE_DICT.get(model_name)
    if not func:
        raise ValueError(f"No layer enable function found for model: {model_name}")
    return func

# %%
# 回答"哪些层"的问题
def plot_layer_ax_sub(ax, mean, model_name):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey"]

    x_axis = np.arange(mean.shape[-1])+1
    for i in range(3):
        ax.plot(x_axis, mean[i], label=f"Top {i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    ax.plot(x_axis, mean[-1], label=f"Median", color=colors[-1], 
                     linestyle="-",  marker="v", markerfacecolor='none', markersize=5)

    ax.set_title(f"{MODEL_TITLE_DICT[model_name]} Layer-wise Analysis", fontsize=18, fontweight="bold")

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel("Magnitudes", fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')

def plot_layer_ax(obj, model_name):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    mean = np.mean(obj, axis=0)
    plot_layer_ax_sub(axs, mean, model_name)
    leg = axs.legend(
        loc='center', bbox_to_anchor=(0.5, -0.25),
        ncol=4, fancybox=True, prop={'size': 14}
    )
    leg.get_frame().set_edgecolor('silver')
    leg.get_frame().set_linewidth(1.0)
    
    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)
        plt.savefig(os.path.join(args.savedir, f"{model_name}_down_layers.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()
    
# %%
enable_layer_func = get_layer_enable_function(args.model)
for layer_id in tqdm(range(len(layers))):
    enable_layer_func(layers[layer_id], layer_id)

testseq_list = lib.get_data(tokenizer, nsamples=10, seqlen=seq_len, device=device)

#%%
stats = []
for seqid, testseq in enumerate(testseq_list):
    print(f"processing seq {seqid}")
    with torch.no_grad():
        model(testseq)

    seq_np = np.zeros((4, len(layers)))
    for layer_id in range(len(layers)):
        feat_abs = layers[layer_id].mlp.feat.abs()
        top_values, _ = torch.topk(feat_abs.flatten(), 3)
        seq_np[:3, layer_id] = top_values
        seq_np[3, layer_id] = torch.median(feat_abs)

    stats.append(seq_np)

#%%
plot_layer_ax(stats, args.model)

# %%
# 回答"哪些通道"的问题
def plot_layer_channel_sub(ax, data, model_name, ylabel, ylim):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey"]

    x_axis = np.arange(data.shape[-1])+1
    for i in range(2):
        ax.plot(x_axis, data[i], label=f"Top {i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)

    ax.set_title(f"Top-{top_k} {ylabel} in {MODEL_TITLE_DICT[model_name]}", fontsize=18, fontweight="bold")

    num_layers = data.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=18, labelpad=0.8)
    ax.set_ylabel(f"{ylabel}", fontsize=18)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')
    ax.set_ylim(0, ylim * 1.1)

    
def plot_layer_channel(data, model_name, ylabel="Activation Channel Index", filename="activation_channels", ylim=None):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4.5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)

    plot_layer_channel_sub(axs, data, model_name, ylabel, ylim)
    leg = axs.legend(
        loc='center', bbox_to_anchor=(0.5, -0.25),
        ncol=4, fancybox=True, prop={'size': 14}
    )
    leg.get_frame().set_edgecolor('silver')
    leg.get_frame().set_linewidth(1.0)
    
    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)
        plt.savefig(os.path.join(args.savedir, f"{model_name}_{filename}.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()    

# %%    
top_k = 2  # 选择Top-k
activation_channel_indices = np.zeros((top_k, len(layers)), dtype=int)

seq = "Summer is warm. Winter is cold.\n"
valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

with torch.no_grad():
    model(valenc)
    
for layer_id in tqdm(range(len(layers))):
    feat_abs = layers[layer_id].mlp.feat.abs()
    # 计算每个通道的激活值的最大值
    channel_max_vals = feat_abs.max(dim=1).values
    # 获取Top-k的通道索引
    top_indices = torch.topk(channel_max_vals, top_k, largest=True).indices
    # 保存通道索引
    activation_channel_indices[:, layer_id] = top_indices.numpy()

#%%
# 绘制激活通道索引图
plot_layer_channel(activation_channel_indices, args.model, ylabel="Activation Channel Index", filename="down_features", ylim=feat_abs.shape[-1])

# %%
# 回答"哪些序列"的问题
def plot_3d_feat_sub(ax, obj, layer_id, model_name, config, savedir=None):
    num_tokens = len(obj[f"seq"])
    num_channels = obj[f"{layer_id}"].shape[2]
    inp_seq = obj[f"seq"]
    inp_seq = [x if x != "<0x0A>" else r"\n" for x in inp_seq]
    xdata = np.array([np.linspace(0,num_tokens-1,num_tokens) for i in range(num_channels)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_channels)])
    zdata = obj[f"{layer_id}"][0].abs().numpy().T
    
    heavy_indices = [0, 3]
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
    for idx in [0, 3]:
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
        plt.savefig(os.path.join(savedir, f"{model_name}_layer_{layer_id+1}_down_seqs.pdf"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()

def plot_3d_feat(obj, layer_id, model_name, config, savedir=None):
    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    
    # for i in range(3):
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3d_feat_sub(ax, obj, layer_id, model_name, config, savedir)

#%%
config = DOWN_INPUT_DICT.get(args.model, DOWN_INPUT_DICT["llama2_7b"])
# 获取适当的启用解码层函数
enable_layer_func = get_layer_enable_function(args.model)

# 判断是否有 layer_params，如果有就循环处理
if "layer_params" in config:
    layer_params = config["layer_params"]
    for layer_id, params in layer_params.items():
        enable_layer_func(layers[layer_id], layer_id)

        stats = {}
        seq = "Summer is warm. Winter is cold.\n"
        valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            model(valenc)

        seq_decoded = []
        for i in range(valenc.shape[1]):
            seq_decoded.append(tokenizer.decode(valenc[0, i].item()))

        stats[f"seq"] = seq_decoded
        feat_abs = layers[layer_id].mlp.feat.abs()
        stats[f"{layer_id}"] = feat_abs

        # 简化参数传递，将params直接传递给plot_3d_feat
        plot_3d_feat(stats, layer_id, args.model, config=params, savedir=args.savedir)

else:
    # 如果没有 layer_params，则处理单一层的情况
    layer_id = config["layer_id"]
    enable_layer_func(layers[layer_id], layer_id)

    stats = {}
    seq = "Summer is warm. Winter is cold.\n"
    valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    with torch.no_grad():
        model(valenc)

    seq_decoded = []
    for i in range(valenc.shape[1]):
        seq_decoded.append(tokenizer.decode(valenc[0, i].item()))

    stats[f"seq"] = seq_decoded
    feat_abs = layers[layer_id].mlp.feat.abs()
    stats[f"{layer_id}"] = feat_abs

    # 直接传递config作为参数
    plot_3d_feat(stats, layer_id, args.model, config=config, savedir=args.savedir)
# %%
