#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib

from tqdm import tqdm

from lib.model_dict import MODEL_TITLE_DICT, MODEL_LAYER_ENABLE_DICT, ACTIVATION_OUTPUT_DICT

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
    func = MODEL_LAYER_ENABLE_DICT.get(model_name)
    if not func:
        raise ValueError(f"No layer enable function found for model: {model_name}")
    return func

# %%
layer_id = 1
enable_layer_func = get_layer_enable_function(args.model)
enable_layer_func(layers[layer_id], layer_id)

testseq_list = lib.get_data(tokenizer, nsamples=100, seqlen=seq_len, device=device)

#%%
stats = {}
stats['counts'] = {'initial_token': 0, 'strong_delimiter': 0, 'weak_semantic': 0, 'other': 0}

# %%
for seqid, testseq in tqdm(enumerate(testseq_list)):
    with torch.no_grad():
        model(testseq)
        
    seq_decoded = []
    for i in range(testseq.shape[1]):
        seq_decoded.append(tokenizer.decode(testseq[0,i].item()))
        
    stats[f"seq"] = seq_decoded

    feat_abs = layers[layer_id].feat.abs()

    # Compute mean and max activations
    mean_feat_abs = feat_abs.mean()
    max_feat_abs = feat_abs.max(dim=-1)[0]  # Shape [bs, seqlen]
    outlier_mask = max_feat_abs > (1000 * mean_feat_abs)

    # Since batch size is 1
    max_feat_abs_seq = max_feat_abs[0]  # Shape [seqlen]
    outlier_mask_seq = outlier_mask[0]  # Shape [seqlen]
    outlier_positions = outlier_mask_seq.nonzero(as_tuple=False).squeeze()  # Positions in seq_len
    
    tokens = testseq[0]  # Shape [seqlen]

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
def plot_outlier_distribution(stats, model_name):
    labels = list(stats['counts'].keys())
    counts = [stats['counts'][label] for label in labels]
    
    # 修改显示的标签，保留字典中的原始键
    display_labels = ['initial\ntoken', '"."', '"_"']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a pie chart with improved aesthetics
    wedges, texts, autotexts = ax.pie(counts, labels=display_labels, autopct='%1.1f%%', 
                                      colors=['cornflowerblue', 'mediumseagreen', 'orange', 'dimgrey'], 
                                      startangle=140, wedgeprops={'linewidth': 0}, 
                                      textprops={'fontsize': 14})
    
    # Set better styling for percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title(f'Token Distribution of Activation Outliers\nin {MODEL_TITLE_DICT[model_name]}', fontsize=18, fontweight='bold')
    
    # Ensure directory exists for saving
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    plt.savefig(os.path.join(args.savedir, 'activation_outliers_pie_chart_improved.pdf'), bbox_inches="tight", dpi=300)
    plt.show()

# Call the function to plot the improved pie chart
plot_outlier_distribution(stats, args.model)

# %%
# 回答"哪些通道"的问题
def plot_layer_channel_sub(ax, data, model_name, ylabel, ylim):
    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal", "dimgrey", "orange", "magenta", "gold", "navy", "purple"]

    x_axis = np.arange(data.shape[-1])+1
    for i in range(data.shape[0]):  # 遍历每个样本
        for j in range(data.shape[1]):  # 遍历 top_k
            ax.scatter(x_axis, data[i, j], label=f"Sample {i+1} Top {j+1}", color=colors[j % len(colors)], 
                       marker="o", edgecolor='none', alpha=1)

    ax.set_title(f"Top-{top_k} {ylabel} in {MODEL_TITLE_DICT[model_name]}", fontsize=18, fontweight="bold")

    num_layers = data.shape[-1]
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
    
    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)
        plt.savefig(os.path.join(args.savedir, f"{model_name}_{filename}.pdf"), bbox_inches="tight", dpi=300)

    plt.show()
    plt.close()    


# %%
enable_layer_func = get_layer_enable_function(args.model)
for layer_id in tqdm(range(len(layers))):
    enable_layer_func(layers[layer_id], layer_id)
    
# Initialize to accumulate results from multiple samples
top_k = 2  # Number of top channels to track
num_samples = 1  # Number of samples
activation_channel_indices = np.zeros((num_samples, top_k, len(layers)), dtype=int)

testseq_list = lib.get_data(tokenizer, nsamples=num_samples, seqlen=seq_len, device=device)

# Process multiple sequences
for seqid, testseq in tqdm(enumerate(testseq_list)):
    with torch.no_grad():
        model(testseq)

    for layer_id in range(len(layers)):
        feat_abs = layers[layer_id].feat.abs()
        channel_max_vals = feat_abs.max(dim=1).values  # Max activation values for each channel
        top_indices = torch.topk(channel_max_vals, top_k, largest=True).indices  # Top-k channels

        # Store top-k indices for each sample and layer
        activation_channel_indices[seqid, :, layer_id] = top_indices.cpu().numpy()

#%%
# 绘制10条样本的激活通道折线图
plot_layer_channel(activation_channel_indices, args.model, ylabel="Activation Channel Index", filename="layer_features_multiple_samples", ylim=feat_abs.shape[-1])

# %%
