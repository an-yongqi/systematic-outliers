#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from types import SimpleNamespace
import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import lib
import seaborn as sns

from lib.model_dict import MODEL_TITLE_DICT, MODEL_ATTN_ENABLE_DICT, ATTN_SCORE_DICT

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


def get_layer_enable_function(model_name):
    func = MODEL_ATTN_ENABLE_DICT.get(model_name)
    if not func:
        raise ValueError(f"No layer enable function found for model: {model_name}")
    return func


# %%
def plot_attn_sub(ax, obj, layer_id, model_name, config, savedir=None):
    corr = obj[f"{layer_id}"].numpy()[0].astype("float64")
    num_tokens = len(obj[f"seq"])
    inp_seq = obj["seq"]
    inp_seq = [x if x != "<0x0A>" and x!= "\n" else r"\n" for x in inp_seq]

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
    
    ax.set_title(f"{MODEL_TITLE_DICT[model_name]}, Layer {layer_id+1}", fontsize=18, fontweight="bold", y=1.015)

    if savedir:
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, f"attno_{model_name}.pdf"), bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()

def plot_attn(obj, layer_id, model_name, config, savedir=None):
    fig = plt.figure(figsize=(8, 4.75))
    fig.tight_layout() 
    plt.subplots_adjust(wspace=0.15)
    
    ax = fig.add_subplot(1, 1, 1)
    plot_attn_sub(ax, obj, layer_id, model_name, config, savedir)


config = ATTN_SCORE_DICT.get(args.model, ATTN_SCORE_DICT["llama2_7b"])
enable_layer_func = get_layer_enable_function(args.model)

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
if args.model == "opt_7b":
    attn_logit = layers[layer_id].self_attn.attn_probs.unsqueeze(0).mean(axis=1)
elif args.model == "mpt_7b":
    attn_logit = layers[layer_id].attn.attn_probs.mean(axis=1)
elif args.model == "falcon_7b":
    attn_logit = layers[layer_id].self_attention.attn_probs.mean(axis=1)
else:
    attn_logit = layers[layer_id].self_attn.attn_probs.mean(axis=1)
stats[f"{layer_id}"] = attn_logit
plot_attn(stats, layer_id, args.model, config, args.savedir)

# %%
