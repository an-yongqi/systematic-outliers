import os

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_ax_gpt2(feat, model_name, inp_seq, savedir, ckpt_iter):
    fig = plt.figure(figsize=(6,5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.1)

    ax = fig.add_subplot(1,1,1, projection='3d')

    name_title = {
        "gpt2_default": "GPT-2 Default",
        "gpt2_attn_bias": "GPT-2 with Attention Bias",
        "gpt2_cabias": "GPT-2 with Context-aware Bias",
        "gpt2_ca_softmax": "GPT-2 with Context-aware Scaling Factor",
        "gpt2_vbias": "GPT-2 with V Bias",
        "gpt2_sigmoid": "GPT-2 with Sigmoid Attenion",
    }
    num_tokens = feat.shape[1]
    num_channels = feat.shape[2]
    xdata = np.array([np.linspace(0,num_tokens-1,num_tokens) for i in range(num_channels)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_channels)])
    zdata = feat[0].numpy().T
        
    ax.plot_wireframe(xdata, ydata, zdata, rstride=0, color="blue", linewidth=1.5)

    ax.set_xticks(np.linspace(0,num_tokens-1,num_tokens), inp_seq, 
                      rotation=60, fontsize=16)

    ax.set(yticklabels=[])
    ax.set_zticks([0, 1000, 2000], ["0", "1k", "2k"], fontsize=18)
    if model_name == "gpt2_ca_softmax":
        title_size = 14
    elif model_name == "gpt2_cabias":
        title_size = 16
    else:
        title_size = 20

    ax.set_title(f"{name_title[model_name]}", fontsize=title_size, fontweight="bold", y=1.005)
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right", va="center",rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), ha="left",rotation_mode="anchor")
    plt.setp(ax.get_zticklabels(), ha="left",rotation_mode="anchor")

    ax.tick_params(axis='x', which='major', pad=-5)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-5)
    plt.savefig(os.path.join(savedir, f"{model_name}_3d.pdf"), bbox_inches="tight", dpi=300)

def plot_layer_ax_gpt2(mean, model_name, savedir=None, ckpt_iter=None):
    fig = plt.figure(figsize=(6,5))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.1)

    ax = fig.add_subplot(1,1,1)

    name_title = {
        "gpt2_default": "GPT-2 Default",
        "gpt2_attn_bias": "GPT-2 with Attention Bias",
        "gpt2_cabias": "GPT-2 with Context-aware Bias",
        "gpt2_ca_softmax": "GPT-2 with Context-aware Scaling Factor",
        "gpt2_vbias": "GPT-2 with V Bias",
        "gpt2_sigmoid": "GPT-2 with Sigmoid Attenion",
    }

    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal", "dimgrey"]
    x_axis = np.arange(mean.shape[-1])+1
    
    for i in range(3):
        ax.plot(x_axis, mean[i], label=f"Top {i+1}", color=colors[i], 
                     linestyle="-",  marker="o", markerfacecolor='none', markersize=5)
    
    ax.plot(x_axis, mean[-1], label=f"Median", color=colors[-1], 
                     linestyle="-",  marker="v", markerfacecolor='none', markersize=5)

    if model_name == "gpt2_ca_softmax":
        title_size = 16
    else:
        title_size = 18

    ax.set_title(f"{name_title[model_name]}", fontsize=title_size, fontweight="bold")
    ax.set_ylabel("Magnitudes", fontsize=18)

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax.set_xticks(xtick_label, xtick_label, fontsize=16)

    ax.set_xlabel('Layers', fontsize=18, labelpad=0.3)
    ax.tick_params(axis='x', which='major', pad=1.0)
    ax.tick_params(axis='y', which='major', pad=0.4)
    ax.grid(axis='x', color='0.75')
    plt.savefig(os.path.join(savedir, f"{model_name}_layerwise.pdf"), bbox_inches="tight", dpi=300)