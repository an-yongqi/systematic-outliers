import monkey_patch as mp

CACHE_DIR_BASE = "../model_weights"

MODEL_DICT_LLMs = {
    ### llama2 model
    "llama2_7b": {
        "model_id": f"{CACHE_DIR_BASE}/Llama-2-7b-hf",
    },
    "llama2_13b": {
        "model_id": f"{CACHE_DIR_BASE}/Llama-2-13b-hf",
    },
    "llama2_70b": {
        "model_id": f"{CACHE_DIR_BASE}/Llama-2-70b-hf", 
    },
    
    ### llama3 model
    "llama3_8b": {
        "model_id": f"{CACHE_DIR_BASE}/Meta-Llama-3-8B", 
    },

    ### llama2 chat model
    "llama2_7b_chat": {
        "model_id": f"{CACHE_DIR_BASE}/Llama-2-7b-chat-hf",
    }, 
    "llama2_13b_chat": {
        "model_id": f"{CACHE_DIR_BASE}/Llama-2-13b-chat-hf",
    }, 
    "llama2_70b_chat": {
        "model_id": f"{CACHE_DIR_BASE}/Llama-2-70b-chat-hf",
    }, 

    ### mistral model 
    "mistral_7b": {
        "model_id": f"{CACHE_DIR_BASE}/Mistral-7B-v0.1",
    },
    "mistral_moe": {
        "model_id": f"{CACHE_DIR_BASE}/Mixtral-8x7B-v0.1",
    },
    "mistral_7b_instruct":{
        "model_id": f"{CACHE_DIR_BASE}/Mistral-7B-Instruct-v0.2",
    },
    "mistral_moe_instruct": {
        "model_id": f"{CACHE_DIR_BASE}/Mixtral-8x7B-Instruct-v0.1",
    }, 

    ### phi-2
    "phi2": {
        "model_id": f"{CACHE_DIR_BASE}/phi-2",
    },

    ### falcon model
    "falcon_7b": {
        "model_id": f"{CACHE_DIR_BASE}/falcon-7b",
    },
    "falcon_40b": {
        "model_id": f"{CACHE_DIR_BASE}/falcon-40b",
    },

    ### mpt model 
    "mpt_7b": {
        "model_id": f"{CACHE_DIR_BASE}/mpt-7b",
    },
    "mpt_30b": {
        "model_id": f"{CACHE_DIR_BASE}/mpt-30b",
    },

    ### opt model 
    "opt_125m": {
        "model_id": f"{CACHE_DIR_BASE}/opt-125m", 
    },
    "opt_350m": {
        "model_id": f"{CACHE_DIR_BASE}/opt-350m", 
    },
    "opt_1.3b": {
        "model_id": f"{CACHE_DIR_BASE}/opt-1.3b", 
    },
    "opt_2.7b": {
        "model_id": f"{CACHE_DIR_BASE}/opt-2.7b", 
    },
    "opt_7b": {
        "model_id": f"{CACHE_DIR_BASE}/opt-6.7b", 
    },
    "opt_13b": {
        "model_id": f"{CACHE_DIR_BASE}/opt-13b", 
    },
    "opt_30b": {
        "model_id": f"{CACHE_DIR_BASE}/opt-30b", 
    },
    "opt_66b": {
        "model_id": f"{CACHE_DIR_BASE}/opt-66b",
    },

    ### gpt2 model 
    "gpt2": {
        "model_id": f"{CACHE_DIR_BASE}/gpt2",
    },
    "gpt2_medium": {
        "model_id": f"{CACHE_DIR_BASE}/gpt2-medium",
    },
    "gpt2_large": {
        "model_id": f"{CACHE_DIR_BASE}/gpt2-large",
    },
    "gpt2_xl": {
        "model_id": f"{CACHE_DIR_BASE}/gpt2-xl",
    },
}


MODEL_TITLE_DICT={"llama2_7b": "LLaMA-2-7B", "mistral_7b": "Mistral-7B", "llama3_8b": "LLaMA-3-8B",
        "llama2_13b_chat": "LLaMA-2-13B-chat", "llama2_70b_chat": "LLaMA-2-70B-chat",
        "llama2_7b_chat": "LLaMA-2-7B-chat", "llama2_13b": "LLaMA-2-13B", "llama2_70b": "LLaMA-2-70B", 
        "mistral_moe":"Mixtral-8x7B", "falcon_7b": "Falcon-7B", "falcon_40b": "Falcon-40B", "phi2": "Phi-2",
        "opt_7b":"OPT-7B", "opt_13b": "OPT-13B", "opt_30b": "OPT-30B", "opt_66b": "OPT-66B",
        "mpt_7b": "MPT-7B", "mpt_30b": "MPT-30B", "pythia_7b": "Pythia-7B", "pythia_12b": "Pythia-12B",
        "gpt2": "GPT-2", "gpt2_large": "GPT-2-Large", "gpt2_xl": "GPT-2-XL", "gpt2_medium": "GPT-2-Medium",
        "mistral_moe_instruct": "Mixtral-8x7B-Instruct", "mistral_7b_instruct": "Mistral-7B-Instruct"}


MODEL_LAYER_ENABLE_DICT = {
    "llama2_7b": mp.enable_llama_custom_decoderlayer,
    "llama2_7b_chat": mp.enable_llama_custom_decoderlayer,
    "llama2_13b": mp.enable_llama_custom_decoderlayer,
    "llama2_13b_chat": mp.enable_llama_custom_decoderlayer,
    "llama3_8b": mp.enable_llama_custom_decoderlayer,
    "mistral_moe": mp.enable_mixtral_custom_decoderlayer,
    "mistral_7b": mp.enable_mistral_custom_decoderlayer,
    "mistral_7b_instruct": mp.enable_mistral_custom_decoderlayer,
    "opt_7b": mp.enable_opt_custom_decoderlayer,
    "phi2": mp.enable_phi2_custom_decoderlayer,
    "mpt_7b": mp.enable_mpt_custom_decoderlayer,
    "falcon_7b": mp.enable_falcon_custom_decoderlayer,
}

ACTIVATION_OUTPUT_DICT = {
    "mistral_moe": {
        "zlim": (0, 5400),
        "yticks": ([2070, 3398], [2070, 3398]),
        "heavy_indices": [3, 8],
        "zticks": ([0, 5000], ["0", "5k"]),
        "layer_id": 1,
    },
    "llama2_13b": {
        "zlim": (0, 1200),
        "yticks": ([2100, 4743], [2100, 4743]),
        "heavy_indices": [0],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 3,
    },
    "llama2_13b_chat": {
        "zlim": (0, 1200),
        "yticks": ([2100, 4743], [2100, 4743]),
        "heavy_indices": [0],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 3,
    },
    "llama2_7b": {
        "zlim": (0, 2400),
        "yticks": ([1415, 2533], [1415, 2533]),
        "heavy_indices": [0, 3],
        "zticks": ([0, 1000, 2000], ["0", "1k", "2k"]),
        "layer_id": 1,
    },
    "llama2_7b_chat": {
        "zlim": (0, 2400),
        "yticks": ([1415, 2533], [1415, 2533]),
        "heavy_indices": [0, 3],
        "zticks": ([0, 1000, 2000], ["0", "1k", "2k"]),
        "layer_id": 1,
    },
    "llama3_8b": {
        "zlim": (0, 350),
        "yticks": ([788, 1384, 4062], [788, 1384, 4062]),
        "heavy_indices": [0],
        "zticks": ([0, 300], ["0", "300"]),
        "layer_id": 1,
    },
    "falcon_7b": {
        "zlim": (0, 1200),
        "yticks": ([], []),
        "heavy_indices": [0],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 3,
    },
    "mpt_7b": {
        "zlim": (0, 1200),
        "yticks": ([], []),
        "heavy_indices": [0],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 3,
    },
    "mistral_7b": {
        "zlim": (0, 350),
        "yticks": ([2070], [2070]),
        "heavy_indices": [4, 9],
        "zticks": ([0, 150, 300], ["0", "150", "300"]),
        "layer_id": 1,
    },
    "mistral_7b_instruct": {
        "zlim": (0, 350),
        "yticks": ([2070], [2070]),
        "heavy_indices": [4, 9],
        "zticks": ([0, 150, 300], ["0", "150", "300"]),
        "layer_id": 1,
    },
    "opt_7b": {
        "zlim": (0, 220),
        "yticks": ([2394], [2394]),
        "heavy_indices": [0],
        "zticks": ([0, 200], ["0", "200"]),
        "layer_id": 0,
    },
    "phi2": {
        "zlim": (0, 240),
        "yticks": ([], []),
        "heavy_indices": [1, 8],
        "zticks": ([0, 100, 200], ["0", "100", "200"]),
        "layer_id": 1,
    }
}

MODEL_DOWN_ENABLE_DICT = {
    "llama2_7b": mp.enable_llama_custom_mlp,
    "llama2_7b_chat": mp.enable_llama_custom_mlp,
    "llama2_13b_chat": mp.enable_llama_custom_mlp,
    "llama2_13b": mp.enable_llama_custom_mlp,
    "llama3_8b": mp.enable_llama_custom_mlp,
    "mistral_moe": mp.enable_mixtral_custom_mlp,
    "mistral_7b": mp.enable_mistral_custom_mlp,
    "mistral_7b_instruct": mp.enable_mistral_custom_mlp,
    "opt_7b": mp.enable_opt_custom_mlp,
    "phi2": mp.enable_phi2_custom_mlp,
    "mpt_7b": mp.enable_mpt_custom_mlp,
    "falcon_7b": mp.enable_falcon_custom_mlp,
}

DOWN_INPUT_DICT = {
    "mistral_moe": {
        "zlim": (0, 5400),
        "heavy_indices": [3, 8],
        "zticks": ([0, 5000], ["0", "5k"]),
        "layer_id": 1,
    },
    "mistral_7b": {
        "zlim": (0, 2100),
        "yticks": ([7310], [7310]),
        "heavy_indices": [4, 9],
        "zticks": ([0, 1000, 2000], ["0", "1k", "2k"]),
        "layer_id": 1,
    },
    "mistral_7b_instruct": {
        "zlim": (0, 2100),
        "yticks": ([7310], [7310]),
        "heavy_indices": [4, 9],
        "zticks": ([0, 1000, 2000], ["0", "1k", "2k"]),
        "layer_id": 1,
    },
    "llama2_13b": {
        "zlim": (0, 1200),
        "yticks": ([8811], [8811]),
        "heavy_indices": [0],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 3,
    },
    "llama2_13b_chat": {
        "zlim": (0, 1200),
        "yticks": ([8811], [8811]),
        "heavy_indices": [0],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 3,
    },
    "llama3_8b": {
        "zlim": (0, 550),
        "yticks": ([2427], [2427]),
        "heavy_indices": [0],
        "zticks": ([0, 500], ["0", "500"]),
        "layer_id": 1,
    },
    "llama2_7b": {
        "layer_params": {
            1: {
                "zlim": (0, 1200),
                "yticks": ([7890], [7890]),
                "heavy_indices": [0, 3],
                "zticks": ([0, 1000], ["0", "1k"]),
            },
            30: {
                "zlim": (0, 1000),
                "yticks": ([3721, 7006], [3721, 7006]),
                "heavy_indices": [0, 3],
                "zticks": ([0, 800], ["0", "800"]),
            },
        }
    },
    "llama2_7b_chat": {
        "zlim": (0, 1200),
        "yticks": ([7890], [7890]),
        "heavy_indices": [0, 3],
        "zticks": ([0, 1000], ["0", "1k"]),
        "layer_id": 1,
    },
    "opt_7b": {
        "zlim": (0, 4.4),
        "yticks": ([], []),
        "heavy_indices": [0],
        "zticks": ([0, 4], ["0", "4"]),
        "layer_id": 0,
    },
    "mpt_7b": {
        "zlim": (0, 5.4),
        "yticks": ([], []),
        "heavy_indices": [0],
        "zticks": ([0, 5], ["0", "5"]),
        "layer_id": 3,
    },
    "falcon_7b": {
        "zlim": (0, 32),
        "yticks": ([], []),
        "heavy_indices": [0],
        "zticks": ([0, 30], ["0", "30"]),
        "layer_id": 3,
    },
    "phi2": {
        "zlim": (0, 12),
        "yticks": ([], []),
        "heavy_indices": [1, 8],
        "zticks": ([0, 10], ["0", "10"]),
        "layer_id": 1,
    }
}

WEIGHT_DICT = {
    "mistral_7b": {
        "top_1": 7310,
        "yticks": ([2070], [2070]),
        "layer_id": 1,
        "selected_indices": [0, 1, 2, -1, -1, 7309, 7310, 7311, -1, -1, 14335],
    },
    "mistral_7b_instruct": {
        "top_1": 7310,
        "yticks": ([2070], [2070]),
        "layer_id": 1,
        "selected_indices": [0, 1, 2, -1, -1, 7309, 7310, 7311, -1, -1, 14335],
    },
    "llama2_13b": {
        "top_1": 8811,
        "yticks": ([2100, 4743], [2100, 4743]),
        "layer_id": 3,
        "selected_indices": [0, 1, 2, -1, -1, 8810, 8811, 8812, -1, -1, 13823],
    },
    "llama2_7b": {
        "top_1": 7890,
        "yticks": ([1415, 2533], [1415, 2533]),
        "layer_id": 1,
        "selected_indices": [0, 1, 2, -1, -1, 7889, 7890, 7891, -1, -1, 11007],
    },
    "llama2_13b_chat": {
        "top_1": 8811,
        "yticks": ([2100, 4743], [2100, 4743]),
        "layer_id": 3,
        "selected_indices": [0, 1, 2, -1, -1, 8810, 8811, 8812, -1, -1, 13823],
    },
    "llama2_7b_chat": {
        "top_1": 7890,
        "yticks": ([1415, 2533], [1415, 2533]),
        "layer_id": 1,
        "selected_indices": [0, 1, 2, -1, -1, 7889, 7890, 7891, -1, -1, 11007],
    },
    "llama3_8b": {
        "top_1": 12638,
        "yticks": ([], []),
        "layer_id": 1,
        "selected_indices": [0, 1, 2, -1, -1, -1, 12637, 12638, 12639, -1, 14335],
    },
    "opt_7b": {
        "top_1": 16351,
        "yticks": ([2394], [2394]),
        "layer_id": 0,
        "selected_indices": [0, 1, 2, -1, -1, -1, 16350, 16351, 16352, -1, 16383],
    },
    "mpt_7b": {
        "top_1": 903,
        "yticks": ([], []),
        "layer_id": 3,
        "selected_indices": [0, 1, 2, -1, 902, 903, 904, -1, -1, -1, 16383],
    },
    "falcon_7b": {
        "top_1": 1135,
        "yticks": ([], []),
        "layer_id": 3,
        "selected_indices": [0, 1, 2, -1, 1134, 1135, 1136, -1, -1, -1, 18175],
    },
    "phi2": {
        "top_1": 617,
        "yticks": ([], []),
        "layer_id": 1,
        "selected_indices": [0, 1, 2, -1, 616, 617, 618, -1, -1, -1, 10239],
    }
}

MODEL_ATTN_ENABLE_DICT = {
    "llama2_7b": mp.enable_llama_custom_attention,
    "llama2_13b": mp.enable_llama_custom_attention,
    "llama3_8b": mp.enable_llama_custom_attention,
    "mistral_moe": mp.enable_mixtral_custom_attention,
    "mistral_7b": mp.enable_mistral_custom_attention,
    "mistral_7b_instruct": mp.enable_mistral_custom_attention,
    "opt_7b": mp.enable_opt_custom_attention,
    "phi2": mp.enable_phi2_custom_attention,
    "mpt_7b": mp.enable_mpt_custom_attention,
    "falcon_7b": mp.enable_falcon_custom_attention,
    "llama2_7b_chat": mp.enable_llama_custom_attention,
    "llama2_13b_chat": mp.enable_llama_custom_attention,
}

ATTN_SCORE_DICT = {
    "mistral_moe": {
        "heavy_indices": [3, 8],
        "layer_id": 2,
    },
    "llama2_13b": {
        "heavy_indices": [0],
        "layer_id": 4,
    },
    "llama2_7b": {
        "heavy_indices": [0, 3],
        "layer_id": 2,
    },
    "llama3_8b": {
        "heavy_indices": [0],
        "layer_id": 2,
    },
    "opt_7b": {
        "heavy_indices": [0],
        "layer_id": 1,
    },
    "mpt_7b": {
        "heavy_indices": [0],
        "layer_id": 4,
    },
    "falcon_7b": {
        "heavy_indices": [0],
        "layer_id": 4,
    },
    "phi2": {
        "heavy_indices": [1, 8],
        "layer_id": 2,
    },
    "mistral_7b": {
        "heavy_indices": [4, 9],
        "layer_id": 2,
    },
    "mistral_7b_instruct": {
        "heavy_indices": [4, 9],
        "layer_id": 2,
    },
    "llama2_13b_chat": {
        "heavy_indices": [0],
        "layer_id": 4,
    },
    "llama2_7b_chat": {
        "heavy_indices": [0, 3],
        "layer_id": 2,
    },
}