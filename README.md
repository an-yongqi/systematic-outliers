# Systematic Outliers in Large Language Models
---

This paper systematically analyzes outliers in large language models, revealing they function as implicit context-aware scaling factors within the attention mechanism, and propose a method to eliminate them to enhance model performance and efficiency.

## Setup 
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Outline
The contents of this repository are as follows:

* [lib](lib) contains the util function for loading models, plotting figures and evaluation.
* [monkey_patch](monkey_patch) contains the code for monkey patching LLMs with custom forward function, with a goal of collecting internal activation and attention statistics.
* [gpt-2](gpt-2) contains the code for Training GPT-2 to validate systematic outliers hypotheses.

### Python Scripts
The repository includes several Python scripts for different analyses related to systematic outliers:

#### Existence Visualization (`existence_*` scripts)
These scripts are used to visualize the existence of outliers in different components of the model:
* `existence_activation_outliers_down.py`: Visualizes the existence of activation outliers in the inputs of the down-projection.
* `existence_activation_outliers_layer.py`: Analyzes and visualizes activation outliers in the outputs of the layer.
* `existence_attention_outliers.py`: Visualizes the presence of outliers within the attention scores across different attention heads and layers.
* `existence_weight_outliers.py`: Visualizes the existence of outliers in the weight matrices of the model.

#### Positional Analysis (`pos_*` scripts)
These scripts are used to visualize the specific positions where outliers occur:
* `pos_activation_outliers_down.py`: Visualizes the specific positions of activation outliers in the inputs of the down-projection.
* `pos_activation_outliers_layer_more.py`: Visualizes the specific positions of activation outliers in the outputs of the layer
* `pos_attention_outliers.py`: Visualizes the specific positions of attention score outliers within the attention mechanism.
* `pos_weight_outliers.py`: Identifies the specific positions of outliers in the weight matrices.

#### Lifecycle Analysis (`lifecycle_*` scripts)
These scripts are used to analyze the lifecycle of outliers, from their formation to disappearance:
* `lifecycle_emergence.py`: Analyzes the emergence of activation outliers from weight outliers.
* `lifecycle_spread.py`: Visualizes the spread of attention outliers from activation outliers.
* `lifecycle_disappearance.py`: Analyzes the disappearance of outliers in the final layers.
