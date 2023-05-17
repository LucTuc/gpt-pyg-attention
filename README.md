
# Self-Attention for Transformers with Graph Attention Networks
This repository contains an implementation of GPT where the original self-attention mechanism is replaced by Graph Attention Networks ([GAT](https://arxiv.org/abs/1710.10903) (from now on referred to as GATv1) and [GATv2](https://arxiv.org/abs/2105.14491)).

## Motivation
My area of expertise are graph neural networks (GNNs), which utilize a message-passing mechanism to exchange information between nodes during training. Interestingly, the self-attention mechanism, which is the "heart" of modern transformer networks, can also be thought of as a communication mechanism between nodes in a directed graph. In this graph, the tokens are represented as nodes connected by directed edges. In the decoder part of the transformer implemented here, each token in a given context is connected to itself and all following tokens. An example of a context graph of size 4 is visualized below, where T<sub>1</sub> through T<sub>4</sub> represent individual tokens within the given context. You can find a more detailed comparison between transformers and GNNs in [this blogpost](https://graphdeeplearning.github.io/post/transformers-are-gnns/).

<p align="center">
    <img src="https://github.com/LucTuc/gpt-pyg-attention/blob/master/illustrations/token_graph.png?raw=true" width="180" class="center">
</p>

Thus, the first goal of this project was to familiarize myself with transformers and create a direct connection to my previous research. The second goal was to create a plug-and-play framework where the self-attention can be replaced by any GNN. As everything is implemented using classes from Pytorch Geometric, it is very simple to replace GAT layers with any [convolutional GNN layer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers) from Pytorch Geometric. So feel free to fork the repository and try out different GNN layers and let me know the results! :)

## Usage
First, install the conda environment:
```
conda env create -f gpt.yml
```
Then, run the code as:
```
python pyg-gpt.py --gat_version GATConv
```
where the `--gat_version` argument is set to either GATConv or GATv2Conv.

## Results
The following plot shows the validation losses for the different models. They were all trained for 5000 epochs on the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset:

<p align="center">
    <img src="https://github.com/LucTuc/gpt-pyg-attention/blob/master/illustrations/ValLoss.png?raw=true" width="512" class="center">
</p>

GATv2, which fixes the static attention issue of vanilla GAT, reaches a similary low loss as the original transformer implementation. You can look at example outputs of all models in the model_outputs folder. Although they all produce non-sense, there is a very obvious improvements in the models that use attention over the simple bigram model, and it undoubtedly starts to resemble Shakespeare's style.

Note that I trained these models on an A100 GPU, so reproduction on your local machine might require downsizing the model parameters quite a bit.

## Acknowledgements 
This codebase and specifically the `gpt.py` and `bigram.py` code was originally created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series by Andrej Karpathy, specifically on the first lecture on nanoGPT. Many thanks to him for making the code available and teaching me the basics of LLMs in an intuitive manner.