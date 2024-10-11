#!/bin/bash

# sudo apt-get update
# sudo apt-get install bzip2 libxml2-dev

# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# rm Miniconda3-latest-Linux-x86_64.sh

source .bashrc
conda init

conda create -n gnn python=3.10

eval "$(conda shell.bash hook)"
conda activate gnn

pip install ogb==1.3.1
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

python gnn_lp.py --K KG_emb__distMult.pt