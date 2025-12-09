# BindCraft MCP service

## Overview
This repository creates MCP service for [BindCraft](https://github.com/google-deepmind/bindcraft). It supports the BindCraft binder design workflow:
1. Run BindCraft with predefined a3m files

## Installation
```shell
mamba env create -p ./env python=3.10 pip -y 
mamba activate ./env
pip install --ignore-installed fastmcp loguru

# Install BindCraft
git clone https://github.com/charlesxu90/bindcraft repo/BindCraft

mamba install pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
    -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu -y

# Install ColabDesign
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps

# Install and extract AlphaFold2 Weights
mkdir -p repo/BindCraft/params
wget -O repo/BindCraft/alphafold_params_2022-12-06.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar tf repo/BindCraft/alphafold_params_2022-12-06.tar
tar -xvf repo/BindCraft/alphafold_params_2022-12-06.tar -C repo/BindCraft/params
rm repo/BindCraft/alphafold_params_2022-12-06.tar

# Chmod 
chmod +x repo/BindCraft/functions/dssp
chmod +x repo/BindCraft/functions/DAlphaBall.gcc
```

## Local usage
### 1. Run Binder design with a PDB file and configs
```shell
python -u scripts/run_bindcraft.py --settings examples/PDL1/target.json --filters examples/PDL1/default_filters.json  --advanced examples/PDL1/default_4stage_multimer.json
```

## MCP usage

### Debug MCP server
```shell
cd tool-mcps/bindcraft_mcp
mamba activate ./env
fastmcp run src/bindcraft_mcp.py:mcp --transport http --port 8001 --python ./env/bin/python 
# Test config path:
# /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/bindcraft_mcp/examples/PDL1/target.json
```

### Install MCP server
```shell
fastmcp install claude-code tool-mcps/bindcraft_mcp/src/bindcraft_mcp.py --python tool-mcps/bindcraft_mcp/env/bin/python
fastmcp install gemini-cli tool-mcps/bindcraft_mcp/src/bindcraft_mcp.py --python tool-mcps/bindcraft_mcp/env/bin/python
```
### Call MCP service
1. Basic end to end structure prediction give sequences
```markdown
Please predict the complex of 1iep (seq: MDPSSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVSAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQ) and ligand (Smiles: Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1) using the bindcraft_mcp. save it to @examples/1iep_raw .

Please convert the relative path to absolution path before calling the MCP servers. 
```