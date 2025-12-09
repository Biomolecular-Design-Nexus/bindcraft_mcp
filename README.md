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
git clone https://github.com/charlesxu90/bindcraft scripts

mamba install pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
    -c conda-forge -c nvidia --channel https://conda.graylab.jhu.edu -y

# Install ColabDesign
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps

# Install and extract AlphaFold2 Weights
mkdir -p scripts/params
wget -O scripts/alphafold_params_2022-12-06.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar tf scripts/alphafold_params_2022-12-06.tar
tar -xvf scripts/alphafold_params_2022-12-06.tar -C scripts/params
rm scripts/alphafold_params_2022-12-06.tar

# Chmod 
chmod +x scripts/functions/dssp
chmod +x scripts/functions/DAlphaBall.gcc
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
1. Submit a job for end to end binder design give a target pdb structure
```markdown
Please design binder for target PDL1 with PDB file @examples/PDL1/PDL1.pdb with settings @examples/PDL1/*.json using the bindcraft_mcp. save it to @examples/PDL1_results_test. 

After submitting the job, please query status 

Please convert the relative path to absolution path before calling the MCP servers. 
```

2. Query the status of a submitted binder design job
```markdown
Please check status of the binder design task @examples/PDL1/result_mcp using the bindcraft_mcp.

Please convert the relative path to absolution path before calling the MCP servers. 
```