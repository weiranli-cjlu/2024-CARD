# CARD
this is the source code of CARD
## Requirement
```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric pygod --torch-backend=cu128
```

## Usage
python main.py --dataset cora --gama 0.6 --beta 0.9

* please make sure the diff of a dataset is generated before running. If not please use the ***gdc()*** in ***aug.py*** to generate.

* For the rest of the datasets, please refer to the .sh files, and simply run:
```Bash
chmod +x runEachDataset.sh
./runEachDataset.sh
```

