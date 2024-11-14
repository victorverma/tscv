#!/bin/bash

conda env create --prefix env/ -f env.yaml
conda activate env/
python -m ipykernel install --user --name=env --display-name="Python (env)"
conda deactivate
