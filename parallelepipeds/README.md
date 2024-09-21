# Parallelepipeds Case Study

## Setup
```
conda create -n ambiguity python=3.8
conda activate ambiguity
pip install -r requirements.txt
```

The data used in parallelpipeds case study is generated on the fly during training and validation (see `ppd_network.py`).

## Usage
For experiments on predicting root relative or absolute 3D shape from 2D image crops (Fig.3 in the paper), run the following commands:
```
CUDA_VISIBLE_DEVICES=0 python ppd_network.py --output_type absolute --lr 1e-2 --fov 70 --version ours
CUDA_VISIBLE_DEVICES=0 python ppd_network.py --output_type absolute --lr 1e-2 --fov 70 --version baseline
CUDA_VISIBLE_DEVICES=0 python ppd_network.py --output_type relative --lr 1e-3 --fov 70 --version ours
CUDA_VISIBLE_DEVICES=0 python ppd_network.py --output_type relative --lr 1e-3 --fov 70 --version baseline
```

The plots can be visualized in `ppd_plots.ipynb`.

The plots in Fig.2 can be generated using `script_ppd.py`.