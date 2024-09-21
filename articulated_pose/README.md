# 3D Pose Estimation of Articulated Objects in Contact

## Setup

Follow the instructions in [ARCTIC](https://github.com/zc-alexfan/arctic) repo to setup the environment and download the dataset. We use the egocentric split for experiments in our work.

After downloading the dataset, set the environment variable for the ARCTIC directory:
```bash
export DATA_DIR = <path_to_arctic_data>
```

## Training

All models are initialized from the ArcticNet-SF model pretrained on the allocentric split of the ARCTIC dataset. This checkpoint can be download from the ARCTIC repo.

To train ArcticNet-SF on the egocentric split, run (default arguments are provided in [arctic_sf.py](src/parsers/configs/arctic_sf.py), `p2` refers to egocentric split):
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts_method/train.py --setup p2 --method arctic_sf --load_ckpt <path_to_pretrained_ckpt> --exp_key <experiment_name> --trainsplit train --valsplit smallval
```

To the train ArcticNet-SF with KPE on the egocentric split, run (default arguments are provided in [arctic_kpe.py](src/parsers/configs/arctic_kpe.py)):
```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> python scripts_method/train.py --setup p2 --method arctic_kpe --load_ckpt <path_to_pretrained_ckpt> --exp_key <experiment_name> --trainsplit train --valsplit smallval
```

The scripts support multi-GPU training using pytorch lightning DDP. Modify the `gpu_ids` to specify the GPUs to use.

## Evaluation

Follow the instructions from the [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/model/README.md) repo for evaluation. We use the same protocol as ARCTIC and evaluate both on the validation set and the [leaderboard](https://arctic-leaderboard.is.tuebingen.mpg.de/leaderboard).

Since our model takes crops as input, we finetune a MaskRCNN model to predict the bounding boxes. The predicted bounding boxes for the val and test (leaderboard) splits are provided [here](https://drive.google.com/drive/folders/19iqjJ8Lt9LZMDk_rOTlk_OMirWIj4uLs?usp=sharing)).

## Acknowledgements

This codebase is modified from the awesome [ARCTIC](https://github.com/zc-alexfan/arctic) repo. We thank the authors for releasing their code and dataset. Check out their work as well.