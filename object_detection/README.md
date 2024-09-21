# 3D Object Detection on KITTI & nuScenes

## Setup

Follow the instructions in the [omni3D](https://github.com/facebookresearch/omni3d) repo for installation and download [KITTI & nuScenes](https://github.com/facebookresearch/omni3d/blob/main/DATA.md) datasets in the required format. 

Set the environment variable for data directory:
```bash
export DATA_DIR = <path_to_omni3d_data>
```

## Usage

We train [Cube R-CNN](https://arxiv.org/pdf/2207.10660) with KPE jointly on KITTI & nuScenes datasets on the car category. Check [configs/cubercnn_kpe.yaml](configs/cubercnn_kpe.yaml) for more details.

```bash
CUDA_VISIBLE_DEVICES = <gpu_ids> python tools/train_net.py --config-file configs/cubercnn_kpe.yaml --num-gpus=<total_gpus>
```

## Acknowledgements

This codebase is modified from the [omni3D](https://github.com/facebookresearch/omni3d) repo. We thank the authors for releasing their code and models. Refer to their repo for more details on licensing and citation.