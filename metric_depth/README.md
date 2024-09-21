# Dense Metric Depth Prediction on NYUv2

## Setup

Follow the instructions in the [ZoeDepth](https://github.com/isl-org/ZoeDepth) repo to download the NYU dataset, pretrained [MiDaS](https://github.com/isl-org/MiDaS) weights and installation.

Set the environment variable for the NYU dataset directory.
```bash
export NYU_ROOT_DIR = <path_to_nyu_dataset>
```

## Training

We train multiple variants of the model: with crop and/or scale augmentations, at two resolutions (96x128, 384x512), with & without pretrained MiDaS weights (change the arguments as needed.).

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py -m zoedepth --bs=4 --use_pretrained_midas=1 --save_dir ./logs/debug --input_height 96 --input_width 128 --random_crop=1 --test_on_crop=1 --use_gt_principal=1 --use_flip_aug=1 --use_scale_aug=1 --epochs=20 --workers=8 --lr 0.00001 --pos_enc dense_latent --n_freq_pos_enc 8 --midas_pos_enc=1
```

## Evaluation

We evaluate on the test set of the NYU dataset. We observe the most gains at lower resolution crops (96x128) with dense KPE.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python eval.py -m zoedepth --bs=1 --use_pretrained_midas=0 --checkpoint <path_to_checkpoint> --input_height 96 --input_width 128 --random_crop=1 --test_on_crop=1 --workers=1 --pos_enc dense_latent --n_freq_pos_enc 8 --midas_pos_enc=1
```

## Acknowledgements

This codebase is modified from the [ZoeDepth](https://github.com/isl-org/ZoeDepth) and [MiDaS](https://github.com/isl-org/MiDaS) repos. We thank the authors for releasing their code and data.