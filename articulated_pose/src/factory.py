import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from common.torch_utils import reset_all_seeds
from common.tb_utils import push_images
from src.datasets.arctic_dataset import ArcticDataset
from src.datasets.arctic_dataset_eval import ArcticDatasetEval
from src.datasets.crop_dataset import CropDataset
from src.datasets.crop_dataset_eval import CropDatasetEval


def fetch_dataset_eval(args, seq=None):
    if args.method in ["arctic_sf"]:
        DATASET = ArcticDatasetEval
    elif args.method in ["arctic_kpe"]:
        DATASET = CropDatasetEval
    else:
        assert False
    if seq is not None:
        split = args.run_on
    ds = DATASET(args=args, split=split, seq=seq)
    return ds


def fetch_dataset_devel(args, is_train, seq=None):
    split = args.trainsplit if is_train else args.valsplit
    if args.method in ["arctic_sf"]:
        if is_train:
            DATASET = ArcticDataset
        else:
            DATASET = ArcticDataset
    elif args.method in ["arctic_kpe"]:
        if is_train:
            DATASET = CropDataset
        else:
            DATASET = CropDataset
    else:
        assert False
    if seq is not None:
        split = args.run_on

    ds = DATASET(args=args, split=split, seq=seq)
    return ds


def collate_custom_fn(data_list):
    data = data_list[0]
    _inputs, _targets, _meta_info = data
    out_inputs = {}
    out_targets = {}
    out_meta_info = {}

    for key in _inputs.keys():
        out_inputs[key] = []

    for key in _targets.keys():
        out_targets[key] = []

    for key in _meta_info.keys():
        out_meta_info[key] = []

    for data in data_list:
        inputs, targets, meta_info = data
        for key, val in inputs.items():
            out_inputs[key].append(val)

        for key, val in targets.items():
            out_targets[key].append(val)

        for key, val in meta_info.items():
            out_meta_info[key].append(val)

    for key in _inputs.keys():
        out_inputs[key] = torch.cat(out_inputs[key], dim=0)

    for key in _targets.keys():
        out_targets[key] = torch.cat(out_targets[key], dim=0)

    for key in _meta_info.keys():
        if key not in ["imgname", "query_names"]:
            out_meta_info[key] = torch.cat(out_meta_info[key], dim=0)
        else:
            out_meta_info[key] = sum(out_meta_info[key], [])

    return out_inputs, out_targets, out_meta_info


def fetch_dataloader(args, mode, seq=None):
    if mode == "train":
        reset_all_seeds(args.seed)
        dataset = fetch_dataset_devel(args, is_train=True)
        if type(dataset) == ArcticDataset or type(dataset) == CropDataset:
            collate_fn = None
        else:
            collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            shuffle=args.shuffle_train,
            collate_fn=collate_fn,
        )

    elif mode == "val" or mode == "eval":
        if "submit_" in args.extraction_mode:
            dataset = fetch_dataset_eval(args, seq=seq)
        else:
            dataset = fetch_dataset_devel(args, is_train=False, seq=seq)
        if type(dataset) in [ArcticDataset, ArcticDatasetEval, CropDataset, CropDatasetEval]:
            collate_fn = None
        else:
            collate_fn = collate_custom_fn
        return DataLoader(
            dataset=dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
    else:
        assert False


def fetch_model(args):
    if args.method in ["arctic_sf"]:
        from src.models.arctic_sf.wrapper import ArcticSFWrapper as Wrapper
    elif args.method in ["arctic_kpe"]:
        from src.models.arctic_kpe.wrapper import ArcticKPEWrapper as Wrapper
    else:
        assert False, f"Invalid method ({args.method})"
    
    if args.logger == "comet":
        model = Wrapper(args)
    elif args.logger == "tensorboard":
        model = Wrapper(args, push_images_fn=push_images)
    return model
