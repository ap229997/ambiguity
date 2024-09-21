from src.parsers.configs.generic import DEFAULT_ARGS_EGO

DEFAULT_ARGS_EGO["img_feat_version"] = ""
DEFAULT_ARGS_EGO["batch_size"] = 32
DEFAULT_ARGS_EGO["test_batch_size"] = 32
DEFAULT_ARGS_EGO["num_workers"] = 8
DEFAULT_ARGS_EGO["img_res"] = 224
DEFAULT_ARGS_EGO["img_res_ds"] = 224 # keep this same as img_res for simplicity
DEFAULT_ARGS_EGO["logger"] = 'tensorboard'
DEFAULT_ARGS_EGO['log_every'] = 50
DEFAULT_ARGS_EGO['vis_every'] = 100 # this only works in debug mode
DEFAULT_ARGS_EGO['backbone'] = 'resnet50'