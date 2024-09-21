from src.callbacks.loss.loss_arctic_sf import compute_loss
from src.callbacks.process.process_arctic import process_data
from src.callbacks.vis.visualize_arctic import visualize_all
from src.models.arctic_kpe.model import ArcticKPE
from src.models.generic.wrapper import GenericWrapper


class ArcticKPEWrapper(GenericWrapper):
    def __init__(self, args, push_images_fn):
        super().__init__(args, push_images_fn)
        self.model = ArcticKPE(
            backbone=args.backbone,
            focal_length=args.focal_length,
            img_res=args.img_res,
            args=args,
        )
        self.process_fn = process_data
        self.loss_fn = compute_loss
        self.metric_dict = [
            "cdev",
            "mrrpe",
            "mpjpe.ra",
            "aae",
            "success_rate",
        ]

        self.vis_fns = [visualize_all]

        self.num_vis_train = 1
        self.num_vis_val = 1

    def inference(self, inputs, meta_info):
        return super().inference_pose(inputs, meta_info)