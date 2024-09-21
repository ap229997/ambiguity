from src.datasets.crop_dataset import CropDataset


class CropDatasetEval(CropDataset):
    def getitem(self, imgname, load_rgb=True):
        return self.getitem_eval(imgname, load_rgb=load_rgb)
