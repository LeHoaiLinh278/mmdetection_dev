from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class VistasDataset(CocoDataset):

    CLASSES = ('person', 'cyclist', 'traffic-light', 'traffic-sign', 'bicycle', 'car', 'motorcycle')
