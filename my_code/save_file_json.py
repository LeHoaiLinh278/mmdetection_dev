import cv2
import json
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.image import imread, imwrite
import mmcv
from imantics import Polygons, Mask
from shapely.geometry import Polygon

def save_json(image, bboxes, segm_result, labels, class_names, score_thr=0, out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = image.copy()

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)

    record = {}
    record['imgHeight'] = img.shape[0]
    record['imgWidth'] = img.shape[1]
    objects = []
    if out_file is not None:
        for i, (bbox, label, segm) in enumerate(zip(bboxes, labels, segms)):
            bbox_int = bbox.astype(np.int32)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            polygons = Mask(mask).polygons().segmentation
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            objects.append({'label' : label_text, 'polygon' : polygons,  'bbox' : bbox_int[:-1].tolist()})
        record['objects'] = objects
        with open(out_file.split('.')[0] + '.json', 'w') as f:
            json.dump(record, f, indent=2)