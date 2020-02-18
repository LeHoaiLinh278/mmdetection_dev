import cv2
import json
import numpy as np
import pycocotools.mask as maskUtils
from mmcv.image import imread, imwrite
import mmcv
from imantics import Polygons, Mask  #(pip install imantics)

def save_json(image, bboxes, segm_result, labels, class_names, score_thr=0, format_file = "coco",out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = mmcv.imread(image)
    #img = img.copy()

    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        indxs = np.where(bboxes[:, -1] > score_thr)[0]

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    if format_file == 'coco':
        record = {}
        record['imgHeight'] = img.shape[0]
        record['imgWidth'] = img.shape[1]
        objects = []
        if out_file is not None:
            for i, (bbox, label, indx) in enumerate(zip(bboxes, labels, indxs)):
                bbox_int = bbox.astype(np.int32)
                mask = maskUtils.decode(segms[indx]).astype(np.bool)
                polygons = Mask(mask).polygons().segmentation
                label_text = class_names[
                    label] if class_names is not None else 'cls {}'.format(label)
                objects.append({'label' : label_text, 'polygon' : polygons,  'bbox' : bbox_int[:-1].tolist()})
            record['objects'] = objects
            with open(out_file.split('.')[0] + '.json', 'w') as f:
                json.dump(record, f, indent=2)

    elif format_file == 'labelme':
        if out_file is not None:
            group_id = 1
            record = {}
            record["version"] = "4.2.9"
            record["flags"] = {}
            objects = []
            for i, (bbox, label, indx) in enumerate(zip(bboxes, labels, indxs)):
                bbox_int = bbox.astype(np.int32)
                mask = maskUtils.decode(segms[indx]).astype(np.bool)
                polygons = Mask(mask).polygons().segmentation
                label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
                if len(polygons) == 1:
                    poly = np.array(polygons).reshape(-1,2)
                    objects.append({'label' : label_text, 'points' : poly.tolist(), "group_id" : None, "shape_type": "polygon", "flags": {} })
                else:
                    for p in polygons:
                        poly = np.array(p).reshape(-1, 2)
                        objects.append({'label': label_text, 'points': poly.tolist(), "group_id": group_id, "shape_type": "polygon","flags": {} })
                    group_id += 1
            record['shapes'] = objects
            record['imagePath'] = image
            record["imageData"] = None
            record["imageHeight"] = img.shape[0]
            record["imageWidth"] = img.shape[1]
            with open(out_file.split('.')[0] + '.json', 'w') as f:
                json.dump(record, f, indent=2)