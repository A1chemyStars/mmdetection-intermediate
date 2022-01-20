import gc
import os
import json
import tracemalloc
import numpy as np
import torch
import torch.nn as nn

from kitti import model_configs
from util import ProgressBar, Hook
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from MulticoreTSNE import MulticoreTSNE as TSNE


data_dir = '/home/kengo/Documents/GitHub/mmdetection-intermediate/data/kitti/data_object_image_2/training/image_2'


def iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box1[3])
    intersec = max(0.0, x_right - x_left + 1) * max(0.0, y_bottom - y_top + 1)
    if intersec == 0:
        return 0
    else:
        return intersec / ((box1[2]-box1[0]) * (box1[3]-box1[1]) + (box2[2]-box2[0]) * (box2[3]-box2[1]) - intersec)


def eval_kitti_detection(detector='faster_rcnn', num_classes=8):
    model = init_detector(model_configs[detector]['config_file'],
                          model_configs[detector]['checkpoint'], device='cuda:0')
    # hook_fc_share = Hook(model.roi_head.bbox_head.shared_fcs._modules['1'])
    # hook_cls = Hook(model.roi_head.bbox_head.fc_cls)

    fcs = []
    logits = []
    softmax = []
    preds = []
    flags = []

    with open('../data/kitti/data_object_label_2/training/label_2_coco/instances.json', 'r') as f:
        anno = json.load(f)
    images = anno['images']
    labels = anno['annotations']
    num_obj = len(labels)
    count = 0
    tracemalloc.start()
    for ind, img in enumerate(images):
        img_file = img['file_name']
        img_path = os.path.join(data_dir, img_file)
        obj_img = []
        while True:
            if count < num_obj and labels[count]['image_id'] == ind:
                bbox = labels[count]['bbox']
                obj = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], labels[count]['category_id']]
                obj_img.append(obj)
                count += 1
            else:
                break
        hook_fc_share = Hook(model.roi_head.bbox_head.shared_fcs._modules['1'])
        hook_cls = Hook(model.roi_head.bbox_head.fc_cls)

        result = inference_detector(model, img_path)
        hook_fc_share.calc()
        fc = hook_fc_share.cal_output.cpu().numpy().copy().tolist()
        logit = hook_cls.output.cpu().numpy().copy().tolist()
        pred = np.argmax(logit, axis=1)
        soft = nn.Softmax(dim=1)
        soft_obj = soft(hook_cls.output).data.cpu().numpy().copy()
        soft_obj = np.max(soft_obj, axis=1)

        for i in range(len(result)-1):
            result_cls = result[i]
            if result_cls is not None:
                ind_cls = np.where(pred == i)[0]
                for item in result_cls:
                    ind_match = None
                    match_flag = False
                    for j in range(len(ind_cls)):
                        if item[-1] == soft_obj[ind_cls[j]]:
                            ind_match = ind_cls[j]
                            break
                    for obj in obj_img:
                        if iou(item[:-1], obj[:-1]) > 0.7 and obj[-1] == i:
                            match_flag = True
                            break
                    if ind_match is not None:
                        preds.append(i)
                        fcs.append(fc[ind_match])
                        logits.append(logit[ind_match])
                        softmax.append(soft_obj[ind_match])
                        if match_flag:
                            flags.append(1)
                        else:
                            flags.append(0)
        del result
        del fc
        del logit
        hook_cls.close()
        hook_fc_share.close()
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

    np.save('fcs.npy', np.array(fcs))
    np.save('logits.npy', np.array(logits))
    np.save('softmax.npy', np.array(softmax))
    np.save('preds.npy', np.array(preds))
    np.save('flags.npy', np.array(flags))


if __name__ == '__main__':
    eval_kitti_detection()
