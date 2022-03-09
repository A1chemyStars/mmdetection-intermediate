import gc
import os
import json
import tracemalloc
import numpy as np
import torch.nn as nn

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from kitti import model_configs
from util import ProgressBar, Hook
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from kitti_extract import iou


data_dir = '/home/kengo/Documents/GitHub/mmdetection-intermediate/data/kitti/data_object_image_2/training/image_2'

'''
BDD100K Categories:        KITTI Categories:
1 person                   1 Car
2 rider                    2 Van
3 car                      3 Truck
4 truck                    4 Pedestrian
5 bus                      5 Person_sitting
6 train                    6 Cyclist
7 motorcycle               7 Tram
8 bicycle                  8 Misc
'''

kitti_bdd_map = {0: 2, 1: 2, 2: 3, 3: 0, 4: 10, 5: 1, 6: 5, 7: 10}


def eval_cityscapes_on_kitti(detector='faster_rcnn_cityscapes', num_classes=8):
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
    # tracemalloc.start()
    progress = ProgressBar(len(images), fmt=ProgressBar.FULL)
    for ind, img in enumerate(images):
        img_file = img['file_name']
        img_path = os.path.join(data_dir, img_file)
        obj_img = []
        while True:
            if count < num_obj and labels[count]['image_id'] == ind:
                bbox = labels[count]['bbox']
                obj = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], kitti_bdd_map[labels[count]['category_id']]]
                obj_img.append(obj)
                count += 1
            else:
                break
        hook_fc_share = Hook(model.roi_head.bbox_head.shared_fcs._modules['1'])
        hook_cls = Hook(model.roi_head.bbox_head.fc_cls)

        results = inference_detector(model, img_path)
        if len(results) != 2:
            continue
            # print(ind, img)
            # print(results)
            # raise ValueError('Length of results greater than 2')
        result = results[0]
        indices = results[1]
        hook_fc_share.calc()
        fc = hook_fc_share.cal_output.cpu().numpy().copy().tolist()
        logit = hook_cls.output.cpu().numpy().copy().tolist()
        # pred = np.argmax(logit, axis=1)
        soft = nn.Softmax(dim=1)
        soft_obj = soft(hook_cls.output).data.cpu().numpy().copy()
        soft_obj = np.max(soft_obj, axis=1).tolist()

        for i in range(len(result)):
            result_cls = result[i]
            indices_cls = indices[i]
            if result_cls is not None:
                for j, item in enumerate(result_cls):
                    ind_match = indices_cls[j] // num_classes
                    ind_offset = indices_cls[j] % num_classes
                    assert ind_offset == i
                    match_flag = False
                    for obj in obj_img:
                        if iou(item[:-1], obj[:-1]) > 0.7:
                            if obj[-1] == i or i == 4 and obj[-1] == 3:
                                match_flag = True
                                break
                    preds.append(i)
                    fcs.append(fc[ind_match])
                    logits.append(logit[ind_match])
                    softmax.append(soft_obj[ind_match])
                    if match_flag:
                        flags.append(1)
                    else:
                        flags.append(0)
        hook_fc_share.close()
        hook_cls.close()
        # gc.collect()
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        progress.current += 1
        progress()

    np.save('city_fcs.npy', np.array(fcs))
    np.save('city_logits.npy', np.array(logits))
    np.save('city_softmax.npy', np.array(softmax))
    np.save('city_preds.npy', np.array(preds))
    np.save('city_flags.npy', np.array(flags))


if __name__ == '__main__':
    eval_cityscapes_on_kitti()
