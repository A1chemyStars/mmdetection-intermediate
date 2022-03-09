import os
import json
import torch
from util import ProgressBar, Hook
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

kitti_root_dir = '../data/kitti_test'
vkitti_root_dir = '../data/vkitti_test'

model_configs = {'faster_rcnn':
                     {'config_file': '../checkpoints/kitti/faster_rcnn/faster_rcnn_r50_fpn_1x_kitti.py',
                      'checkpoint': '../checkpoints/kitti/faster_rcnn/epoch_16.pth'},
                 'yolov3':
                     {'config_file': '../checkpoints/kitti/yolo/yolov3_d53_mstrain-608_273e_kitti.py',
                      'checkpoint': '../checkpoints/kitti/yolo/latest.pth'},
                 'retinanet_r50':
                     {'config_file': '../checkpoints/kitti/retinanet/r50/retinanet_r50_fpn_1x_kitti.py',
                      'checkpoint': '../checkpoints/kitti/retinanet/r50/latest.pth'},
                 'retinanet_r101': {'config_file': '', 'checkpoint': ''},
                 'ssd512': {'config_file': '', 'checkpoint': ''},
                 'yolox': {'config_file': '', 'checkpoint': ''},
                 'cornernet': {'config_file': '', 'checkpoint': ''},
                 'centernet': {'config_file': '', 'checkpoint': ''},
                 'faster_rcnn_bdd':
                     {'config_file': '../configs/bdd100k/faster_rcnn_r50_fpn_1x_det_bdd100k.py',
                      'checkpoint': '../checkpoints/bdd100k/faster_rcnn_r50_fpn_1x_det_bdd100k.pth'},
                 'faster_rcnn_cityscapes':
                     {'config_file': '../configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py',
                      'checkpoint': '../checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth'}
                 }


def kitti_evaluate(detector='faster_rcnn'):
    model = init_detector(model_configs[detector]['config_file'],
                          model_configs[detector]['checkpoint'], device='cuda:0')
    hook_fpn = Hook(model.rpn_head.rpn_cls)
    hook = Hook(model.roi_head.bbox_head.shared_fcs._modules['1'])
    hook2 = Hook(model.roi_head.bbox_head.fc_cls)
    hook_reg = Hook(model.roi_head.bbox_head.fc_reg)
    dates = sorted(os.listdir(kitti_root_dir))
    for date in dates:
        date_path = os.path.join(kitti_root_dir, date)
        drives = sorted(os.listdir(date_path))
        for drive in drives:
            infer_results = {'classes': {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
                                         'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7},
                             'results': []}
            drive_path = os.path.join(date_path, drive)
            image_select = os.path.join(drive_path, 'image_02/data')
            images = sorted(os.listdir(image_select))
            progress = ProgressBar(len(images), fmt=ProgressBar.FULL)
            print('\nDate:', date, 'Drive:', drive)
            for i, image in enumerate(images):
                infer_frame = {'frame_id': i, 'objs': []}
                image_path = os.path.join(image_select, image)
                result = inference_detector(model, image_path)
                hook.calc()
                soft = torch.nn.Softmax(dim=1)
                temp = soft(hook2.output)
                temp = temp.data.cpu().numpy()
                for cls in result:
                    infer_frame['objs'].append(cls.tolist())
                infer_results['results'].append(infer_frame)
                progress.current += 1
                progress()
                model.show_result(image_path, result, font_size=10, score_thr=0.7,
                                  out_file=os.path.join('../results', date, drive, image))
            with open(os.path.join('../results', date, '{}.json'.format(drive)), 'w') as f:
                json.dump(infer_results, f)


def vkitti_evaluate(detector='faster_rcnn'):
    model = init_detector(model_configs[detector]['config_file'],
                          model_configs[detector]['checkpoint'], device='cuda:0')

    scenes = sorted(os.listdir(vkitti_root_dir))
    for scene in scenes:
        scene_path = os.path.join(vkitti_root_dir, scene)
        variations = sorted(os.listdir(scene_path))
        progress = ProgressBar(100, fmt=ProgressBar.FULL)
        count = 0
        print('\nScene:', scene)
        for variation in variations:
            variation_path = os.path.join(scene_path, variation)
            image_select = os.path.join(variation_path, 'frames/rgb/Camera_0')
            images = sorted(os.listdir(image_select))
            for image in images:
                image_path = os.path.join(image_select, image)
                result = inference_detector(model, image_path)
                count += 1
                progress.current = count * 100 / (len(variations) * len(images))
                progress()
                model.show_result(image_path, result, font_size=10, score_thr=0.7,
                                  out_file=os.path.join('../results', scene, variation, image))


if __name__ == '__main__':
    kitti_evaluate('faster_rcnn')
