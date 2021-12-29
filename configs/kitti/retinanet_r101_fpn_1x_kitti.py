_base_ = './retinanet_r50_fpn_1x_kitti.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    bbox_head = dict(
        num_classes=8
    ))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=32)  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
load_from = 'checkpoints/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth'
