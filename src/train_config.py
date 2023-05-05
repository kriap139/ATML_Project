# The new config inherits a base config to highlight the necessary modification
_base_ = '../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# Change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2), mask_head=dict(num_classes=2)
    )
)

# Modify dataset related settings
data_root = 'data/dataset/v3-87/'

metainfo = {
    'classes': ('Valve_Lever_Brown', 'Valve_Lever_Brown_Mark'),
    'palette': [
         (217, 125, 88), # (217, 125, 88), (88, 125, 217)
         (168, 140, 25) # (168, 140, 25), (25,140,168)
    ]
}

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/ann.json',
        data_prefix=dict(img='train/')
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/ann.json',
        data_prefix=dict(img='val/')
    )
)

test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(
  ann_file=data_root + 'val/ann.json'
)

test_evaluator = val_evaluator

# Use a pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
resume = False # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.

# train config
train_cfg = dict(
  max_epochs=400, 
  val_interval=1 # Validation intervals. Set to 1 to run validation every epoch.
)

# Optimizer Configuration
optim_wrapper = dict(
  type='OptimWrapper',
  optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# Logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook')
    ]
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

checkpoint_config = dict(interval=1, max_keep_ckpts=3, by_epoch=True, priority='VERY_HIGH')
