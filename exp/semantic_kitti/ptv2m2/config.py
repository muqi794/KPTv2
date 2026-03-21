weight = None
resume = False
evaluate = True
test_only = False
seed = 23334315
save_path = 'exp/semantic_kitti/ptv2m2'
num_worker = 32
batch_size = 1
batch_size_val = None
batch_size_test = 1
epoch = 20
eval_epoch = 20
save_freq = None
eval_metric = 'mIoU'
sync_bn = False
enable_amp = True
empty_cache = True
find_unused_parameters = False
max_batch_points = 100000000.0
mix_prob = 0.0
param_dicts = None
test = dict(type='SegmentationTest')
model = dict(
    type='ptv2m2',
    in_channels=4,
    num_classes=19,
    patch_embed_depth=2,
    patch_embed_channels=48,
    patch_embed_groups=6,
    patch_embed_neighbours=16,
    enc_depths=(2, 6, 2),
    enc_channels=(96, 192, 384),
    enc_groups=(12, 24, 48),
    enc_neighbours=(16, 16, 16),
    dec_depths=(1, 1, 1),
    dec_channels=(48, 96, 192),
    dec_groups=(6, 12, 24),
    dec_neighbours=(16, 16, 16),
    grid_sizes=(0.1, 0.2, 0.4),
    attn_qkv_bias=True,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.0,
    drop_path_rate=0.2,
    enable_checkpoint=True,
    unpool_backend='interp')
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.05)
scheduler = dict(type='MultiStepLR', milestones=[12, 16], gamma=0.1)
dataset_type = 'SemanticKITTIDataset'
data_root = 'data/semantic_kitti/dataset'
ignore_label = 255
names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
    'traffic-sign'
]
learning_map = dict({
    0: 255,
    1: 255,
    10: 0,
    11: 1,
    13: 4,
    15: 2,
    16: 4,
    18: 3,
    20: 4,
    30: 5,
    31: 6,
    32: 7,
    40: 8,
    44: 9,
    48: 10,
    49: 11,
    50: 12,
    51: 13,
    52: 255,
    60: 8,
    70: 14,
    71: 15,
    72: 16,
    80: 17,
    81: 18,
    99: 255,
    252: 0,
    253: 6,
    254: 5,
    255: 7,
    256: 4,
    257: 4,
    258: 3,
    259: 4
})
data = dict(
    num_classes=19,
    ignore_label=255,
    names=[
        'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
        'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        'pole', 'traffic-sign'
    ],
    train=dict(
        type='SemanticKITTIDataset',
        split='train',
        data_root='data/semantic_kitti/dataset',
        learning_map=dict({
            0: 255,
            1: 255,
            10: 0,
            11: 1,
            13: 4,
            15: 2,
            16: 4,
            18: 3,
            20: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            44: 9,
            48: 10,
            49: 11,
            50: 12,
            51: 13,
            52: 255,
            60: 8,
            70: 14,
            71: 15,
            72: 16,
            80: 17,
            81: 18,
            99: 255,
            252: 0,
            253: 6,
            254: 5,
            255: 7,
            256: 4,
            257: 4,
            258: 3,
            259: 4
        }),
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='PointClip', point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(
                type='Voxelize',
                voxel_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'label'),
                return_discrete_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'discrete_coord', 'label'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        loop=1),
    val=dict(
        type='SemanticKITTIDataset',
        split='val',
        data_root='data/semantic_kitti/dataset',
        learning_map=dict({
            0: 255,
            1: 255,
            10: 0,
            11: 1,
            13: 4,
            15: 2,
            16: 4,
            18: 3,
            20: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            44: 9,
            48: 10,
            49: 11,
            50: 12,
            51: 13,
            52: 255,
            60: 8,
            70: 14,
            71: 15,
            72: 16,
            80: 17,
            81: 18,
            99: 255,
            252: 0,
            253: 6,
            254: 5,
            255: 7,
            256: 4,
            257: 4,
            258: 3,
            259: 4
        }),
        transform=[
            dict(
                type='PointClip', point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(
                type='Voxelize',
                voxel_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'label'),
                return_discrete_coord=True),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'discrete_coord', 'label'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False),
    test=dict(
        type='SemanticKITTIDataset',
        split='val',
        data_root='data/semantic_kitti/dataset',
        learning_map=dict({
            0: 255,
            1: 255,
            10: 0,
            11: 1,
            13: 4,
            15: 2,
            16: 4,
            18: 3,
            20: 4,
            30: 5,
            31: 6,
            32: 7,
            40: 8,
            44: 9,
            48: 10,
            49: 11,
            50: 12,
            51: 13,
            52: 255,
            60: 8,
            70: 14,
            71: 15,
            72: 16,
            80: 17,
            81: 18,
            99: 255,
            252: 0,
            253: 6,
            254: 5,
            255: 7,
            256: 4,
            257: 4,
            258: 3,
            259: 4
        }),
        transform=[
            dict(
                type='PointClip', point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='Voxelize',
                voxel_size=0.05,
                hash_type='fnv',
                mode='test',
                return_discrete_coord=True,
                keys=('coord', 'strength')),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'discrete_coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1,
                'unit': 'pi'
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1,
                               'unit': 'pi'
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1,
                               'unit': 'pi'
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1,
                               'unit': 'pi'
                           }]])))
criteria = [dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=255)]
