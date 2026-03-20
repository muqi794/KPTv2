_base_ = ['../_base_/default_runtime.py',
          '../_base_/tests/segmentation.py']


batch_size = 1 
mix_prob = 0.0
empty_cache = True
enable_amp = True 

# 2. 模型显存优化（4卡必须开，避免显存溢出）
model = dict(
    type="ptv2m2",
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
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    enable_checkpoint=True,  # 关键：开启梯度检查点，节省30%显存（4卡必开）
    unpool_backend="interp",
)


epoch = 20
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.05)  # 4卡适配的lr
scheduler = dict(
    type='MultiStepLR', 
    milestones=[12,16],  
    gamma=0.1
)

# ===================== 原配置保留（仅修改SphereCrop点数） =====================
dataset_type = "SemanticKITTIDataset"
data_root = "data/semantic_kitti/dataset"
ignore_label = 255
names = ["car", "bicycle", "motorcycle", "truck", "other-vehicle",
         "person", "bicyclist", "motorcyclist", "road", "parking",
         "sidewalk", "other-ground", "building", "fence", "vegetation",
         "trunk", "terrain", "pole", "traffic-sign"]

learning_map = {
    0: ignore_label, 1: ignore_label, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4,
    30: 5, 31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12, 51: 13, 52: ignore_label,
    60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18, 99: ignore_label, 252: 0, 253: 6,
    254: 5, 255: 7, 256: 4, 257: 4, 258: 3, 259: 4
}

# 4. 数据预处理：降低点云数量（4卡显存优化）+ 移除冗余batch_size
data = dict(
    num_classes=19,
    ignore_label=ignore_label,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        learning_map=learning_map,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type="PointClip", point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="Voxelize", voxel_size=0.05, hash_type='fnv', mode='train',
                 keys=("coord", "strength", "label"), return_discrete_coord=True),
            #dict(type="SphereCrop", point_max=100000, mode='random'),  # 关键：4卡显存优化，降到4万点
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=("coord", "strength"))
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        learning_map=learning_map,
        transform=[
            dict(type="PointClip", point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type="Voxelize", voxel_size=0.05, hash_type='fnv', mode='train',
                 keys=("coord", "strength", "label"), return_discrete_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=("coord", "strength"))
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        learning_map=learning_map,
        transform=[
            dict(type="PointClip", point_cloud_range=(-80, -80, -3, 80, 80, 1)),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="Voxelize",
                          voxel_size=0.05,
                          hash_type="fnv",
                          mode="test",
                          return_discrete_coord=True,
                          keys=("coord", "strength")
                          ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "discrete_coord", "index"), feat_keys=("coord", "strength"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis='z', center=[0, 0, 0], p=1, unit='pi')],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis='z', center=[0, 0, 0], p=1, unit='pi')],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis='z', center=[0, 0, 0], p=1, unit='pi')],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis='z', center=[0, 0, 0], p=1, unit='pi')]
            ]
        )
    ),
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=data["ignore_label"])
]