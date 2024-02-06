_base_ = '../../_base_/default_runtime.py'

pretrained_model="openai/clip-vit-base-patch32"
model = dict(
    type='CLIPSimilarity_split', 
    visual_encoder=dict(type='VITCLIPPretrained_STAN', pretrained_model=pretrained_model),
    text_encoder=dict(type='CLIPTextPretrained', pretrained_model=pretrained_model),
    to_float32=True,
    frozen_layers=False,    
    class_path = 'tools/data/hmdb51/label_map.txt',
    task="recognition",
    data_preprocessor=dict(
        type='MultiModalDataPreprocessor',
        preprocessors=dict(
            imgs=dict(
                type='ActionDataPreprocessor',
                mean=[122.771, 116.746, 104.093],
                std=[68.500, 66.632, 70.323],
                format_shape='NCHW'),
            text=dict(type='ActionDataPreprocessor', to_float32=False))),
    tau = 0.01,
    loss = dict(type='CrossEntropyLoss'),
    adapter=None)

load_from = None #Path to the post-pretrained ckpt
classes = 51
data_root = '/{your path}/mmaction2/data/hmdb51/'
dataset_type = 'VideoDataset'
file_client_args= dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize', length=32),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=12, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='CLIPTokenize', length=32),
    dict(type='PackActionInputs', collect_keys=('imgs', 'text'))
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file = 'hmdb51_train_split_1_videos.txt',
        num_classes=classes,
        data_root= data_root,
        data_prefix=dict(video='videos'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='hmdb51_val_split_1_videos.txt', 
        num_classes=classes,
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file='hmdb51_val_split_1_videos.txt', 
        num_classes=classes,
        data_root=data_root,
        data_prefix=dict(video='videos'),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='ZeroShotAccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optional: wandb integration
#visualizer=dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend', init_kwargs=dict(project='STAN'))])

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=4.5,
        eta_min=0,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-06,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0., bias_decay_mult=0.,
        custom_keys={
            'STAN': dict(lr_mult=10.),
    }),
    clip_grad=dict(max_norm=5, norm_type=2)
)

default_hooks = dict(
    checkpoint=dict(type='printBest_CheckpointHook', interval=-1, save_best='auto', rule='greater'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
