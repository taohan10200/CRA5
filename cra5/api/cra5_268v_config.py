# dataset settings
dataset_type = 'era5_base_npy'


train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='RandomResizedCrop', size=224, backend='pillow'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    # dict(type='Collect', keys=['img', 'gt_label'])
    dict(type='PackNwpInputs')
]
val_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1), backend='pillow'),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img'])
    dict(type='PackNwpInputs')
]

test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1), backend='pillow'),
    # dict(type='CenterCrop', crop_size=224),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='ImageToTensor', keys=['img']),
    # dict(type='Collect', keys=['img'])
    dict(type='PackNwpInputs')
]


bucket_name =dict(bucket_name = 'era5_np_float32',
                  endpoint ='http://10.140.31.254'
                  )
local_root ='/nvme/hantao/era5_local/era5_np/' #  '/mnt/petrelfs/hantao/NWP/era5_local/era5_np128x256/' #
format='.npy' #'.nc'
vnames=dict(
    pressure=['z','q', 'u', 'v', 't', 'r','w'],
    single=['v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp', 'msl']) # 'tisr'




total_levels = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
 775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
 350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
 70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
 1.]

pressure_level = total_levels

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    # prefetch_factor=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        data_prefix=local_root,
        s3_bucket = bucket_name,
        format=format,
        vname=vnames,
        pressure_level = pressure_level,
        crop_size=(721, 1440),
        ori_size =(721, 1440),
        norm_type='channel',  # 'overall' or 'channel''
        is_norm=False,
        time_interval='1H',
        sequence_cfg = dict(input=[0], 
                            gt=[6], 
                            data_interval=6,
                        ),
                            #for training, we input the current hourly data to predict the next hourly data.
        year=dict(start='1998-05-04', periods=1, end='2017-12-31'),
        ),
    )
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    prefetch_factor=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        pipeline=val_pipeline,
        year=dict(start='2018-01-01', periods=1, end='2018-12-31'),
        data_prefix=local_root,
        s3_bucket = bucket_name,
        format=format,
        vname=vnames,
        pressure_level = pressure_level,
        crop_size=(721, 1440),
        ori_size = (721, 1440),
        time_interval='12H',
        sequence_cfg = dict(input=[0],
                            gt=[6,24,72,120],
                            data_interval=6, #unit hour
                            ),  # the data is 6-hour, so the gt is 1/3/5 day
        test_val = True,
        norm_type='channel',  # 'overall' or 'channel''
        is_norm=False,
        )
    )
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    prefetch_factor=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        year=dict(start='2018-01-01', periods=1, end='2018-12-31'),
        data_prefix=local_root,
        s3_bucket = bucket_name,
        format=format,
        vname=vnames,
        pressure_level = pressure_level,
        crop_size=(721, 1440),
        ori_size =(721, 1440),
        time_interval='12H',
        sequence_cfg = dict(input=[0],
                            gt=[24,72,120,240],
                           data_interval=6, #unit hour
                           ),  # the data is 6-hour, so the gt is 1/3/5 day
        test_val = True,
        norm_type='channel',
        is_norm=False,
        )
    )
# val_evaluator = dict(type='Accuracy', topk=(1, ))
val_evaluator = dict(
    type='Era5_RMSE',
    metric_name = ['WRMSE', 'MSE']
)

test_evaluator =dict(
    type='Era5_RMSE',
    metric_name = ['WRMSE', 'MSE']
)