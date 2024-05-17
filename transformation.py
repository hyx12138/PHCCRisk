import numpy as np
from monai import transforms
import math

def ResNet3D_transformations(flag):
    train_transforms = [
        transforms.LoadImaged(keys=[f'{flag}_image', f'{flag}_seg'], dtype=np.float32, image_only=True),
        transforms.EnsureChannelFirstd(keys=[f'{flag}_image', f'{flag}_seg'], channel_dim = 'no_channel'),
        transforms.ScaleIntensityRangePercentilesd(keys=[f'{flag}_image'], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, channel_wise=True),
        transforms.Spacingd(keys=[f'{flag}_image', f'{flag}_seg'],
                pixdim=(1.25,1.25,2.5),
                mode=('trilinear', 'nearest')),
        transforms.CropForegroundd(keys=[f'{flag}_image', f'{flag}_seg'], source_key=f'{flag}_seg', select_fn=lambda x: x > 0, margin=10),
        transforms.Resized(keys=[f'{flag}_image', f'{flag}_seg'], spatial_size=(48,48,32), mode=('trilinear', 'nearest')),
        transforms.RandAdjustContrastd(keys=[f'{flag}_image'], prob=0.5),
        transforms.RandZoomd(keys=[f'{flag}_image', f'{flag}_seg'], min_zoom=1.0, max_zoom = 1.2, keep_size=True, mode="trilinear", prob=1.0),
        transforms.RandGaussianSmoothd(keys=[f'{flag}_image'], prob=0.5),
        transforms.RandFlipd(keys=[f'{flag}_image'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=[f'{flag}_image'], prob=0.5, spatial_axis=1),
        transforms.RandRotated(keys=[f'{flag}_image'], range_x=math.pi/9, keep_size=True, prob=0.5),
        transforms.RandCoarseDropoutd(keys=[f'{flag}_image'], holes=1, spatial_size=2, fill_value=0, max_spatial_size=8, prob=0.5),
        transforms.NormalizeIntensityd(keys=[f'{flag}_image'], nonzero=True, channel_wise=True),
        ]

    val_transforms = [
        transforms.LoadImaged(keys=[f'{flag}_image', f'{flag}_seg'], dtype=np.float32, image_only=True),
        transforms.EnsureChannelFirstd(keys=[f'{flag}_image', f'{flag}_seg'], channel_dim = 'no_channel'),
        transforms.ScaleIntensityRangePercentilesd(keys=[f'{flag}_image'], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, channel_wise=True),
        transforms.Spacingd(keys=[f'{flag}_image', f'{flag}_seg'],
                pixdim=(1.25,1.25,2.5),
                mode=('trilinear', 'nearest')),
        transforms.CropForegroundd(keys=[f'{flag}_image', f'{flag}_seg'], source_key=f'{flag}_seg', select_fn=lambda x: x > 0, margin=10),
        transforms.Resized(keys=[f'{flag}_image', f'{flag}_seg'], spatial_size=(48,48,32), mode=('trilinear', 'nearest')),
        transforms.NormalizeIntensityd(keys=[f'{flag}_image'], nonzero=True, channel_wise=True),
        ]

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)