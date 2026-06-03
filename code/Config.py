import albumentations as A
from albumentations.pytorch import ToTensorV2

class CFG:

    save_root_dir = f'./result'
    # model = 'SwinV1-Small'
    # model = 'Segformer-b3'
    model = 'pvt-unet-mp'
    # model = 'coat-daformer'
    # model = 'dual-daformer'
    # model = 'convnext-smpunet'

    seed = 42

    num_fold = 5
    dataset = 'Teeth-Seg'
    start_lr = 1e-4
    min_lr = 1e-6

    epochs = 84
    initial_checkpoint = None

    # num_workers = 16
    num_workers = 8
    # num_workers = 0

    Dice_loss_weight = 0.7
    Focal_loss_weight = 0.3
    BCE_loss_weight = 0.3

    train_batch_size = 8
    val_batch_size = train_batch_size * 2

    gpu_id = '0'

    ce_weight = 0.5
    dice_weight = 0.5
    drop_rate = 0.4
    
    scale = 1.0

    in_chans = 3

    TTA = True

    valid_aug_list = [
        # A.Resize(size, size),
        # A.Normalize(
        #     mean= [0] * in_chans,
        #     std= [1] * in_chans
        # ),
        # ToTensorV2(transpose_mask=True),
    ]



    # ============== augmentation =============
    # augmentation V0

    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        # A.Resize(size, size),

        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(P=0.5),
        # A.RandomBrightnessContrast(p=0.75),
        # A.ShiftScaleRotate(p=0.75),

        # A.OneOf([
        #         A.GaussNoise(var_limit=[10, 50]),
        #         A.GaussianBlur(),
        #         A.MotionBlur(),
        #         ], p=0.5),

        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        # A.CoarseDropout(max_holes=4, max_width=int(8), max_height=int(16), 
        #                 mask_fill_value=0, p=0.5),

        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),


        # A.Normalize(
        #     mean= [0] * in_chans,
        #     std= [1] * in_chans
        # ),
        # ToTensorV2(transpose_mask=True),
    ]

