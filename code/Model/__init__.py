from .model import build_modelv1
from .model_segformer_mit_for_train import build_modelv2
from .model_pvt_v2_unet_for_train import build_modelv3
from .model_coat_daformer_for_train import build_modelv4
from .model_dualvit_daformer_for_train import build_modelv5
from .model_convnext_smp_unet_for_train import build_modelv6
from .model_pvt2b3meanpool_for_train import build_modelv7


def build_model(args):
    if args.model == 'SwinV1-Small':
        return build_modelv1(args)
    elif args.model == 'Segformer-b3':
        return build_modelv2(args)
    elif args.model == 'pvt-unet':
        return build_modelv3(args)
    elif args.model == 'coat-daformer':
        return build_modelv4(args)
    elif args.model == 'dual-daformer':
        return build_modelv5(args)
    elif args.model == 'convnext-smpunet':
        return build_modelv6(args)
    elif args.model == 'pvt-unet-mp':
        return build_modelv7(args)
    else:
        raise ValueError('Please select a supported model version')
