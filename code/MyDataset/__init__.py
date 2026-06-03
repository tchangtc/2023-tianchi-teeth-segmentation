from .dataset import build

def build_dataset(image_set, args, fold):

    if args.dataset == 'Teeth-Seg':
        return build(image_set, args, fold)

    else:
        raise ValueError(f'dataset {args.dataset} not supported')
