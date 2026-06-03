import os
import torch
import random
import numpy as np
# from li .metrics import *
from Config import CFG
# from .Libs.metrics import *

def time_to_str(t, mode='min'):
    if mode == 'min':
        t  = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode=='sec':
        t   = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def scheduler(epoch):
    #return start_lr
    # num_epoch=6       # EffNet
    num_epoch = CFG.epochs / 3 * 2      # NextViT
    start_lr = CFG.start_lr
    min_lr = CFG.min_lr
    
    lr = (num_epoch-epoch)/num_epoch * (start_lr-min_lr) + min_lr
    lr = max(min_lr,lr)
    return lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def metric_to_text(ink, label, mask):
	text = []

	p = ink.reshape(-1)
	t = label.reshape(-1)
	pos = np.log(np.clip(p,1e-7,1))
	neg = np.log(np.clip(1-p,1e-7,1))
	bce = -(t*pos +(1-t)*neg).mean()
	text.append(f'bce={bce:0.5f}')


	mask_sum = mask.sum()
	#print(f'{threshold:0.1f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text.append('p_sum  th   prec   recall   fpr   dice   score')
	text.append('-----------------------------------------------')
	for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		p = ink.reshape(-1)
		t = label.reshape(-1)
		p = (p > threshold).astype(np.float32)
		t = (t > 0.5).astype(np.float32)

		tp = p * t
		precision = tp.sum() / (p.sum() + 0.0001)
		recall = tp.sum() / t.sum()

		fp = p * (1 - t)
		fpr = fp.sum() / (1 - t).sum()

		beta = 0.5
		#  0.2*1/recall + 0.8*1/prec
		score = beta * beta / (1 + beta * beta) * 1 / recall + 1 / (1 + beta * beta) * 1 / precision
		score = 1 / score

		dice = 2 * tp.sum() / (p.sum() + t.sum())
		p_sum = p.sum()/mask_sum

		# print(fold, threshold, precision, recall, fpr,  score)
		text.append( f'{p_sum:0.2f}, {threshold:0.2f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text = '\n'.join(text)
	return text





class AverageMeter(object):

    def __init__(self):
        
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_all_random_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode='train'):
    set_all_random_seed(cfg.seed, True)
    # set_env_name()
    # set_dataset_path(cfg)
    if mode == 'train':
        make_dirs(cfg)


tensor_key = ['image', 'mask']
def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_key:
            v = torch.stack(v, 0)
        d[k] = v
    return d


def image_to_tensor(image, mode='bgr'): #image mode
	if mode=='bgr':
		image = image[:,:,::-1]
	x = image
	x = x.transpose(2,0,1)
	x = np.ascontiguousarray(x)
	x = torch.tensor(x, dtype=torch.float)
	return x

def mask_to_tensor(mask):
	x = torch.tensor(mask, dtype=torch.float)
	x = x.unsqueeze(0)
	return x

import torch.nn as nn

class FocalLoss(nn.Module):
 
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha=alpha
    def forward(self, input, target):
        logp = self.bce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        loss = self.alpha*loss
        return loss.mean()
    

# tensor_key = ['image', 'mask']
# def null_collate(batch):
#     d = {}
#     key = batch[0].keys()
#     for k in key:
#         v = [b[k] for b in batch]
#         if k in tensor_key:
#             v = torch.stack(v)
#         d[k] = v
#     d['image'] = d['image'].unsqueeze(1)
#     d['mask']  = d['mask'].unsqueeze(1)
#     return d

# tensor_list = ['mask', 'image']
# def null_collate(batch):
#     d = {}
#     key = batch[0].keys()
#     for k in key:
#         v = [b[k] for b in batch]
#         if k in tensor_list:
#             v = torch.stack(v)
#         d[k] = v
#     d['mask'] = d['mask'].unsqueeze(1)
#    # d['organ'] = d['organ'].reshape(-1)
#     return d


# tensor_key = ['image', 'cancer']
# def null_collate(batch):
#     d = {}
#     key = batch[0].keys()
#     for k in key:
#         v = [b[k] for b in batch]
#         if k in tensor_key:
#             v = torch.stack(v,0)
#         d[k] = v
#     d['image']= d['image'].unsqueeze(1)
#     d['cancer']= d['cancer'].reshape(-1)
#     return d


# def null_collate(batch):
#     batch_size = len(batch)
#     d = {}
#     key = batch[0].keys()
#     for k in key:
#         d[k] = [b[k] for b in batch]
#     d['label'] = torch.LongTensor(d['label'])
#     return d


