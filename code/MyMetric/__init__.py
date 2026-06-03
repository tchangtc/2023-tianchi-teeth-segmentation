import torch
import torch.nn.functional as F

def criterion_dice_loss(logit, mask):
    smooth = 1e-7

    batch_size, C, H, W = logit.shape
    p = torch.sigmoid(logit).reshape(batch_size, -1)
    t = mask.reshape(batch_size, -1)

    intersection = (p * t).sum(-1)
    union = p.sum(-1) + t.sum(-1)
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    dice = dice.mean()
    
    return dice

def criterion_binary_cross_entropy(logit, mask):
    logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
    loss = F.binary_cross_entropy_with_logits(logit, mask)
    return loss


def criterion_cross_entropy(logit, mask, organ):
	logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
	batch_size, C, H, W = logit.shape
	
	label = mask.long() * organ.reshape(batch_size, 1, 1, 1)
	
	#https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
	log_softmax = F.log_softmax(logit,1)
	loss = F.nll_loss(log_softmax,label.squeeze(1))
	return loss


def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit, mask) * 0.3 + 0.7 * criterion_dice_loss(logit, mask)
	return loss