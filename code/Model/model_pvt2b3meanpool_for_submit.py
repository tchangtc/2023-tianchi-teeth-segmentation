
import sys
import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .pvt_v2_2 import pvt_v2_b4
from .daformer import DaformerDecoder
from einops import rearrange, reduce, repeat
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import *

# from Libs.libs import FocalLoss
# focal_loss = FocalLoss()

#######################################################################################

class SmpUnetDecoder(nn.Module):
	def __init__(self,
	             in_channel,
	             skip_channel,
	             out_channel,
	             ):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel, ] + out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
			for i, s, o in zip(i_channel, s_channel, o_channel)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			s = skip[i]
			d = block(d, s)
			decode.append(d)

		last = d
		return last, decode


#######################################################################################

class Net(nn.Module):

	def load_pretrain(self, ):
		# pass
		# # pretain = pretrain_dir + '/' + self.encoder.pretrain
		# pdb.set_trace()
		pretrain = self.encoder.pretrain
		print('load %s' % pretrain)
		state_dict = torch.load(pretrain, map_location=lambda storage, loc: storage)  # True
		# state_dict['patch_embed1.proj.weight'] = []
		print(self.encoder.load_state_dict(state_dict, strict=False), 'Load_pretrain Done!')  # True

	def __init__(self,):
		super().__init__()

		# --------------------------------
		# self.d = 5
		self.d = 1

		encoder_dim = [64, 128, 320, 512]
		decoder_dim = 256

		self.encoder = pvt_v2_b4()
		# self.load_pretrain()
		self.encoder.patch_embed1.proj = nn.Conv2d(self.d, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
		# self.load_pretrain()

		# pdb.set_trace()

		self.decoder = DaformerDecoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
			fuse='conv1x1',
			dilation=None,  # [1, 6, 12, 18],
		)
		self.logit =  nn.Conv2d(decoder_dim, 1, kernel_size=1)

		self.aux = nn.ModuleList([
			nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
		])

		# self.aux_de = nn.ModuleList([
		# 	nn.Conv2d(decoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(decoder_dim))
		# ])

	def forward(self, batch):

		# pdb.set_trace()

		# v = batch['image']

		v= batch

		B, C, H, W = v.shape	# [bs, 32, 256, 256]
			# v[:, i:i + self.d] for i in range(0,C-self.d+1,2)
		vv = [
			v[:, i:i + self.d] for i in range(0,C-self.d+1,1)
		]

		# pdb.set_trace()			# vv[i] = [3, 5, 256, 256]
		K = len(vv)				# K = 14
		x = torch.cat(vv, 0)	# x = [14*3, 5, 256, 256]

		# ---------------------------------
		# pdb.set_trace()
		encoder = self.encoder(x)
		for i in range(len(encoder)):
			e = encoder[i]
			_, c, h, w = e.shape
			e = rearrange(e, '(K B) c h w -> K B c h w', K=K, B=B, h=h, w=w)  #
			m1 = e.mean(0)
			encoder[i] = m1
		##[print('encoder',i,f.shape) for i,f in enumerate(encoder)]


		# pdb.set_trace()
		last, decoder = self.decoder(encoder)
		logit = self.logit(last)

		logit = self.logit(last)
		# logit = F.dropout(logit, p=self.args.drop_rate, training=self.training)

		if logit.shape[2:]!=(H, W):
			logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)

		

		output = {}
		if 1:
			if logit.shape[2:]!=(H, W):
				logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
			output = torch.sigmoid(logit)

		return output

def criterion_dice_loss(logit, mask):
	smooth = 1e-5
	
	batch_size, C, H, W = logit.shape
	p = torch.sigmoid(logit).reshape(batch_size, -1)
	t = mask.reshape(batch_size, -1)
	
	intersection = (p * t).sum(-1)
	union = p.sum(-1) + t.sum(-1)
	dice = 1 - (2 * intersection + smooth) / (union + smooth)
	dice = dice.mean()
	return dice

# def criterion_aux_loss(logit, mask):
# 	mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest')
# 	loss = focal_loss(logit, mask) * 0.3 + 0.7 * criterion_dice_loss(logit, mask)
# 	return loss

def build_modelv7(args):
	net = Net(args)
	return net


def run_check_net():

	# pdb.set_trace()

	height, width = 256, 256
	depth = 3
	batch_size = 4

	# height, width = 256, 256
	# depth = 32
	# batch_size = 3

	# height,width = 1024, 1024
	# depth = 3
	# batch_size = 1

	batch = {
		'volume': torch.from_numpy(np.random.choice(256, (batch_size, depth, height, width))).cuda().float(),
		'label': torch.from_numpy(np.random.choice(2, (batch_size, 1, height, width))).cuda().float(),
	}
	#batch['label'] *=0
	net = Net().cuda()
	# net.load_pretrain()

	# pdb.set_trace()
	# print(net)
	while 1:
		with torch.no_grad():
			with torch.cuda.amp.autocast(enabled=True):
				print('running')
				output = net(batch)
				break

	# ---
	print('batch')
	for k, v in batch.items():
		print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print(f'{k:>32} : {v.shape} ')
	print('loss')
	for k, v in output.items():
		if 'loss' in k:
			print(f'{k:>32} : {v.item()} ')


# ###################################################### main #################################################################
if __name__ == '__main__':
	run_check_net()





