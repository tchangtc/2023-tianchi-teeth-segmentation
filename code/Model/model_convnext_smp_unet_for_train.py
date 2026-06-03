import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.convnext import convnext_large_384_in22ft1k

from .smp_unet import *
import segmentation_models_pytorch as smp
criterion_DiceLoss = smp.losses.DiceLoss(mode='binary')
#################################################################
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L15
class LayerNorm2d(nn.Module):
	def __init__(self, dim, eps=1e-6):
		super().__init__()
		self.dim = dim
		self.weight = nn.Parameter(torch.ones(dim))
		self.bias = nn.Parameter(torch.zeros(dim))
		self.eps = eps

	def forward(self, x):
		batch_size, C, H, W = x.shape
		# assert C==self.dim, 'C=%d, self.dim=%d'%(C,self.dim)
		# print('C=%d, self.dim=%d'%(C,self.dim))

		u = x.mean(1, keepdim=True)
		s = (x - u).pow(2).mean(1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.eps)
		x = self.weight[:, None, None] * x + self.bias[:, None, None]
		return x
# ---------------------------------------------



class RGB(nn.Module):
	# /segmentation_models_pytorch/encoders/efficientnet.py
	IMAGE_RGB_MEAN = [0.5, 0.5, 0.5]  # [0.485, 0.456, 0.406]
	IMAGE_RGB_STD = [0.5, 0.5, 0.5]  # [0.229, 0.224, 0.225]

	def __init__(self, ):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
		self.register_buffer('std', torch.ones(1, 3, 1, 1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

	def forward(self, x):
		x = (x - self.mean) / self.std
		return x


class Net(nn.Module):

	def load_pretrain(self, ):
		pass

	def __init__(self,
	      		 args,
				 encoder=convnext_large_384_in22ft1k,
				 decoder=smp_unet,
	             encoder_cfg=dict(),
	             decoder_cfg=dict(),
				 ):
		super(Net, self).__init__()
		pretrained = encoder_cfg.get('pretrained', True)
		n_blocks = decoder_cfg.get('n_blocks',5)
		decoder_dim = [256, 128, 64, 32, 16][:n_blocks]

		self.output_type = ['inference', 'loss']
		self.rgb = RGB()
		self.args = args
		conv_dim = 32
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
			LayerNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
			LayerNorm2d(conv_dim),
			nn.ReLU(inplace=True),
		)


		self.encoder = encoder(pretrained=pretrained)
		if encoder == convnext_large_384_in22ft1k: encoder_dim = [192,384,768,1536]

		self.decoder = decoder(
			encoder_channels=[0, conv_dim] + encoder_dim,
			decoder_channels=decoder_dim,
			n_blocks=n_blocks,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)

		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)
		)

		self.aux = nn.ModuleList([
			nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(encoder_dim))
		])

		# self.aux_de = nn.ModuleList([
		# 	nn.Conv2d(decoder_dim[i], 1, kernel_size=1, padding=0) for i in range(len(decoder_dim))
		# ])
	


	def forward(self, batch):

		x = batch['image']
		x = self.rgb(x)
		B, C, H, W = x.shape

		conv = self.conv(x)
		##print('conv', conv.shape)

		# ---------------------------------
		#encoder = self.encoder.forward_features(x)
		if 1:
			e = self.encoder
			x = e.stem(x)

			encoder = []
			for i, blk in enumerate(e.stages):
				x = blk(x)
				encoder.append(x)
		##print('encoder', [f.shape for f in encoder])

		# ---------------------------------

		last, decoder = self.decoder([conv] + encoder)
		##print('decoder',[f.shape for f in decoder])
		# ---------------------------------------------------------

		logit = self.logit(last)
		logit = F.dropout(logit, p=self.args.drop_rate, training=self.training)
		##print(logit.shape)
		# logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
		# output = {}
		# probability_from_logit = torch.sigmoid(logit)
		# output['probability_from_logit'] = probability_from_logit
		# output['probability'] = probability_from_logit

		# return output


		output = {}
		if 'loss' in self.output_type:
			# output['loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask']) * 0.3 + criterion_dice_loss(logit, batch['mask']) * 0.7
			output['loss'] = self.args.Dice_loss_weight * criterion_DiceLoss(logit, batch['mask']) + self.args.BCE_loss_weight * F.binary_cross_entropy_with_logits(logit,batch['mask'])

			
			for i in range(len(self.aux)):
				output['aux%d_loss' % i] = criterion_aux_loss(self.aux[i](encoder[i]), batch['mask'])

			# for j in range(len(self.aux_de)):
			# 	output['aux_de%d_loss' % j] = criterion_aux_loss(self.aux_de[j](decoder[j]), batch['mask'])
		
		if 'inference' in self.output_type:
			#probability_from_logit = torch.softmax(logit,1)
			probability_from_logit = torch.sigmoid(logit)
			output['probability_from_logit'] = probability_from_logit
			output['probability'] = probability_from_logit
		
		return output

def build_modelv6(args):
	net = Net(args)
	return net


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


# http://jck.bio/pytorch_onehot/
num_organ=5
def criterion_multi_binary_cross_entropy(logit, mask, organ):
	logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
	batch_size, C, H, W = logit.shape
	
	label = mask.long() * organ.reshape(batch_size, 1, 1, 1)
	onehot = torch.zeros((batch_size, num_organ + 1, H, W)).to(mask)
	onehot = onehot.scatter(1, label, 1)
	# onehot[:,0] = 1-onehot[:,0]
	
	loss = F.binary_cross_entropy_with_logits(logit, onehot)
	return loss


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


def run_check_net():
	batch_size = 2
	image_size = 512 #800
	
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# ############################## main ##################################

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '2'
	run_check_net()
