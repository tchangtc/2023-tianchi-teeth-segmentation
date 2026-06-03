import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.convnext import convnext_large_384_in22ft1k

from .smp_unet import *

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
				 encoder=convnext_large_384_in22ft1k,
				 decoder=smp_unet,
	             encoder_cfg=dict(),
	             decoder_cfg=dict(),
				 ):
		super(Net, self).__init__()
		pretrained = encoder_cfg.get('pretrained',False)
		n_blocks = decoder_cfg.get('n_blocks',5)
		decoder_dim = [256, 128, 64, 32, 16][:n_blocks]

		self.output_type = ['inference', 'loss']
		self.rgb = RGB()

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
		##print(logit.shape)

		output = {}
		probability_from_logit = torch.sigmoid(logit)
		# output['probability_from_logit'] = probability_from_logit
		output = probability_from_logit

		return output





def run_check_net():
	batch_size = 4
	image_size = 320

	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}

	net = Net().cuda()
	# torch.save({ 'state_dict': net.state_dict() },  'model.pth' )
	# exit(0)

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


# main #################################################################
if __name__ == '__main__':
	run_check_net()