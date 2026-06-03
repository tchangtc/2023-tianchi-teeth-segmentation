from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class smp_unet(UnetDecoder):
	def __init__(self, **kwargs):
		super(smp_unet, self).__init__(
			**kwargs)

	def forward(self, encoder):
		feature = encoder[::-1]  # reverse channels to start from head of encoder
		head = feature[0]
		skip = feature[1:] + [None]
		d = self.center(head)

		decoder = []
		for i, decoder_block in enumerate(self.blocks):
			# print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
			# print(decoder_block.conv1[0])
			# print('')
			s = skip[i]
			d = decoder_block(d, s)
			decoder.append(d)

		last  = d
		return last, decoder