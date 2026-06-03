import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .daformer import daformer_conv3x3, daformer_conv1x1
from .dualvit import dual_vit_b


#################################################################

class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

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
        # pretain = pretrain_dir + '/' + self.encoder.pretrain
        pretain = self.encoder.pretrain
        print('load %s' % pretain)
        state_dict = torch.load(pretain, map_location=lambda storage, loc: storage)  # True
        # print(self.encoder.load_state_dict(state_dict, strict=False))  # True

    def __init__(self,
                 args,
                 encoder=dual_vit_b,
                 decoder=daformer_conv1x1,
                #  decoder=daformer_conv3x3,
                 encoder_cfg=dict(),
                 decoder_cfg=dict(),
                 ):
        super(Net, self).__init__()
        decoder_dim = decoder_cfg.get('decoder_dim', 320)

        # ----
        self.output_type = ['inference', 'loss']
        self.rgb = RGB()
        self.args = args

        self.encoder = encoder(
            **encoder_cfg
        )
        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
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
        encoder = self.encoder(x)
        #print([f.shape for f in encoder])

        last, decoder = self.decoder(encoder)

        #---
        logit = self.logit(last)

        logit = F.dropout(logit, p=self.args.drop_rate, training=self.training)

        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
		##print('logit',logit.shape)
		
        output = {}
        if 'loss' in self.output_type:
            output['loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask']) * 0.3 + criterion_dice_loss(logit, batch['mask']) * 0.7
            
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


def build_modelv5(args):
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
    image_size = 768 #800

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


# main #################################################################
if __name__ == '__main__':
    run_check_net()