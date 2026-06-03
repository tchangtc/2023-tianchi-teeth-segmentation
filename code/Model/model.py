import warnings

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

from .swin_v1_variable import LayerNorm2d, SwinTransformerV1
from .upernet import *

warnings.filterwarnings('ignore')
criterion_DiceLoss = smp.losses.DiceLoss(mode='binary')


########################################
# from loss.TILoss import TI_Loss
# criterion_TILoss = TI_Loss(dim=2, connectivity=4, )
########################################

class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406] #[0.5, 0.5, 0.5]
    IMAGE_RGB_STD  = [0.229, 0.224, 0.225] #[0.5, 0.5, 0.5]

    def __init__(self,):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1,3,1,1))
        self.register_buffer('std', torch.ones(1,3,1,1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x-self.mean)/self.std
        return x

############################################################
####### Configuration
############################################################
# pretrain_dir = '../input/swin-tiny-small-22k-pretrained/'
pretrain_dir = './pretrained/'
cfg = dict(
        #configs/_base_/models/upernet_swin.py
        basic = dict(
            swin=dict(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False
            ),
        ),

        #configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
        swin_tiny_patch4_window7_224_22k=dict(
            checkpoint = pretrain_dir+'swin_tiny_patch4_window7_224_22k.pth',
            swin = dict(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
            ),
            upernet=dict(
                in_channels=[96, 192, 384, 768],
            ),
        ),

        #/configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py
        swin_small_patch4_window7_224_22k=dict(
            checkpoint = pretrain_dir+'swin_small_patch4_window7_224_22k.pth',

            swin = dict(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False
            ),
            upernet=dict(
                in_channels=[96, 192, 384, 768],
            ),
        ),
    )


class Net(nn.Module):
    
    def load_pretrain(self, ):

        checkpoint = cfg[self.arch]['checkpoint']
        # debug
        # print('loading %s ...'%checkpoint)
     
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)['model']
        if 0:
            skip = ['relative_coords_table','relative_position_index']
            filtered={}
            for k,v in checkpoint.items():
                if any([s in k for s in skip ]): continue
                filtered[k]=v
            checkpoint = filtered
        print(self.encoder.load_state_dict(checkpoint,strict=False))  #True


    def __init__(self,
                 args,
                 encoder=SwinTransformerV1,
                 decoder=UPerDecoder,
                ):
        super(Net, self).__init__()
        self.output_type = ['inference', 'loss']
        self.args = args
        self.rgb = RGB()
        self.arch = 'swin_small_patch4_window7_224_22k'

        # self.encoder = SwinTransformerV1(
        #     ** {**cfg['basic']['swin'], **cfg[self.arch]['swin'],
        #         **{'out_norm' : LayerNorm2d} }
        # )
        self.encoder = encoder(
            **{**cfg['basic']['swin'], **cfg[self.arch]['swin'],
                **{'out_norm' : LayerNorm2d} }
        )

        encoder_dim =cfg[self.arch]['upernet']['in_channels']
        #[96, 192, 384, 768]

        # self.decoder = UPerDecoder(
        #     in_dim=encoder_dim,
        #     ppm_pool_scale=[1, 2, 3, 6],
        #     ppm_dim=512,
        #     fpn_out_dim=256
        # )
        self.decoder = decoder(
            in_dim=encoder_dim,
            ppm_pool_scale=[1, 2, 3, 6],
            ppm_dim=512,
            fpn_out_dim=256,
        )

        self.logit = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.aux = nn.ModuleList([
            nn.Conv2d(256, 1, kernel_size=1, padding=0) for i in range(4)
            # nn.Conv2d(256, 1, kernel_size=1, padding=0) for i in range(4)
        ])



    def forward(self, batch):
        x = batch['image']
        B,C,H,W = x.shape
        x = self.rgb(x)
        encoder = self.encoder(x)
        last, decoder = self.decoder(encoder)

        last = F.dropout(last, p=self.args.drop_rate, training=self.training)
        logit = self.logit(last)
        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)


        # # for submit
        # output = {}	
        # probability_from_logit = torch.sigmoid(logit)
        # output['probability'] = probability_from_logit
		############################################################################################

        # # for training
        output = {}


        if 'loss' in self.output_type:
            # output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
            output['loss'] = self.args.Dice_loss_weight * criterion_DiceLoss(logit, batch['mask']) + self.args.BCE_loss_weight * F.binary_cross_entropy_with_logits(logit,batch['mask'])
            for i in range(4):
                output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'])

        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit)

        return output

def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit,mask) * 0.3 \
        + 0.7 * criterion_DiceLoss(logit, mask)
    # 0.5 * criterion_DiceLoss(logit, mask) + 0.5 * F.binary_cross_entropy_with_logits(logit, mask)
    return loss

def build_modelv1(args):
    net = Net(args)
    return net

# ############################################################
# ####### Upnet + Net + MLP
# ############################################################
# def run_check_net():
#     batch_size = 2
#     image_size = 512

#     #---
#     batch = {
#         'image' : torch.from_numpy( np.random.uniform(-1,1,(batch_size,3,image_size,image_size)) ).float(),
#         'mask'  : torch.from_numpy( np.random.choice(2,(batch_size,1,image_size,image_size)) ).float(),
#         'organ' : torch.from_numpy( np.random.choice(5,(batch_size)) ).long(),
#     }
#     batch = {k:v.cuda() for k,v in batch.items()}

#     net = Net().cuda()
#     net.load_pretrain()

#     with torch.no_grad():
#         with torch.cuda.amp.autocast(enabled=True):
#             output = net(batch)

#     print('batch')
#     for k,v in batch.items():
#         print('%32s :'%k, v.shape)

#     print('output')
#     for k,v in output.items():
#         if 'loss' not in k:
#             print('%32s :'%k, v.shape)
#     for k,v in output.items():
#         if 'loss' in k:
#             print('%32s :'%k, v.item())
            
            
# # main #################################################################
# if __name__ == '__main__':
# 	run_check_net()
