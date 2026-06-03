import os
import pdb
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Config import CFG
import torch.cuda.amp as amp
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from Libs.libs import time_to_str, image_to_tensor, mask_to_tensor

from Model.model_dualvit_daformer_for_submit import Net as DualDaFormer
from Model.model_pvt_v2_unet_for_submit import Net as PVTUnet
from Model.model_segformer_mit_for_submit import Net as Segformer
from Model.model_coat_daformer_for_submit import Net as CoatDaformer
from Model.model_swin_for_submit import Net as SwinSmall
from Model.model_convnext_smp_unet_for_submit import Net as ConvNextUNet
from Model.model_pvt2b3meanpool_for_submit import Net as PVT2B4Daformer 


def TTA(x:torch.Tensor,model:nn.Module):
    #x.shape=(batch,c,h,w)
    # pdb.set_trace()
    if CFG.TTA:
        x = x['image']
        y = x
        shape=x.shape

        x=[x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]

        x=torch.cat(x,dim=0)
        
        x=model(x)
        x=torch.sigmoid(x)
        
        # pdb.set_trace()
        x=x.reshape(4,shape[0],*shape[2:])
        
        x=[torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
        
        # pdb.set_trace()

        x=torch.stack(x,dim=0)

        y = x.mean(0, False)
        return y
    
    else :
        x=model(x)
        x=torch.sigmoid(x)
        return x
# -------------------------- Dataset ------------------------------
class BaseDataset(Dataset):
    def transforms_v0(self,):
        pass

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.length = len(self.df)

    def __len__(self,):
        return self.length

    def __str__(self) -> str:
        num_image = len(self.df)
        string = ''
        string += f'\tnum_image = {num_image}\n'
        return string

    def __getitem__(self, index):
        dd = self.df.iloc[index]
        image = cv2.imread(f'{dd.img_path}', cv2.IMREAD_COLOR)

        # pdb.set_trace()
        pad_h = (640 - image.shape[0]) // 2
        pad_w = (640 - image.shape[1]) // 2

        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        # mask  = np.pad(mask,  ((pad_h, pad_h), (pad_w, pad_w)))
        
        # image = cv2.resize(image, dsize=None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        # mask  = cv2.resize(mask,  dsize=None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, dsize=None, fx=CFG.scale, fy=CFG.scale, interpolation=cv2.INTER_LINEAR)
        # mask  = cv2.resize(mask,  dsize=None, fx=CFG.scale, fy=CFG.scale, interpolation=cv2.INTER_LINEAR)

        # image = cv2.resize(image, dsize=None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        # image = cv2.resize(image, dsize=None, fx=CFG.scale, fy=CFG.scale, interpolation=cv2.INTER_LINEAR)
        # mask  = cv2.resize(mask,  dsize=None, fx=CFG.scale, fy=CFG.scale, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        
        # Normalization
        image = image / 255.0

        rr = {}
        rr['index'] = index
        rr['img_name'] = dd.img_name
        rr['image'] = image_to_tensor(image)
        return rr

# --------------- Model ------------------
# Uncomment the model(s) you want to ensemble.
# Place your SWA checkpoint files under ./checkpoints/<model_name>/
model = [
    # --- Swin-Small (image_size=768) ---
    # [SwinSmall, './checkpoints/Swin-small/SwinV1-small-fold-0-swa.pth'],
    # [SwinSmall, './checkpoints/Swin-small/SwinV1-small-fold-1-swa.pth'],
    # [SwinSmall, './checkpoints/Swin-small/SwinV1-small-fold-2-swa.pth'],
    # [SwinSmall, './checkpoints/Swin-small/SwinV1-small-fold-3-swa.pth'],
    # [SwinSmall, './checkpoints/Swin-small/SwinV1-small-fold-4-swa.pth'],

    # --- SegFormer MIT-B3 (image_size=768) ---
    # [Segformer, './checkpoints/Segformer-mitb3/Segformer-b3-fold-0-swa.pth'],
    # [Segformer, './checkpoints/Segformer-mitb3/Segformer-b3-fold-1-swa.pth'],
    # [Segformer, './checkpoints/Segformer-mitb3/Segformer-b3-fold-2-swa.pth'],
    # [Segformer, './checkpoints/Segformer-mitb3/Segformer-b3-fold-3-swa.pth'],
    # [Segformer, './checkpoints/Segformer-mitb3/Segformer-b3-fold-4-swa.pth'],

    # --- PVT-V2 + U-Net (image_size=768) ---
    # [PVTUnet, './checkpoints/PVTV4-UNet/pvt-unet-fold-0-swa.pth'],
    # [PVTUnet, './checkpoints/PVTV4-UNet/pvt-unet-fold-1-swa.pth'],
    # [PVTUnet, './checkpoints/PVTV4-UNet/pvt-unet-fold-2-swa.pth'],
    # [PVTUnet, './checkpoints/PVTV4-UNet/pvt-unet-fold-3-swa.pth'],
    # [PVTUnet, './checkpoints/PVTV4-UNet/pvt-unet-fold-4-swa.pth'],

    # --- ConvNeXt-Large + SMP U-Net (image_size=1280) ---
    # [ConvNextUNet, './checkpoints/Convnext-Smpunet/convnext-smpunet-fold-0-swa.pth'],
    # [ConvNextUNet, './checkpoints/Convnext-Smpunet/convnext-smpunet-fold-1-swa.pth'],
    # [ConvNextUNet, './checkpoints/Convnext-Smpunet/convnext-smpunet-fold-2-swa.pth'],
    # [ConvNextUNet, './checkpoints/Convnext-Smpunet/convnext-smpunet-fold-3-swa.pth'],
    # [ConvNextUNet, './checkpoints/Convnext-Smpunet/convnext-smpunet-fold-4-swa.pth'],

    # --- CoAt + DAFormer (image_size=768) ---
    # [CoatDaformer, './checkpoints/Coat_DaFormer3x3/coat-daformer-3x3-fold-0-swa.pth'],
    # [CoatDaformer, './checkpoints/Coat_DaFormer3x3/coat-daformer-3x3-fold-1-swa.pth'],
    # [CoatDaformer, './checkpoints/Coat_DaFormer3x3/coat-daformer-3x3-fold-2-swa.pth'],
    # [CoatDaformer, './checkpoints/Coat_DaFormer3x3/coat-daformer-3x3-fold-3-swa.pth'],
    # [CoatDaformer, './checkpoints/Coat_DaFormer3x3/coat-daformer-3x3-fold-4-swa.pth'],

    # --- DualViT + DAFormer (image_size=768) ---
    # [DualDaFormer, './checkpoints/Dual_DaFormer/dual-daformer-fold-0-swa.pth'],
    # [DualDaFormer, './checkpoints/Dual_DaFormer/dual-daformer-fold-1-swa.pth'],
    # [DualDaFormer, './checkpoints/Dual_DaFormer/dual-daformer-fold-2-swa.pth'],
    # [DualDaFormer, './checkpoints/Dual_DaFormer/dual-daformer-fold-3-swa.pth'],
    # [DualDaFormer, './checkpoints/Dual_DaFormer/dual-daformer-fold-4-swa.pth'],

    # --- PVT-V2-B3 + MeanPool DAFormer (final best model) ---
    [PVT2B4Daformer, './checkpoints/pvt-unet-mp/pvt-unet-mp-fold-0-swa.pth'],
    [PVT2B4Daformer, './checkpoints/pvt-unet-mp/pvt-unet-mp-fold-1-swa.pth'],
    [PVT2B4Daformer, './checkpoints/pvt-unet-mp/pvt-unet-mp-fold-2-swa.pth'],
    [PVT2B4Daformer, './checkpoints/pvt-unet-mp/pvt-unet-mp-fold-3-swa.pth'],
    [PVT2B4Daformer, './checkpoints/pvt-unet-mp/pvt-unet-mp-fold-4-swa.pth'],
]

threshold = 0.55

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

test_csv = './test.csv'
test_df = pd.read_csv(test_csv)


def off_submit():
    num_net = len(model)
    print(f'num_net = {num_net}')

    net = []
    for i in range(num_net):
        Net, checkpoint = model[i]
        n = Net()
        f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        n.load_state_dict(f['state_dict'], strict=False)  # True
        # n.load_state_dict(f['state_dict'], strict=True)  # True
        n.cuda()
        n.eval()
        net.append(n)
    

    test_dataset = BaseDataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size  = 1,
        drop_last   = False,
        num_workers = 8,     
        pin_memory  = False,
        collate_fn = null_collate,
    )


    if 1:
        test_num = 0

        start_timer = timer()
        for t, batch in enumerate(test_loader):
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].cuda()
            #print(t, batch['image'].shape)

            p = 0
            count = 0
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    # pdb.set_trace()
                    for i in range(num_net):
                        
                        p += TTA(batch, net[i])

                        # p += net[i](batch)  # net(input)#
                        count += 1

                        # TTA
                        # batch['image'] = torch.flip(batch['image'], dims=[3, 1])
                        # p += net[i](batch)
                        # count += 1

            p = p / count
            # result['probability'].append(p.float().data.cpu().numpy())
            test_num += batch_size
            # pdb.set_trace()
            probability = (p.float().data.cpu().numpy())
            probability[probability>threshold] = 1
            probability[probability<threshold] = 0
            
            fold_dir = f'./output/PVT2B4Daformer-thr={threshold}-TTA'


            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            # pdb.set_trace()
            # cv2.imwrite(fold_dir + '/{}'.format(batch['img_name'][0]), 255 * probability.squeeze(0))
            probability = 255 * probability.squeeze(0)
            # probability = 255 * probability.squeeze(0).squeeze(0)
            cv2.imwrite(fold_dir + '/{}'.format(batch['img_name'][0]), probability[160:-160, :])
            # cv2.imwrite(fold_dir + '/{}'.format(batch['img_name'][0]), probability.squeeze(0).squeeze(0))
            # cv2.imwrite(fold_dir + '/{}'.format(batch['img_name'][0]), probability.squeeze(0).squeeze(0))
            # print(probability.shape)
            
            # pdb.set_trace()
            # test_df.loc[test_df['img_name'].isin(batch['id']), 'pred_prob'] = probability
            
            print('\r %8d / %d  %s' % (test_num, len(test_dataset), time_to_str(timer() - start_timer, 'sec')), end='', flush=True)
        print('')
    

if __name__ == '__main__':
    off_submit()