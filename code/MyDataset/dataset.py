import pdb
import cv2
import torch
import numpy as np
import pandas as pd
from Config import CFG
from torch.utils.data import Dataset
from Libs.libs import image_to_tensor, mask_to_tensor
from sklearn.model_selection import KFold

def read_csv_by_fold(fold):
    df = pd.read_csv('../train.csv')
    print(fold)
    num_fold = CFG.num_fold
    # num_fold = cfg.fold_num

    kf = KFold(n_splits=num_fold, shuffle=True, random_state=27)

    for f, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, 'fold'] = f

    train_df = df[df.fold!=fold].reset_index(drop=True)
    valid_df = df[df.fold==fold].reset_index(drop=True)

    return train_df, valid_df


# def read_all_df(args):

#     return 

def read_all_df():
    df = pd.read_csv('../train.csv')
    # sign_to_label = read_json_file(args.js_path)

    train_df = df.reset_index(drop=True)
    valid_df = df.reset_index(drop=True)
    return train_df, valid_df


class BaseDataset(Dataset):

    def transforms_v0(self,):

        pass

    def __init__(self, df, transforms=None, image_size=640):

        self.df = df
        self.transforms = transforms
        self.length = len(self.df)
        self.image_size = image_size

    def __len__(self,):

        return self.length


    def __str__(self) -> str:
        num_image = len(self.df)

        string = ''
        string += f'\tnum_image = {num_image}\n'

        # count = dict(self.df.img_label.value_counts())
        # for k in [0, 1]:
        #     string += f'\t\label{k} = {count[k]:5d} ({count[k]/len(self.df):0.3f})\n'
        return string


    def __getitem__(self, index):
        

        # try:  augmentation 
        #       normalization


        dd = self.df.iloc[index]
        # id = d['id']
        
        # image = cv2.imread(f'{dd.img_path}')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask  = cv2.imread(f'{dd.mask_path}', cv2.IMREAD_UNCHANGED)
        image = cv2.imread(f'{dd.img_path}', cv2.IMREAD_COLOR)
        mask  = cv2.imread(f'{dd.mask_path}', cv2.IMREAD_UNCHANGED)

        assert image[:,:,0].shape == mask.shape
        assert mask.shape == (320, 640)

        # pdb.set_trace()
        pad_h = (self.image_size - image.shape[0]) // 2
        pad_w = (self.image_size - image.shape[1]) // 2

        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        mask  = np.pad(mask,  ((pad_h, pad_h), (pad_w, pad_w)))
        
        # image = cv2.resize(image, dsize=None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        # mask  = cv2.resize(mask,  dsize=None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, dsize=None, fx=CFG.scale, fy=CFG.scale, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  dsize=None, fx=CFG.scale, fy=CFG.scale, interpolation=cv2.INTER_LINEAR)

        # pdb.set_trace()
        image = image.astype(np.float32)
        mask  = mask.astype(np.float32)
        
        sample = {'image': image, 'mask': mask}
        # Data Augmentation
        if self.transforms:
            # sample = self.transforms(image=image, mask=mask)
            image = sample['image']
            mask = sample['mask']
            sample = self.transforms(sample)

        image = sample['image']
        mask = sample['mask']
        
        # Normalization
        image = image / 255.0
        # mask  = mask / 255.0
        mask[mask > 0] = 1


        rr = {}
        rr['index'] = index
        # rr['image'] = image
        # rr['mask'] = mask
        rr['image'] = image_to_tensor(image)
        rr['mask'] = mask_to_tensor(mask)

        return rr



def build(image_set, args, fold):
    if image_set == 'Train_Val':
        train_df, valid_df = read_csv_by_fold(fold)
        return train_df, valid_df 
    elif image_set == 'All':
        train_df, valid_df = read_all_df()
        return train_df, valid_df
    else:
        raise ValueError('Please Train Frist')
    

#################################################################################

# def run_check_dataset():

#     dataset = SignDataset(valid_df)
#     print(dataset)

#     for i in range(12):
#         r = dataset[i]
#         print(r['index'], '--------------------')
#         print(r["d"], '\n')
#         for k in tensor_key:
#             if k =='label': continue
#             v = r[k]
#             print(k)
#             print('\t', 'dtype:', v.dtype)
#             print('\t', 'shape:', v.shape)
#             if len(v)!=0:
#                 print('\t', 'min/max:', v.min().item(),'/', v.max().item())
#                 print('\t', 'is_contiguous:', v.is_contiguous())
#                 print('\t', 'values:')
#                 print('\t\t', v.reshape(-1)[:5].data.numpy().tolist(), '...')
#                 print('\t\t', v.reshape(-1)[-5:].data.numpy().tolist())
#         print('')
#         if 0:
#             # #draw
#             # cv2.waitKey(1)
#             pass



#     loader = DataLoader(
#         dataset,
#         sampler=SequentialSampler(dataset),
#         batch_size=8,
#         drop_last=True,
#         num_workers=0,
#         pin_memory=False,
#         worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
#         collate_fn=null_collate,
#     )
#     print(f'batch_size   : {loader.batch_size}')
#     print(f'len(loader)  : {len(loader)}')
#     print(f'len(dataset) : {len(dataset)}')
#     print('')

#     for t, batch in enumerate(loader):
#         if t > 5: break
#         print('batch ', t, '===================')
#         print('index', batch['index'])

#         for k in tensor_key:
#             v = batch[k]

#             if k =='label':
#                 print('label:')
#                 print('\t', v.data.numpy().tolist())

#             if k =='x':
#                 print('x:')
#                 print('\t', v.data.shape)

#             if k =='xyz':
#                 print('xyz:')
#                 for i in range(len(v)):
#                     print('\t', v[i].shape)

#         if 1:
#             pass
#         print('')


# # main #################################################################
# if __name__ == '__main__':
#     run_check_dataset()