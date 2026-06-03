# import sys
# import logging
# import argparse
# from loguru import logger

import os
import torch
import warnings
import numpy as np
import torch.nn as nn
from engine import do_valid
from Model import build_model

from torch.optim import RAdam
from MyOptimizer.optim import Lookahead
from MyDataset import build_dataset
from MyDataset.transforms import make_transforms
from timeit import default_timer as timer
from torch.nn.parallel import data_parallel
from MyDataset.dataset import BaseDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from Libs.libs import scheduler, get_learning_rate, adjust_learning_rate, time_to_str, set_all_random_seed, null_collate
from Config import CFG

warnings.filterwarnings("ignore", category=UserWarning)

IDENTIFIER = CFG.model

# parser = argparse.ArgumentParser('Task Name -- Training and Evaluation Scripts', parents=[get_args_parser()])
# args = parser.parse_args()


def main(fold):

    start_lr   = CFG.start_lr 
    num_epoch = CFG.epochs
    skip_save_epoch = CFG.epochs // 2
    
    fold_type = 'tianchi-fold'

    name = 'teeth-seg'

    root_dir = CFG.save_root_dir + '_' + name
    out_dir  = root_dir + f'/{IDENTIFIER}-3x3/'
    fold_dir = out_dir+ f'/fold-{fold}-{fold_type}'

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # logging.basicConfig(filename=fold_dir + "/basic_config.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
                        
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))

    initial_checkpoint = CFG.initial_checkpoint
    
    # ------------------------------------- setup  ----------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(fold_dir +'/'+f, exist_ok=True)


    # -------------------------------- log = Logger() ------------------------------------
    log = open(fold_dir+'/log.train.txt',mode='a')
    # log.write(f'\n--- [START {log.timestamp()}] {"-"*64}\n\n')
    # log.write(f'\t{set_environment()}\n')
    log.write(f'\t__file__ = {__file__}\n')
    log.write(f'\tfold_dir = {fold_dir}\n')
    log.write(f'\n')

    #------------------------------------ dataset ----------------------------------------
    log.write('** dataset setting **\n')

    if fold_type == 'tianchi-fold':
        train_df, valid_df = build_dataset(image_set='Train_Val', args=CFG, fold=fold)

    elif fold_type == 'tianchi-all':
        train_df, valid_df = build_dataset(image_set='All', args=CFG, fold=fold)

    else:
        raise ValueError('Please Select One Type!')

    train_dataset = BaseDataset(train_df, make_transforms(image_set='Train'))
    valid_dataset = BaseDataset(valid_df, None)

    train_loader  = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size  = CFG.train_batch_size,
        drop_last   = False,
        num_workers = CFG.num_workers,        # 16
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn = null_collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = CFG.val_batch_size,
        drop_last   = False,
        num_workers = CFG.num_workers,        # 16
        pin_memory  = False,
        collate_fn = null_collate,
    )

    log.write(f'fold_type = {fold_type}\n')
    log.write(f'fold = {fold}\n')
    log.write(f'train_dataset : \n{str(train_dataset)}\n')
    log.write(f'valid_dataset : \n{str(valid_dataset)}\n')
    log.write('\n')

    #------------------------------------- net ----------------------------------------
    log.write('** net setting **\n')

    scaler = torch.cuda.amp.GradScaler(enabled = True)
    net = build_model(args=CFG)
    net.cuda()
    net.load_pretrain()


    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f.get('iteration',0)
        start_epoch = f.get('epoch',0)
        state_dict = f['state_dict']
        print(net.load_state_dict(state_dict,strict=False))  #True
    else:
        start_iteration = 0
        start_epoch = 0
    log.write(f'\tinitial_checkpoint = {initial_checkpoint}\n')
    log.write(f'\n')


    #------------------------------------ optimizer ----------------------------------
    if 0: ##freeze
        for p in net.encoder.parameters():   p.requires_grad = False
        pass

    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    #freeze_bn(net)

    optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('\n')

    num_iteration = num_epoch*len(train_loader)
    iter_log   = len(train_loader) * 3
    iter_valid = iter_log
    iter_save  = iter_log

    #---------------------------------- start training here! ----------------------------------------------
    log.write('** start training here! **\n')
    log.write('   batch_size = %d \n'%(CFG.train_batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                           |-------------- VALID ---------|----------- TRAIN/BATCH -----------\n')
    log.write('rate      iter       epoch | loss   dice   IoU    hd95    |  loss1   loss0		  | time     \n')
    log.write('----------------------------------------------------------------------------------------------\n')

    def message(mode='print'):
        asterisk = ' '
        if mode==('print'):
            loss = batch_loss
        if mode==('log'):
            loss = train_loss
            if (iteration % iter_save == 0): asterisk = '*'
        
        text = \
            ('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
            '%4.4f  %4.4f  %4.4f  %4.4f  %4.4f  | '%(*valid_loss,) + \
            '%4.4f  %4.4f  | '%(*loss,) + \
            '%s' % (time_to_str(timer() - start_timer,'min'))

        return text

    valid_loss = np.zeros(5,np.float32)
    train_loss = np.zeros(2,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0


    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    pass

    while iteration <= num_iteration:
        for t, batch in enumerate(train_loader):
        # for t, batch in tqdm(enumerate(train_loader),total = len(train_loader)):
            
            if iteration%iter_save==0 and epoch > skip_save_epoch:
                if iteration != start_iteration:
                    n = iteration if epoch > skip_save_epoch else 0
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, fold_dir + '/checkpoint/%03d_%s.pth' % (int(epoch+1), IDENTIFIER))
                    # }, fold_dir + f'/checkpoint/{n:08d}.model.pth')
                    pass
            
            
            if (iteration >= 0 and iteration % iter_valid == 0): # or (t==len(train_loader)-1):
                # if iteration!=start_iteration:
                    # valid_loss = do_valid(net, valid_loader, f'{iteration:08d}')  #
                valid_loss = do_valid(net, valid_loader, CFG)  #
                # pass
            
            
            if (iteration % iter_log == 0) or (iteration % iter_valid == 0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')
                
                
            # learning rate schduler ------------
            adjust_learning_rate(optimizer, scheduler(epoch))
            # rate = get_learning_rate(optimizer)[0] #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)
            rate = get_learning_rate(optimizer) #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)
            
            # one iteration update  -------------
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].cuda()
            batch['mask' ] = batch['mask' ].cuda()
            net.train()

            if epoch <= 48:
                CFG.drop_rate = 0
            elif epoch >= 49 and epoch <= 64:
                CFG.drop_rate = 0.4
            else:
                CFG.drop_rate = 0.2

            net.output_type = ['loss', 'inference']
            #with torch.autograd.set_detect_anomaly(True):
            if 1:
                with torch.cuda.amp.autocast(enabled=True):
                    output = data_parallel(net,batch)

                    loss1  = output['loss'].mean()
                    # loss2  = output['dice_loss'].mean()
                    loss0  = output['aux2_loss'].mean()
                    

                optimizer.zero_grad()
                scaler.scale(loss1 + 0.2 * loss0).backward()
                
                #scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                
                scaler.step(optimizer)
                scaler.update()
            
            # print statistics  --------
            batch_loss[:2] = [loss1.item(), loss0.item()]
            sum_train_loss += batch_loss
            sum_train += 1
            if t % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0
            
            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
            epoch += 1 / len(train_loader)
            iteration += 1
            
        torch.cuda.empty_cache()
    log.write('\n')


if __name__ == '__main__':

    set_all_random_seed(seed=CFG.seed, cudnn_deterministic=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = CFG.gpu_id

    for fold in range(CFG.num_fold):
        main(fold)
