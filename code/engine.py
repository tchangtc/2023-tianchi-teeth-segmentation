import pdb
import torch
import numpy as np
# from medpy.metric import binary
from Libs.libs import time_to_str
from torch.nn.parallel import data_parallel
from timeit import default_timer as timer
from Libs import metrics



# def calculate_distance(label_pred, label_true, spacing, C, percentage=95):
#     # the input args are torch tensors
#     if label_pred.is_cuda:
#         label_pred = label_pred.cpu()
#         label_true = label_true.cpu()

#     label_pred = label_pred.numpy()
#     label_true = label_true.numpy()
#     spacing = spacing.numpy()

#     ASD_list = np.zeros(C-1)
#     HD_list = np.zeros(C-1)

#     for i in range(C-1):
#         tmp_surface = metrics.compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
#         dis_gt_to_pred, dis_pred_to_gt = metrics.compute_average_surface_distance(tmp_surface)
#         ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2 

#         HD = metrics.compute_robust_hausdorff(tmp_surface, percentage)
#         HD_list[i] = HD

#     return ASD_list, HD_list


# def hd95(probability, mask):
#     #labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
#     #labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
#     #hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
#     #hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
#     #return hausdorffcomputer.GetAverageHausdorffDistance()
	
# 	# pdb.set_trace()

# 	# for p, t in zip(probability, mask):


# 	if probability.sum() > 0 and mask.sum() > 0:
# 		# pdb.set_trace()
# 		hd95 = binary.hd95(probability, mask)
# 		# print(hd95)
# 		return  hd95
# 	else:
# 		return 0

def compute_IoU_score(probability, mask, thr):
	N = len(probability)
	p = probability.reshape(N, -1)
	t = mask.reshape(N, -1)

	p = p > thr
	t = t > thr
	union = p.sum(-1) + t.sum(-1)
	overlap = (p * t).sum(-1)

	IoU = overlap / (union + 1e-9 - overlap)
	return IoU

def compute_dice_score(probability, mask, thr):
    N = len(probability)
    p = probability.reshape(N,-1)
    t = mask.reshape(N,-1)

    p = p > thr
    t = t > thr
    uion = p.sum(-1) + t.sum(-1)
    overlap = (p * t).sum(-1)
    dice = 2 * overlap / (uion + 1e-9)
    return dice

def np_cross_entropy(probability, truth):
	p = np.clip(probability,1e-4,1-1e-4)
	logp = -np.log(p)
	loss = logp[np.arange(len(logp)),truth]
	loss = loss.mean()
	return loss


@torch.no_grad()
def do_valid(net, valid_loader, args):
	valid_num = 0
	valid_probability = []
	valid_mask = []
	valid_loss = 0

	net = net.eval()
	start_timer = timer()
	for t, batch in enumerate(valid_loader):
	# for t, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
		
		net.output_type = ['loss', 'inference']
		with torch.no_grad():
			with torch.cuda.amp.autocast(enabled = True):
				# pdb.set_trace()

				batch_size = len(batch['index'])
				batch['image'] = batch['image'].cuda()
				batch['mask' ] = batch['mask' ].cuda()

				# batch['xyz'] = [xyz.half().cuda() for xyz in batch['xyz']]
				# output = net(batch) #data_parallel(net, batch) #
				output = data_parallel(net, batch) #
				loss  = output['loss'].mean() \
				# if args.focal_loss:
				# 	focal_loss = output['focal_loss'].mean()
				# 	focal_loss_value = focal_loss.item()
				# else:
				# 	focal_loss_value = 0.0

				# if args.dice_loss:
				# 	dice_loss = output['dice_loss'].mean()
				# 	dice_loss_value = dice_loss.item()
				# else:
				# 	dice_loss_value = 0.0

		valid_probability.append(output['probability'].data.cpu().numpy())
		valid_mask.append(batch['mask'].data.cpu().numpy())
		valid_num += batch_size

		valid_loss += batch_size * loss.item()

		#---
		print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
		#if valid_num==200*4: break

	#print('')
	assert(valid_num == len(valid_loader.dataset))
	#------
	probability = np.concatenate(valid_probability)
	mask = np.concatenate(valid_mask)

	# pdb.set_trace()

	loss = valid_loss / valid_num
	
	dice = compute_dice_score(probability, mask, thr=0.5)
	IoU  = compute_IoU_score(probability, mask, thr=0.5)
	# hd = hd95(probability, mask)

	dice = dice.mean()
	IoU  = IoU.mean()
	# hd   = hd.mean()

	# iou = dice / (2 - dice)
	
	# print(' \n')
	# print('droprate: ', args.drop_rate)
	# print(' \n')

	# score = 0.4 * dice + 0.3 * IoU + 0.3 * hd


	return [loss, dice, IoU, 0, 0]
	# return [loss, dice, IoU, hd, score]
