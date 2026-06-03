import os
import torch 
import torch.nn as nn

def do_swa(checkpoint):
	skip = ['relative_position_index', 'num_batches_tracked']
	K = len(checkpoint)
	swa = None
	for k in range(K):
		state_dict = torch.load(checkpoint[k], map_location=lambda storage, loc: storage)['state_dict']
		if swa is None:
			swa = state_dict
		else:
			for k, v in state_dict.items():
				# print(k)
				if any(s in k for s in skip): continue
				swa[k] += v
	for k, v in swa.items():
		if any(s in k for s in skip): continue
		swa[k] /= K
	return swa

def main():
    # Set your result directory here
    out_dir = f'./result/pvt-unet-mp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    valid = {
        0: [
            '084_pvt-unet-mp.pth',
            '081_pvt-unet-mp.pth',
            '079_pvt-unet-mp.pth',
            '076_pvt-unet-mp.pth',
            '073_pvt-unet-mp.pth',
        ],
        1: [
            '084_pvt-unet-mp.pth',
            '081_pvt-unet-mp.pth',
            '079_pvt-unet-mp.pth',
            '076_pvt-unet-mp.pth',
            '073_pvt-unet-mp.pth',
        ],
        2: [
            '084_pvt-unet-mp.pth',
            '081_pvt-unet-mp.pth',
            '079_pvt-unet-mp.pth',
            '076_pvt-unet-mp.pth',
            '073_pvt-unet-mp.pth',
        ],
        3: [
            '084_pvt-unet-mp.pth',
            '081_pvt-unet-mp.pth',
            '079_pvt-unet-mp.pth',
            '076_pvt-unet-mp.pth',
            '073_pvt-unet-mp.pth',
        ],
        4: [
            '084_pvt-unet-mp.pth',
            '081_pvt-unet-mp.pth',
            '079_pvt-unet-mp.pth',
            '076_pvt-unet-mp.pth',
            '073_pvt-unet-mp.pth',
        ],
    }

    for f, checkpoint in valid.items():
        if len(checkpoint) == 0: continue

        project_name = out_dir.split('/')[-1]
        fold_dir = out_dir + '/fold-%d-tianchi-fold' % f

        checkpoint = [fold_dir + '/checkpoint/' + c for c in checkpoint]
        swa = do_swa(checkpoint)
        torch.save({
            'state_dict': swa,
            'swa': [c.split('/')[-1] for c in checkpoint],
        }, fold_dir + '/%s-fold-%d-swa.pth' % (project_name, f))

        ## setup  ----------------------------------------
        submit_dir = fold_dir + '/valid/swa'
        os.makedirs(submit_dir, exist_ok=True)


if __name__ == '__main__':
    main()