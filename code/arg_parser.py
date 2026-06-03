import argparse

def get_args_parser():

	parser = argparse.ArgumentParser('Task Name', add_help=False)
	
	# ------------------------------- path setting -------------------------------
	parser.add_argument('--data_dir', default='./data', type=str)
	parser.add_argument('--save_root_dir', default='./result', type=str)

	# ------------------------------- model setting -------------------------------
	parser.add_argument('--model', default='model_v3', type=str)
	parser.add_argument('--label_smoothing', default=0.75, type=float)
	parser.add_argument('--num_class', default=250, type=int)
	parser.add_argument('--max_length', default=128, type=int)
	parser.add_argument('--embed_dim', default=384, type=int)
	parser.add_argument('--num_point', default=1056, type=int)
	parser.add_argument('--num_head', default=4, type=int)
	parser.add_argument('--num_block', default=1, type=int)
	
	parser.add_argument('--aug', default='v8', type=str)
	parser.add_argument('--drop_rate', default=0.1, type=float)


	parser.add_argument("--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters")
	parser.add_argument("--model-ema-steps", type=int, default=32, help="the number of iterations that controls how often to update the EMA model (default: 32)",)
	parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)


	# ------------------------------- CE Loss -------------------------------
	parser.add_argument('--ce_loss', default=True, type=bool)
	parser.add_argument('--ce_weight', default=1.0, type=float)

	# ------------------------------- Focal Loss -------------------------------
	parser.add_argument('--focal_loss', default=False, type=bool)
	parser.add_argument('--focal_weight', default=0, type=float)
	parser.add_argument('--alpha', default=0.25, type=int)
	parser.add_argument('--gamma', default=2, type=int)

	# ------------------------------- Dice Loss -------------------------------
	parser.add_argument('--dice_loss', default=False, type=bool)
	parser.add_argument('--dice_weight', default=0, type=float)
	# parser.add_argument('--alpha', default=0.5, type=int)
	# parser.add_argument('--beta', default=0.5, type=int)


	parser.add_argument('--seed', default=123, type=int)

	parser.add_argument('--dataset', default='Your Dataset', type=str)
	parser.add_argument('--gpu_id', default='7', type=str)

	parser.add_argument('--start_lr', default=1e-4, type=float)
	parser.add_argument('--min_lr', default=1e-4, type=float)
	parser.add_argument('--train_batch_size', default=64, type=int)
	parser.add_argument('--val_batch_size', default=128, type=int)
	parser.add_argument('--num_workers', default=8, type=int)
	
	parser.add_argument('--initial_checkpoint', default=None, type=str)
	parser.add_argument('--epochs', default=84, type=int)
	parser.add_argument('--skip_save_epoch', default=42, type=int)
	
	parser.add_argument('--decay_epochs', default=40, type=int)

	parser.add_argument('--num_fold', default=5, type=int)
	parser.add_argument('--fold', default=0, type=int)
	
	parser.add_argument('--is_tf32', default=0, type=int)
	parser.add_argument('--is_tf16', default=1, type=int)
	parser.add_argument('--is_tf8', default=0, type=int)

	return parser

# class CFG:

#     def __init__(self):
#         # self.image_size          = 1024

#         self.seed                = 0
#         # self.seed                = 7
#         # self.seed                = 27
#         self.seed                = 42
#         # self.seed                = 127
#         # self.seed                = 277
#         # self.seed                = 777
#         # self.seed                = 2777
#         # self.seed                = 4227
#         # self.seed                = 980127
#         # self.seed                = 98012742

#         self.start_lr            = 2e-4      
#         self.min_lr              = 1e-5     

#         self.train_batch_size    = 64
#         self.val_batch_size      = 128
#         self.num_workers         = 8        # 0

#         self.initial_checkpoint  = None
#         self.decay_epochs        = 40        
#         self.epochs              = 70        
#         self.fold_num            = 5

#         # self.label_smoothing     = 0.75
#         self.label_smoothing     = 0.5

#         self.num_class           = 250
#         self.max_length          = 256     #256    # 384     # 60
#         self.embed_dim           = 384     # 
#         self.num_point           = 960     # 82
#         # self.point_dim           = 960     # 82
#         self.num_head            = 4
#         self.num_block           = 1

#     # def info_print(self):
#     #     print('image_size:{},\n'  
#     #            'start_lr:{},\n' 
#     #            'train_batch_size:{},\n' 
#     #            'val_batch_size:{},\n' 
#     #            'num_workers:{},\n'
#     #            'initial_checkpoint:{},\n' 
#     #            'fold_num:{},\n'
#     #            'epochs:{},'.format(
#     #                                self.image_size,
#     #                                self.start_lr, 
#     #                                self.train_batch_size, 
#     #                                self.val_batch_size,
#     #                                self.num_workers,
#     #                                self.initial_checkpoint,
#     #                                self.fold_num,
#     #                                self.epochs))