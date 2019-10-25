# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
#import argparse

import torch


def add_parser_params(parser):
	"""add argument to the parser"""

	# checkpoint
	parser.add_argument('--resume', type=str, default=None,
							help='put the path to resuming file if needed')

	parser.add_argument('--checkname', type=str, default=None,
							help='the name of the checkpoint.')

	parser.add_argument('--save_ckpt_steps', type=int, default=500,
							help='save checkpoints per save_ckpt_steps')

	parser.add_argument('--max_ckpt_nums', type=int, default=15,
							help='the max numbers of checkpoints')

	parser.add_argument('--model_dir', type=str, default='/home/zhaoshuai/models/deeplabv3_cbl_2/',
						help='Base directory for the model.')

	parser.add_argument('--output_dir', type=str, default='/home/zhaoshuai/models/deeplabv3_cbl_2/',
						help='output directory of model.')

	# base model and the network
	parser.add_argument('--seg_model', type=str, default='deeplabv3',
							choices=['deeplabv3', 'deeplabv3+', 'pspnet'],
							help='The segmentation model.')

	parser.add_argument('--backbone', type=str, default='resnet101',
							choices=['resnet50', 'resnet101', 'resnet152',
							'resnet50_beta', 'resnet101_beta', 'resnet152_beta'],
							help='backbone name (default: resnet101)')

	parser.add_argument('--out_stride', type=int, default=16,
							help='network output stride (default: 16)')
	# batch size and crop size
	parser.add_argument('--batch_size', type=int, default=16, metavar='N',
							help='input batch size for training (default: auto)')

	parser.add_argument('--accumulation_steps', type=int, default=1, metavar='N',
							help='Accumulation steps when calculate the gradients when training')

	parser.add_argument('--test_batch_size', type=int, default=None, metavar='N',
							help='input batch size for testing (default: auto)')

	# dataset information
	parser.add_argument('--dataset', type=str, default='pascal'	,
							choices=['pascal', 'coco', 'cityscapes', 'camvid'],
							help='dataset name (default: pascal)')

	parser.add_argument('--train_split', type=str, default='train'	,
							choices=['train', 'trainaug', 'trainval', 'val', 'test'],
							help='training set name (default: train)')

	parser.add_argument('--data_dir', type=str, default='/dataset',
						help='Path to the directory containing the PASCAL VOC data.')

	parser.add_argument('--use_sbd', action='store_true', default=False,
							help='whether to use SBD dataset (default: True)')

	parser.add_argument('--workers', type=int, default=8, metavar='N',
							help='dataloader threads')

	parser.add_argument('--base_size', type=int, default=513, help='base image size')

	parser.add_argument('--crop_size', type=int, default=513, help='crop image size')

	# batch normalization
	parser.add_argument('--sync_bn', type=bool, default=None,
							help='whether to use sync bn (default: auto)')

	parser.add_argument('--freeze_bn', type=bool, default=False,
							help='whether to freeze bn parameters (default: False)')

	parser.add_argument('--bn_mom', type=float, default=0.1, metavar='M',
							help='momentum (default: 0.1) for running mean and var of batch normalization')

	# training hyper params
	parser.add_argument('--epochs', type=int, default=46, metavar='N',
							help='Number of training epochs: '
							'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
							'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
							'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
							'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
							'For 30K iteration with batch size 12, train_epoch = 34.02 (= 30K * 12 / 10,582). '
							'For 30K iteration with batch size 14, train_epoch = 39.69 (= 30K * 14 / 10,582). '
							'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
							'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

	parser.add_argument('--start_epoch', type=int, default=0,
							metavar='N', help='start epochs (default:0)')

	parser.add_argument('--init_global_step', type=int, default=0,
							help='Initial global step for controlling learning rate when fine-tuning model.')

	parser.add_argument('--use_balanced_weights', action='store_true', default=False,
							help='whether to use balanced weights (default: False)')

	# optimizer params, such as learning rate
	parser.add_argument('--init_lr', type=float, default=0.007,
							help='learning rate (default: auto)')

	parser.add_argument('--lr_multiplier', type=float, default=1.0,
					help='Learning rate multiplier for the unpretrained model.')

	parser.add_argument('--slow_start_lr', type=float, default=1e-4,
					help='Learning rate employed during slow start.')

	parser.add_argument('--slow_start_steps', type=int, default=0,
					help='Training model with small learning rate for few steps.')

	parser.add_argument('--lr_scheduler', type=str, default='poly',
							choices=['poly', 'step', 'cos'],
							help='lr scheduler mode: (default: poly)')

	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
							help='momentum (default: 0.9)')

	parser.add_argument('--weight_decay', type=float, default=1e-4,
							metavar='M', help='w-decay (default: 1e-4)')

	parser.add_argument('--nesterov', action='store_true', default=False,
							help='whether use nesterov (default: False)')

	# cuda, seed and logging
	parser.add_argument('--no_cuda', action='store_true', default=False,
							help='disables CUDA training')

	parser.add_argument('--gpu_ids', type=str, default='0',
							help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

	parser.add_argument('--main_gpu', type=int, default=0,
							help='The main gpu')

	parser.add_argument('--seed', type=int, default=1, metavar='S',
							help='random seed (default: 1)')

	# finetuning pre-trained models
	parser.add_argument('--ft', action='store_true', default=False,
							help='finetuning on a different dataset')

	# evaluation option
	parser.add_argument('--eval_interval', type=int, default=2,
							help='evaluuation interval (default: 2)')

	parser.add_argument('--no_val', action='store_true', default=False,
							help='skip validation during training')

	# loss type
	parser.add_argument('--loss_type', type=int, default=0,
							help='The loss type used.')

	parser.add_argument('--loss_weight_lambda', type=float, default=0.5,
							help='The realtive weight factor for the loss.')

	# process info
	parser.add_argument('--proc_name', type=str, default='DeepLabv3',
							help='The name of the process.')

	# region mutual information
	parser.add_argument('--rmi_pool_way', type=int, default=1,
							help="The pool way when calculate RMI loss, 1 - avg pool, 0 - max pool")

	parser.add_argument('--rmi_pool_size', type=int, default=2,
							help="The pool size of the pool operation before calculate RMI loss")

	parser.add_argument('--rmi_pool_stride', type=int, default=2,
							help="The pool stride of the pool operation before calculate RMI loss")

	parser.add_argument('--rmi_radius', type=int, default=3,
							help="The square radius of rmi [1, 3, 5, 7], they have a center")

	# CRF iter steps
	parser.add_argument('--crf_iter_steps', type=int, default=1,
							help='The iter steps of the crf')

	# torch.parallel.DistributedDataParallel(), not avaliable now.
	parser.add_argument('--local_rank', type=int, default=0)

	parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')

	parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

	parser.add_argument('--multiprocessing_distributed', action='store_true',
							help='Use multi-processing distributed training to launch '
							'N processes per node, which has N GPUs. This is the '
							'fastest way to use PyTorch for either single node or '
							'multi node data parallel training')

	# parse
	args, unparsed = parser.parse_known_args()

	# RMI parameters
	args.rmi_pool_stride = args.rmi_pool_size

	# use gpu or not
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
			args.gpu_ids = [i for i in range(0, len(args.gpu_ids))]
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

	# We only have one node and N GPUs.
	args.world_size = int(len(args.gpu_ids))
	# distributed parrallel or not
	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	# use synchronized batch normalization across multi gpus, or not
	if args.sync_bn is None:
		args.sync_bn = True if (args.cuda and len(args.gpu_ids) > 1) else False

	# default settings for epochs, batch_size and lr
	if args.epochs is None:
		epoches = {
					'coco': 30,
					'cityscapes': 200,
					'pascal': 46,
				}
		args.epochs = epoches[args.dataset.lower()]

	# train batch size
	assert args.accumulation_steps in [1, 2, 4]
	assert args.batch_size in [4, 8, 12, 16, 32, 36, 48, 64]
	args.batch_size = args.batch_size // args.accumulation_steps
	# test batch size
	if args.test_batch_size is None:
		args.test_batch_size = args.batch_size

	# learning rate
	if args.init_lr is None:
		lrs = {
			'coco': 0.1,
			'cityscapes': 0.01,
			'pascal': 0.007,
		}
		args.init_lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

	# checkpoint name
	if args.checkname is None:
		args.checkname = str(args.seg_model) + str(args.backbone)

	# some default parameters to ensure the justice of the experiments.
	if args.backbone in ['resnet101']:
		args.weight_decay = 1e-4
		args.bn_mom = 0.05
		if args.seg_model == 'deeplabv3':
			# the default setting for deeplabv3
			args.lr_multiplier = 10.0
		elif args.seg_model == 'deeplabv3+':
			# the default setting for deeplabv3+
			args.lr_multiplier = 5.0
		elif args.seg_model == 'pspnet':
			# the default setting for pspnet
			args.lr_multiplier = 10.0
		else:
			raise NotImplementedError
	else:
		args.weight_decay = 4e-5
		args.bn_mom = 0.0003

	# dataset related paramters
	if 'pascal' in args.dataset:
		args.slow_start_steps = 1500
	elif 'cityscapes' in args.dataset:
		args.slow_start_steps = 3000
	elif 'camvid' in args.dataset:
		args.slow_start_steps = 300
		args.init_lr = 0.025
		args.lr_multiplier = 10.0
	else:
		raise NotImplementedError
	return args


def save_hp_to_json(args):
	"""Save hyperparameters to a json file
	"""
	if args.freeze_bn is False:
		filename = os.path.join(args.model_dir, 'hparams01.json')
	else:
		filename = os.path.join(args.model_dir, 'hparams02.json')
	#hparams = FLAGS.flag_values_dict()
	hparams = vars(args)
	with open(filename, 'w') as f:
		json.dump(hparams, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="PyTorch Segmentation Model Training")
	args = add_parser_params(parser)
	print(args)