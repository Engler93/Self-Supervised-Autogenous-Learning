import os.path as pt

import torch
from torch import nn
from torchvision.models import resnet50

from helpers import Normalizer
from helpers import PiecewiseLinear
from loader import make_loader
from trainer import Trainer
from torch.optim import SGD
from resnet_aux import resnet50_aux, Loss

from helper_pytorch import *

import argparse


def main(
		datadir,
		batch_size,
		val_batch_size,
		num_workers,
		outdir,
		lr,
		wd,
		warmup,
		num_epochs,
		resume,
		with_aux,
		aux_lr_scale,
		save_model
):
	device = torch.device('cuda:0')  # cuda device is required
	torch.cuda.set_device(device)

	# create dataloaders
	train = make_loader(datadir, 'train', batch_size, num_workers)
	val = make_loader(datadir, 'val', val_batch_size, num_workers)

	# lr is scaled linearly to original batch size of 256
	k = batch_size / 256
	lr = k * lr
	print("wd:", wd)
	if with_aux:
		# number of groups for the 3 SSAL branches
		num_coarse_classes = [200, 334, 500]
		# groupings to use on the 3 SSAL branches
		grouping = ['200_group_similar_v2', '334_group_similar_v2', '500_group_similar_v2']
		cats = create_coarse_data('imagenet', grouping)
		model = resnet50_aux(num_coarse_classes).to(device)
	else:
		cats = []
		model = resnet50()

	# update batchnorm momentum to reflect larger batch size
	bn_momentum = 0.1
	k_bn = batch_size / 32
	for m in model.modules():
		if isinstance(m, nn.BatchNorm2d):
			m.momentum = 1 - (1 - bn_momentum) ** k_bn

	if not with_aux:
		model = Normalizer(model).to(device)  # normalize images in network
		model = Unpacker(model)  # convert input from dict and output to dict

	print('Parameters:', count_parameters(model))

	if with_aux:
		output_keys = ['logits', 'aux1', 'aux2', 'aux3']
		model_params = {'params': [p for name, p in model.named_parameters()
								   if 'aux' not in name]}
		aux_params = {'params': [p for name, p in model.named_parameters()
								 if 'aux' in name], 'lr': lr * aux_lr_scale}

		optimizer = SGD([model_params, aux_params], lr=lr, momentum=0.9, weight_decay=wd)
		loss = Loss(cats=cats).to(device)
		metric = AuxAccuracyMetric(top_k=1, output_keys=output_keys, cats=cats, compute_combined=False,
								   exp_combination_factor=0.3)
		val_metric = AuxAccuracyMetric(top_k=1, output_keys=output_keys, cats=cats, compute_combined=True,
									   exp_combination_factor=0.3)
	else:
		optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
		loss = CrossEntropyLoss(output_key='output', target_key='label').to(device)
		metric = AuxAccuracyMetric(top_k=1, output_keys=['output'], cats=[], compute_combined=False)
		val_metric = AuxAccuracyMetric(top_k=1, output_keys=['output'], cats=[], compute_combined=False)

	policy = PiecewiseLinear(optimizer, [0, warmup - 1, num_epochs], [1 / warmup, 1, 0])

	trainer = Trainer(model, optimizer, loss, metric, policy, train, val, outdir=outdir,
					  snapshot_interval=num_epochs if save_model else None, quiet=False, val_metric=val_metric,
					  device=device)

	# restore state of given snapshot to resume training
	start_epoch = 0
	if resume is not None:
		print('loading', resume)
		state = torch.load(resume)
		model.load_state_dict(state['model_state'])
		start_epoch = state['epoch']
		policy.step(start_epoch, None)

	# train the model
	trainer.train(num_epochs, start_epoch=start_epoch)
	if resume:
		trainer.validate_epoch()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--datadir', required=True,
					    help='Dataset directory path, expects to contain ImageNet folder for Pytorch ImageNet Dataset.')
	parser.add_argument('--outdir', required=False, default='../models', help='Snapshot directory.')
	parser.add_argument('--batch_size', required=False, default=100, type=int, help='Batch size for training set.')
	parser.add_argument('--val_batch_size', required=False, default=125, type=int,
						help='Batch size for validation set.')
	parser.add_argument('--num_workers', required=False, default=6, type=int, help='Number of workers for the loader.')
	parser.add_argument('--lr', required=False, default=0.1, type=float,
						help='Learning rate. Will be modulated by piecewise linear schedule.')
	parser.add_argument('--warmup', required=False, default=5, type=int, help='Number of epochs for the warmup.')
	parser.add_argument('--wd', required=False, default=1e-4, type=float, help='L2 regularization aka weight decay.')
	parser.add_argument('--num_epochs', required=False, default=90, type=int, help='Number of training epochs.')
	parser.add_argument('--resume', required=False, default=None,
						help='Model file (.pth) to load. If None, train from scratch.')
	parser.add_argument('--with_aux', required=False, action='store_true', default=False,
						help='Use auxiliary (SSAL branches).')
	parser.add_argument('--aux_lr_scale', required=False, default=1, type=float,
						help='Custom scaling factor of learning rate on SSAL branches.')
	parser.add_argument('--save_model', required=False, action='store_true', default=False,
						help='Save the trained model.')

	args = parser.parse_args()

	main(args.datadir,
		 args.batch_size,
		 args.val_batch_size,
		 args.num_workers,
		 args.outdir,
		 args.lr,
		 args.wd,
		 args.warmup,
		 args.num_epochs,
		 args.resume,
		 args.with_aux,
		 args.aux_lr_scale,
		 args.save_model)
