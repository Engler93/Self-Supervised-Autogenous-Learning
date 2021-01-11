import sys
import os
from argparse import ArgumentParser
from urllib import request

from train import train


def get_model(url, file_path, exp_name):
	"""
	Check if model is found at filepath otherwise download from url.
	"""
	if not os.path.exists('../models'):
		os.makedirs('../models')
	if not os.path.isfile(file_path):
		print("downloading pretrained model for " + exp_name + "...")
		request.urlretrieve(url, file_path)
		print("pretrained model downloaded to " + file_path)


def run_experiments(exp_names, pretrained):
	for exp_name in exp_names:
		print("=========================================================================")
		print("Experiment:", exp_name)
		print("=========================================================================")

		if exp_name == "resnet50_baseline":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=cifar100_exp_3282.h5'
				file_path = '../models/resnet50_baseline.h5'
				get_model(url, file_path, exp_name)

			updates = {'epochs': 80, 'learning_rate': 0.088, 'momentum': 0.93,
					   'aux_weight': [], 'aux_depth': [],
					   'grouping': [],
					   'num_coarse_classes': [], 'with_aux': False, 'optimize': False,
					   'aux_layouts': [],
					   'with_augmentation': True, 'batch_size': 64, 'network': 'resnet50',
					   'use_pretrained': file_path if pretrained else None
					   }
		elif exp_name == "resnet50_ssal":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=cifar100_exp_676.h5'
				file_path = '../models/resnet50_ssal.h5'
				get_model(url, file_path, exp_name)

			updates = {'epochs': 80, 'learning_rate': 0.092, 'momentum': 0.93,
					   'aux_weight': [0.42, 0.59, 0.76], 'aux_depth': [[1, -1], [2, -1], [3, -1]],
					   'grouping': ['20_group_similar_v2', '33_group_similar_v2', '50_group_similar_v2'],
					   'num_coarse_classes': [20, 33, 50], 'with_aux': True, 'optimize': False,
					   'aux_layouts': [
						   [['cbr', 128, 5, 2], ['cbr', 128, 3], ['inception'], ['gap']],
						   [['cbr', 128, 5], ['cbr', 128, 3], ['inception'], ['gap']],
						   [['cbr', 128, 3], ['cbr', 128, 3], ['inception'], ['gap']]],
					   'with_augmentation': True, 'batch_size': 64, 'network': 'resnet50',
					   'exp_combination_factor': 0.3,
					   'use_pretrained': file_path if pretrained else None
					   }
		elif exp_name == "wrn28-10_baseline":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=cifar100_exp_3060.h5'
				file_path = '../models/wrn28-10_baseline.h5'
				get_model(url, file_path, exp_name)

			updates = {'epochs': 80, 'learning_rate': 0.016, 'momentum': 0.9, 'weight_decay': 7e-4,
					   'aux_weight': [], 'aux_depth': [],
					   'grouping': [],
					   'num_coarse_classes': [], 'with_aux': False, 'optimize': False,
					   'aux_layouts': [],
					   'with_augmentation': True, 'batch_size': 64, 'network': 'WRN',
					   'wide_depth': 28, 'wide_width': 10, 'mean_std_norm': True, 'nesterov': True,
					   'use_pretrained': file_path if pretrained else None
					   }
		elif exp_name == "wrn28-10_ssal":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=cifar100_exp_3421.h5'
				file_path = '../models/wrn28-10_ssal.h5'
				get_model(url, file_path, exp_name)

			updates = {'epochs': 80, 'learning_rate': 0.021, 'momentum': 0.9, 'weight_decay': 7e-4,
					   'aux_weight': [0.8, 0.9, 1.0], 'aux_depth': [[2, 2], [3, 2], [3, -1]],
					   'grouping': ['20_group_similar_v2', '33_group_similar_v2', '50_group_similar_v2'],
					   'num_coarse_classes': [20, 33, 50], 'with_aux': True, 'optimize': False,
					   'aux_layouts': [
						   [['cbr', 128, 5, 2], ['cbr', 128, 3], ['cbr'], ['cbr'], ['inception'],
							['inception'], ['gap']],
						   [['cbr', 128, 5], ['cbr', 128, 3], ['cbr'], ['cbr'], ['gap'], ['dense']],
						   [['cbr', 128, 3], ['cbr', 128, 3], ['inception'], ['inception'], ['gap']]],
					   'with_augmentation': True, 'batch_size': 64, 'network': 'WRN',
					   'wide_depth': 28, 'wide_width': 10, 'aux_weight_decay': 3e-4, 'mean_std_norm': True,
					   'nesterov': True, 'exp_combination_factor': 0.4,
					   'use_pretrained': file_path if pretrained else None
					   }
		elif exp_name == "densenet190-40_baseline":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=cifar100_exp_153.h5'
				file_path = '../models/densenet190-40_baseline.h5'
				get_model(url, file_path, exp_name)

			updates = {'epochs': 100, 'learning_rate': 0.08, 'momentum': 0.9, 'weight_decay': 3e-4,
					   'aux_weight': [], 'aux_depth': [],
					   'grouping': [],
					   'num_coarse_classes': [], 'with_aux': False, 'optimize': False,
					   'aux_layouts': [],
					   'with_augmentation': True, 'batch_size': 32, 'network': 'DenseNet',
					   'dense_depth': 190, 'dense_growth': 40,
					   'mean_std_norm': True, 'nesterov': True, 'label_smoothing': True,
					   'use_pretrained': file_path if pretrained else None
					   }
		elif exp_name == "densenet190-40_ssal":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=cifar100_exp_55.h5'
				file_path = '../models/densenet190-40_ssal.h5'
				get_model(url, file_path, exp_name)

			updates = {'epochs': 100, 'learning_rate': 0.08, 'momentum': 0.9, 'weight_decay': 3e-4,
					   'aux_weight': [0.8, 0.9, 1.0], 'aux_depth': [[1, -1], [2, 0], [3, 0]],
					   'grouping': ['20_group_similar_v2', '33_group_similar_v2', '50_group_similar_v2'],
					   'num_coarse_classes': [20, 33, 50], 'with_aux': True, 'optimize': False,
					   'aux_layouts': [
						   [['cbr', 128, 5, 2], ['cbr', 128, 3], ['cbr'], ['cbr'], ['inception'],
							['inception'], ['gap']],
						   [['cbr', 128, 5], ['cbr', 128, 3], ['cbr'], ['cbr'], ['gap'], ['dense']],
						   [['cbr', 128, 3], ['cbr', 128, 3], ['inception'], ['inception'], ['gap']]],
					   'with_augmentation': True, 'batch_size': 32, 'network': 'DenseNet',
					   'dense_depth': 190, 'dense_growth': 40, 'aux_weight_decay': 1e-4,
					   'mean_std_norm': True, 'nesterov': True, 'label_smoothing': True,
					   'exp_combination_factor': 0.5,
					   'use_pretrained': file_path if pretrained else None
					   }
		else:
			raise NotImplementedError(str(exp_name) + "is not a valid experiment identifier")

		train(**updates)


if __name__ == '__main__':
	"""
	Run experiments of specified identifiers (see above).
	"""
	parser = ArgumentParser()
	parser.add_argument('-e', '--experiment_names', nargs='+',
						help='Setup identifiers for experiments to run. Can give multiple ones.')
	parser.add_argument('--pretrained', required=False, action='store_true', default=False,
						help='Load a trained model, downloads if file not found.')
	opts = parser.parse_args(sys.argv[1:])
	run_experiments(opts.experiment_names, opts.pretrained)
