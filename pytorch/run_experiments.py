import sys
import os
from argparse import ArgumentParser
from urllib import request

from train import main


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


def run_experiments(exp_names, pretrained, datadir):
	for exp_name in exp_names:
		print("=========================================================================")
		print("Experiment:", exp_name)
		print("=========================================================================")

		if exp_name == "resnet50_baseline":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=imagenet_exp_40.pth'
				file_path = '../models/resnet50_baseline.pth'
				get_model(url, file_path, exp_name)

			main(datadir=datadir, batch_size=100, val_batch_size=125, num_workers=6, outdir="../models", lr=0.1,
				 wd=1e-4, warmup=5, num_epochs=90, resume=file_path if pretrained else None,
				 with_aux=False, aux_lr_scale=None, save_model=False)
		elif exp_name == "resnet50_ssal":
			if pretrained:
				url = 'https://cloud.dfki.de/owncloud/index.php/s/GnBNwtfkDZQRMeb/download?path=%2F&files=epoch_90_.pth'
				file_path = '../models/resnet50_ssal.pth'
				get_model(url, file_path, exp_name)

			main(datadir=datadir, batch_size=100, val_batch_size=125, num_workers=6, outdir="../models", lr=0.1,
				 wd=1e-4, warmup=5, num_epochs=90, resume=file_path if pretrained else None,
				 with_aux=True, aux_lr_scale=1, save_model=False)
		else:
			raise NotImplementedError(str(exp_name) + "is not a valid experiment identifier")


if __name__ == '__main__':
	"""
	Run experiments of specified identifiers (see above).
	"""
	parser = ArgumentParser()
	parser.add_argument('-e', '--experiment_names', nargs='+', required=True,
						help='Setup identifiers for experiments to run. Can give multiple ones.')
	parser.add_argument('--datadir', required=True,
						help='Dataset directory path, expects to contain ImageNet folder for Pytorch ImageNet Dataset.')
	parser.add_argument('--pretrained', required=False, action='store_true', default=False,
						help='Load a trained model, downloads if file not found.')
	opts = parser.parse_args(sys.argv[1:])
	run_experiments(opts.experiment_names, opts.pretrained, opts.datadir)
