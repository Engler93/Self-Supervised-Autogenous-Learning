import tensorflow as tf
from tensorflow.python import keras
import time
import os
import sys
import numpy
import json

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import LearningRateScheduler, LambdaCallback
import tensorflow as K

from tensorflow.python.keras.layers import Dense, Concatenate, Input

from tensorflow.python.keras.utils import plot_model

from tensorflow.python.keras.datasets import cifar100

from ResNet_aux_tf import ResnetBuilder
from WideResNet_aux_tf import WideResidualNetwork
from DenseNet_aux_tf import DenseNet
from aux_nets_tf import *

from helper import *


def train(**kwargs):
	"""
	Train model with specified parameters. For example parameters / parameters used for the paper check
		"run_experiments.py".
	:param with_aux: Flag, whether auxiliary (SSAL) classifiers should be used.
	:param aux_weight: Weight of the auxiliary loss compared to the base loss. List, if multiple aux losses.
	:param aux_depth: List of positions for aux nets. A position is either a list of 2 integers or just one integer.
		List of two integers encode major and minor block position of aux net. It is placed in the major block after the
		given minor block. Minor 0 means typically before the first minor block. Minor -1 is the last position of a
		major block, if existent. It is typically behind the last minor block, but before pooling, if existent at the
		end or after the major block. A single integer defaults to the last position in the specified major block.
	:param batch_size: Batch size.
	:param epochs: Number of epochs of training.
	:param learning_rate: Maximum learning rate for the training.
	:param weight_decay: Weight decay for the training.
	:param momentum: Momentum for the training.
	:param num_coarse_classes: Number of coarse categories in the auxiliary classifier. List, if multiple aux
		classifiers. Must fit with the grouping specified in grouping parameter.
	:param save_model: Flag, whether model should be saved after training (in models folder). See param use_pretrained.
	:param max_epoch: Epoch in which learning rate reaches its peak (for slanted triangular learning rate).
	:param use_pretrained: Path to .h5 model to load.
	:param optimize: If true, use split of train set as validation set.
	:param network: Current options: 'resnet50', 'WRN', 'DenseNet'
	:param grouping: Identifier of the grouping to be loaded from txt to array. Labels file should list the coarse label
		for each fine label and be readable by numpy.loadtxt. File 'cifar100_' + <grouping> + '.txt' must exist
		in labels folder.
	:param aux_layouts: List of layouts for each aux network. A aux network layout is a list of blocks. A block is also
		a list. The first element defines the type of block. Currently implemented is 'inception' for an inception block
		and 'cbr' for a CBR layer. The following elements are optional and can further adjust the block's layout.
		Example: '[[['cbr'],['inception']],[['inception']]]' contains one CBR layer + one inception block in the first
			aux network and just one inception block in the second one.
	:param aux_weight_decay: If None, copied from weight_decay
	:param se_net: Include squeeze & excitation blocks. Currently only for WRN.
	:param mean_std_norm: If True, normalize dataset by mean and stddev of training set, else normalize images to [0,1].
	:param label_smoothing: If True, smooth ground truth logits s.t. the true label only has a value of 0.9.
	:return: None
	"""
	params = {'with_aux': False, 'aux_weight': 0.3, 'aux_depth': [], 'batch_size': 256, 'epochs': 1,
			  'learning_rate': 0.01, 'weight_decay': None, 'aux_weight_decay': None,
			  'momentum': 0.95, 'num_coarse_classes': 20, 'exp_combination_factor': 1.0,
			  'save_model': False, 'use_pretrained': None, 'max_epoch': 10,
			  'grouping': 'default', 'optimize': False, 'network': None, 'aux_layouts': None,
			  'wide_width': 10, 'wide_depth': 28, 'se_net': False, 'dense_depth': 100, 'dense_growth': 12,
			  'nesterov': False, 'mean_std_norm': False, 'label_smoothing': False
			  }
	params.update(kwargs)
	
	# create explicit session as possible solution against memory leak during meta parameter exploration
	cfg = tf.ConfigProto()
	cfg.gpu_options.allow_growth = True
	tf.Session(config=cfg)

	grouping = params['grouping']
	if not isinstance(grouping, list):
		grouping = [grouping]

	hyper = params
	aux_depth = params['aux_depth']
	if isinstance(aux_depth, int):
		aux_depth = [aux_depth]
	aux_depth = aux_depth.copy()
	for i in range(len(aux_depth)):
		if not isinstance(aux_depth[i], list):
			aux_depth[i] = (int(aux_depth[i]), -1)

	with_aux = hyper['with_aux']
	num_auxs = len(aux_depth)

	max_epoch = params['max_epoch']  # epoch in which learning rate reaches it's maximum value

	batch_size = hyper['batch_size']
	epochs = hyper['epochs']
	aux_weight = hyper['aux_weight']
	if not isinstance(aux_weight, list):
		aux_weight = [aux_weight]

	aux_weight_decay = params['aux_weight_decay']
	if aux_weight_decay is None:
		aux_weight_decay = params['weight_decay']

	learning_rate = hyper['learning_rate']  # maximum learning rate

	num_coarse_classes = hyper['num_coarse_classes']
	if not isinstance(num_coarse_classes, list):
		num_coarse_classes = [num_coarse_classes]

	aux_layouts = hyper['aux_layouts']

	if with_aux:
		if not isinstance(aux_layouts[0][0], list):
			raise TypeError('Bad aux_layouts format. Expect list with 2-element list for each SSAL branch')
		if aux_layouts is not None:
			# repeat last aux_layout if more positions specified than layouts
			if len(aux_layouts) < len(aux_depth):
				while len(aux_layouts) < len(aux_depth):
					aux_layouts.append(aux_layouts[-1])

	num_classes = 100

	#  load cifar100 data and create train set and test set
	(x_train, y_train), (x_test, y_test) = cifar100.load_data()

	if params['optimize']:
		len_val = round(0.85 * len(y_train))
		x_test = x_train[len_val:]
		x_train = x_train[:len_val]
		y_test = y_train[len_val:]
		y_train = y_train[:len_val]

	print('x_train:', x_train.shape)
	print('y_train:', y_train.shape)
	print('x_test:', x_test.shape)
	print('y_test:', y_test.shape)

	# (x_train_c, y_train_c), (x_test_c, y_test_c) = cifar100.load_data(label_mode='coarse')  # only used to map classes
	steps_per_epoch = int(len(x_train) / batch_size)

	# logs custom combined accuracy
	combined_accuracy_log = []

	# create coarse labels
	cats = []
	y_train_c = []
	y_test_c = []
	if with_aux:
		cats, y_train_c, y_test_c = create_coarse_data(y_train, y_test, 'cifar100', grouping)

	# prepares output for categorical crossentropy
	if not params['label_smoothing']:
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)
		if with_aux:
			for i in range(0, num_auxs):
				y_train_c[i] = keras.utils.to_categorical(y_train_c[i], num_coarse_classes[i])
				y_test_c[i] = keras.utils.to_categorical(y_test_c[i], num_coarse_classes[i])

	else:
		y_train = to_categorical_smooth(y_train, num_classes, temperature=0.1)
		y_test = to_categorical_smooth(y_test, num_classes, temperature=0.1)
		if with_aux:
			for i in range(0, num_auxs):
				y_train_c[i] = to_categorical_smooth(y_train_c[i], num_coarse_classes[i], temperature=0.1)
				y_test_c[i] = to_categorical_smooth(y_test_c[i], num_coarse_classes[i], temperature=0.1)

	# normalize pixel values
	if not params['mean_std_norm']:
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255
	else:
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		avg = numpy.average(x_train, axis=(0, 1, 2))
		stddev = numpy.std(x_train, axis=(0, 1, 2))
		x_train = (x_train - avg) / stddev
		x_test = (x_test - avg) / stddev
	weight_decay = params['weight_decay']

	use_pretrained = params['use_pretrained']
	if use_pretrained is not None:
		# load .h5 model
		model = load_model(use_pretrained)
	else:
		if params['network'] == 'resnet50':
			if with_aux:
				model = ResnetBuilder.build_resnet_50(x_train.shape[1:], num_classes, aux_depth=aux_depth,
													  num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts)
			else:
				model = ResnetBuilder.build_resnet_50(x_train.shape[1:], num_classes)
		elif params['network'] == 'WRN':
			depth = params['wide_depth']
			width = params['wide_width']
			if with_aux:
				model = WideResidualNetwork(depth=depth, width=width, input_shape=x_train.shape[1:],
											classes=num_classes, activation='softmax', aux_depth=aux_depth,
											num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
											aux_weight_decay=aux_weight_decay, se_net=params['se_net'],
											aux_init='he_normal')
			else:
				model = WideResidualNetwork(depth=depth, width=width, input_shape=x_train.shape[1:],
											classes=num_classes, activation='softmax', aux_depth=[],
											num_coarse_classes=[], aux_layouts=[], se_net=params['se_net'])

			if weight_decay is not None:
				# manually add weight decay as loss
				for layer in model.layers:
					if len(layer.losses) == 0:
						if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
							layer.add_loss(keras.regularizers.l2(weight_decay)(layer.kernel))
						if hasattr(layer, 'bias_regularizer') and layer.use_bias:
							layer.add_loss(keras.regularizers.l2(weight_decay)(layer.bias))
		elif params['network'] == 'DenseNet':
			depth = params['dense_depth']
			growth = params['dense_growth']

			decay = weight_decay if weight_decay is not None else 0
			# 100-12, 250-24, 190-40
			if with_aux:
				model = DenseNet(input_shape=x_train.shape[1:],
								 depth=depth,  # L parameter / Depth parameter
								 growth_rate=growth,  # k parameter
								 bottleneck=True,  # True for DenseNet-B
								 reduction=0.5,  # 1-theta (1-compression), >0.0 for DenseNet-C
								 subsample_initial_block=False,  # Keep false for CIFAR
								 weight_decay=decay,
								 classes=num_classes, aux_depth=aux_depth,
								 num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
								 initialization='orthogonal',
								 aux_init='he_normal')

			else:
				model = DenseNet(input_shape=x_train.shape[1:],
								 depth=depth,  # L parameter / Depth parameter
								 growth_rate=growth,  # k parameter
								 bottleneck=True,  # True for DenseNet-B
								 reduction=0.5,  # 1-theta (1-compression), >0.0 for DenseNet-C
								 subsample_initial_block=False,  # Keep false for CIFAR
								 weight_decay=decay,
								 classes=num_classes, aux_depth=[],
								 num_coarse_classes=[], aux_layouts=[], initialization='orthogonal')
		else:
			raise NotImplementedError("Unknown Model: " + str(params['network']))

		# stochastic gradient descent
		sgd = SGD(lr=hyper['learning_rate'], momentum=hyper['momentum'], nesterov=params['nesterov'])

		print('Free parameters:', model.count_params())

		# plot_model(model, to_file='resnet50_1aux_model.pdf', show_shapes=True)

		update_lr = LearningRateScheduler(lambda epoch, lr: lr_scheduler(epoch, epochs, learning_rate, max_epoch))

		if with_aux:
			model.compile(optimizer=sgd, loss='categorical_crossentropy', loss_weights=[1.0] + aux_weight,
						  metrics=['accuracy'])
		else:
			model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

		log_call = LambdaCallback(
			on_epoch_end=lambda epoch, logs: log_combined_acc(model, x_test, y_test, cats, num_classes,
															  combined_accuracy_log, params['exp_combination_factor']),
			on_train_end=lambda logs: classification_analysis(model, x_test, y_test, cats,
															  num_classes))  # called during training after each epoch

		start_time = time.time()

		datagen = ImageDataGenerator(width_shift_range=5, height_shift_range=5, horizontal_flip=True,
									 fill_mode='reflect')
		# train
		if with_aux:
			model.fit(
				generator_for_multi_outputs(datagen, x_train, [y_train] + y_train_c, batch_size=batch_size),
				epochs=epochs, validation_data=(x_test, [y_test] + y_test_c), callbacks=[update_lr, log_call],
				verbose=1, steps_per_epoch=steps_per_epoch, shuffle=True)
		else:
			model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, shuffle=True,
								validation_data=(x_test, y_test), callbacks=[update_lr],
								steps_per_epoch=steps_per_epoch, verbose=1)

		end_time = time.time()
		print('duration', end_time - start_time, 's')
		if params['save_model']:
			model.save('../models/cifar100_model.h5')


	# evaluate and print results
	if with_aux:
		score = model.evaluate(x_test, [y_test] + y_test_c, batch_size=batch_size)
		print('Test acc (fine):', score[2 + num_auxs])
		for i in range(0, num_auxs):
			print('Test acc (SSAL ' + str(i) + '):', score[3 + num_auxs + i])
		log_combined_acc(model, x_test, y_test, cats, num_classes, [],
						 exp_combination_factor=params['exp_combination_factor'])  # also print combined acc!
	else:
		score = model.evaluate(x_test, y_test, batch_size=batch_size)
		print('Test acc:', score[1])


	del model
