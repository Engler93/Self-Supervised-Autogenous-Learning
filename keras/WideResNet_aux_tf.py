""" Adapted from: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/wide_resnet.py

Wide Residual Network models for Keras.
# Reference
- [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input, Conv2D, Concatenate, Lambda, Multiply, Add
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.utils.layer_utils import convert_all_kernels_in_model
from tensorflow.python.keras.utils.data_utils import get_file
import tensorflow as K

from aux_nets_tf import *
from helper import obtain_input_shape

TH_WEIGHTS_PATH = ('https://github.com/titu1994/Wide-Residual-Networks/'
				   'releases/download/v1.2/wrn_28_8_th_kernels_th_dim_ordering.h5')
TF_WEIGHTS_PATH = ('https://github.com/titu1994/Wide-Residual-Networks/'
				   'releases/download/v1.2/wrn_28_8_tf_kernels_tf_dim_ordering.h5')
TH_WEIGHTS_PATH_NO_TOP = ('https://github.com/titu1994/Wide-Residual-Networks/releases/'
						  'download/v1.2/wrn_28_8_th_kernels_th_dim_ordering_no_top.h5')
TF_WEIGHTS_PATH_NO_TOP = ('https://github.com/titu1994/Wide-Residual-Networks/releases/'
						  'download/v1.2/wrn_28_8_tf_kernels_tf_dim_ordering_no_top.h5')

global initia
global aux_initia


def squeeze_and_excitation_layer(lst_layer, r=16, activation='relu'):
	"""
	Squeeze and excitation layer.
	:param lst_layer: keras layer. Layer to append the SE block to.
	:param r: int. Ratio to squeeze the number of channels as defined in the original paper (default: 16)
	:param activation: str. Name of non-linear activation to use. ReLU is the default one.
	:return: keras layer. Next layer (output)
	"""
	num_channels = int(lst_layer.get_shape()[-1])
	gap = GlobalAveragePooling2D()(lst_layer)
	reduct = Dense(num_channels // r, activation=activation, kernel_initializer='orthogonal')(gap)
	expand = Dense(num_channels, activation='sigmoid', kernel_initializer='orthogonal')(reduct)
	return Multiply()([lst_layer, expand])  # Broadcast adds dimensions at the beginning of expand


def WideResidualNetwork(depth=28, width=8, dropout_rate=0.0,
						include_top=True, weights=None,
						input_tensor=None, input_shape=None,
						classes=10, activation='softmax', aux_depth=None, num_coarse_classes=None, aux_layouts=None,
						aux_weight_decay=3e-4, se_net=False, initialization='glorot_uniform', aux_init='he_normal'):
	"""Instantiate the Wide Residual Network architecture,
		optionally loading weights pre-trained
		on CIFAR-10. Note that when using TensorFlow,
		for best performance you should set
		`image_dim_ordering="tf"` in your Keras config
		at ~/.keras/keras.json.
		The model and the weights are compatible with both
		TensorFlow and Theano. The dimension ordering
		convention used by the model is the one
		specified in your Keras config file.
		# Arguments
			depth: number or layers in the DenseNet
			width: multiplier to the ResNet width (number of filters)
			dropout_rate: dropout rate
			include_top: whether to include the fully-connected
				layer at the top of the network.
			weights: one of `None` (random initialization) or
				"cifar10" (pre-training on CIFAR-10)..
			input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
				to use as image input for the model.
			input_shape: optional shape tuple, only to be specified
				if `include_top` is False (otherwise the input shape
				has to be `(32, 32, 3)` (with `tf` dim ordering)
				or `(3, 32, 32)` (with `th` dim ordering).
				It should have exactly 3 inputs channels,
				and width and height should be no smaller than 8.
				E.g. `(200, 200, 3)` would be one valid value.
			classes: optional number of classes to classify images
				into, only to be specified if `include_top` is True, and
				if no `weights` argument is specified.
		# Returns
			A Keras model instance.
		"""

	if weights not in {'cifar10', None}:
		raise ValueError('The `weights` argument should be either '
						 '`None` (random initialization) or `cifar10` '
						 '(pre-training on CIFAR-10).')

	if weights == 'cifar10' and include_top and classes != 10:
		raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
						 ' as true, `classes` should be 10')

	if (depth - 4) % 6 != 0:
		raise ValueError('Depth of the network must be such that (depth - 4)'
						 'should be divisible by 6.')

	# Determine proper input shape
	input_shape = obtain_input_shape(input_shape,
									 default_size=32,
									 min_size=8,
									 data_format='tf',
									 require_flatten=include_top)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	global initia
	initia = initialization
	global aux_initia
	aux_initia = aux_init

	model = __create_wide_residual_network(classes, img_input, include_top, depth, width,
										   dropout_rate, activation, aux_depth=aux_depth,
										   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
										   aux_weight_decay=aux_weight_decay, se_net=se_net)

	return model


def __conv1_block(input):
	x = Conv2D(16, (3, 3), padding='same', kernel_initializer=initia)(input)

	channel_axis = 1 if 'channels_last' == 'channels_first' else -1

	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)
	return x


def __conv2_block(input, k=1, dropout=0.0, se_net=False):
	init = input

	channel_axis = 1 if 'channels_last' == 'channels_first' else -1

	# Check if input number of filters is same as 16 * k, else create
	# convolution2d for this input
	if 'channels_last' == 'channels_first':
		if input.get_shape().as_list()[-1] != 16 * k:
			init = Conv2D(16 * k, (1, 1), activation='linear', padding='same', kernel_initializer=initia)(init)
	else:
		if input.get_shape().as_list()[-1] != 16 * k:
			init = Conv2D(16 * k, (1, 1), activation='linear', padding='same', kernel_initializer=initia)(init)

	x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer=initia)(input)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if dropout > 0.0:
		x = Dropout(dropout)(x)

	x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer=initia)(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if se_net:
		x = squeeze_and_excitation_layer(x)

	m = add([init, x])
	return m


def __conv3_block(input, k=1, dropout=0.0, se_net=False):
	init = input

	channel_axis = 1 if 'channels_last' == 'channels_first' else -1

	# Check if input number of filters is same as 32 * k, else
	# create convolution2d for this input
	if 'channels_last' == 'channels_first':
		if input.get_shape().as_list()[-1] != 32 * k:
			init = Conv2D(32 * k, (1, 1), activation='linear', padding='same', kernel_initializer=initia)(init)
	else:
		if input.get_shape().as_list()[-1] != 32 * k:
			init = Conv2D(32 * k, (1, 1), activation='linear', padding='same', kernel_initializer=initia)(init)

	x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer=initia)(input)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if dropout > 0.0:
		x = Dropout(dropout)(x)

	x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer=initia)(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if se_net:
		x = squeeze_and_excitation_layer(x)

	m = add([init, x])
	return m


def ___conv4_block(input, k=1, dropout=0.0, se_net=False):
	init = input

	channel_axis = 1 if 'tf' == 'th' else -1

	# Check if input number of filters is same as 64 * k, else
	# create convolution2d for this input
	if 'tf' == 'th':
		if input.get_shape().as_list()[-1] != 64 * k:
			init = Conv2D(64 * k, (1, 1), activation='linear', padding='same', kernel_initializer=initia)(init)
	else:
		if input.get_shape().as_list()[-1] != 64 * k:
			init = Conv2D(64 * k, (1, 1), activation='linear', padding='same', kernel_initializer=initia)(init)

	x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer=initia)(input)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if dropout > 0.0:
		x = Dropout(dropout)(x)

	x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer=initia)(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	if se_net:
		x = squeeze_and_excitation_layer(x)

	m = add([init, x])
	return m


def __create_wide_residual_network(nb_classes, img_input, include_top=True, depth=28,
								   width=8, dropout=0.0, activation='softmax', aux_depth=None, num_coarse_classes=None,
								   aux_layouts=None, aux_weight_decay=3e-4, se_net=False):
	''' Creates a Wide Residual Network with specified parameters
	Args:
		nb_classes: Number of output classes
		img_input: Input tensor or layer
		include_top: Flag to include the last dense layer
		depth: Depth of the network. Compute N = (n - 4) / 6.
			   For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
			   For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
			   For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
		width: Width of the network.
		dropout: Adds dropout if value is greater than 0.0
	Returns:a Keras Model
	'''
	curr_id = 0
	aux_list = []

	N = (depth - 4) // 6

	# AUX1.0
	aux_list_temp, curr_id = apply_aux(img_input, major_id=1, minor_id=0, curr_id=curr_id, aux_depth=aux_depth,
									   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
									   aux_weight_decay=aux_weight_decay, initialization=aux_initia)
	aux_list = aux_list + aux_list_temp

	x = __conv1_block(img_input)
	nb_conv = 4

	for i in range(N):
		# AUX2.i
		aux_list_temp, curr_id = apply_aux(x, major_id=2, minor_id=i, curr_id=curr_id, aux_depth=aux_depth,
										   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
										   aux_weight_decay=aux_weight_decay, initialization=aux_initia)
		aux_list = aux_list + aux_list_temp

		x = __conv2_block(x, width, dropout, se_net=se_net)
		nb_conv += 2

	# AUX2.-1
	aux_list_temp, curr_id = apply_aux(x, major_id=2, minor_id=-1, curr_id=curr_id, aux_depth=aux_depth,
									   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
									   aux_weight_decay=aux_weight_decay, initialization=aux_initia)
	aux_list = aux_list + aux_list_temp

	x = MaxPooling2D((2, 2))(x)

	for i in range(N):
		# AUX3.i
		aux_list_temp, curr_id = apply_aux(x, major_id=3, minor_id=i, curr_id=curr_id, aux_depth=aux_depth,
										   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
										   aux_weight_decay=aux_weight_decay, initialization=aux_initia)
		aux_list = aux_list + aux_list_temp

		x = __conv3_block(x, width, dropout, se_net=se_net)
		nb_conv += 2

	# AUX3.-1
	aux_list_temp, curr_id = apply_aux(x, major_id=3, minor_id=-1, curr_id=curr_id, aux_depth=aux_depth,
									   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
									   aux_weight_decay=aux_weight_decay, initialization=aux_initia)
	aux_list = aux_list + aux_list_temp

	x = MaxPooling2D((2, 2))(x)

	for i in range(N):
		# AUX4.i
		aux_list_temp, curr_id = apply_aux(x, major_id=4, minor_id=i, curr_id=curr_id, aux_depth=aux_depth,
										   num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts,
										   aux_weight_decay=aux_weight_decay, initialization=aux_initia)
		aux_list = aux_list + aux_list_temp

		x = ___conv4_block(x, width, dropout, se_net=se_net)
		nb_conv += 2

	if include_top:
		x = GlobalAveragePooling2D()(x)

		x = Dense(nb_classes, activation=activation, name='output', kernel_initializer=initia)(x)

		if len(aux_depth) > 0:
			model = Model(inputs=img_input, outputs=[x] + aux_list)
		else:
			model = Model(inputs=img_input, outputs=x)

	return model
