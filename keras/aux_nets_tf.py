import os
import sys

from tensorflow.python.keras.layers import (
	Input,
	Activation,
	Dense,
	Flatten,
	GlobalAveragePooling2D,
	GlobalMaxPooling2D,
	Dropout,
	Lambda,
	Concatenate,
)
from tensorflow.python.keras.layers.convolutional import (
	Conv2D,
	MaxPooling2D,
	AveragePooling2D
)
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

import tensorflow as K

init = 'he_normal'


def _bn_relu(input):
	"""Helper to build a BN -> relu block
	"""
	norm = BatchNormalization(axis=3)(input)
	return Activation("relu")(norm)


def _conv_bn_relu(name=None, aux_weight_decay=3e-4, **conv_params):
	"""Helper to build a conv -> BN -> relu block
	"""
	filters = conv_params["filters"]
	kernel_size = conv_params["kernel_size"]
	strides = conv_params.setdefault("strides", (1, 1))
	kernel_initializer = conv_params.setdefault("kernel_initializer", init)
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(aux_weight_decay))

	def f(input):
		if name is None:
			conv = Conv2D(filters=filters, kernel_size=kernel_size,
						  strides=strides, padding=padding,
						  kernel_initializer=kernel_initializer,
						  kernel_regularizer=kernel_regularizer)(input)
		else:
			conv = Conv2D(filters=filters, kernel_size=kernel_size,
						  strides=strides, padding=padding,
						  kernel_initializer=kernel_initializer,
						  kernel_regularizer=kernel_regularizer, name=name)(input)
		return _bn_relu(conv)

	return f


def get_remap_to_fine_classes(cat):
	def remap_to_fine_classes(x):
		unit_list = []
		for elem in cat:
			unit_list.append(x[:, elem:elem + 1])
		return Concatenate()(unit_list)

	return remap_to_fine_classes


def inception_block_high_level(input, aux_weight_decay=3e-4):
	# Warning: axis was changed from default (-1) to 1 in concatenate layers. May have broken compatibility with previous setups.
	axis = -1
	branch1x1 = _conv_bn_relu(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same",
							  aux_weight_decay=aux_weight_decay)(input)

	branch3x3 = _conv_bn_relu(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same",
							  aux_weight_decay=aux_weight_decay)(input)
	branch3x3_1 = _conv_bn_relu(filters=128, kernel_size=(1, 3), strides=(1, 1), padding="same",
								aux_weight_decay=aux_weight_decay)(branch3x3)
	branch3x3_2 = _conv_bn_relu(filters=128, kernel_size=(3, 1), strides=(1, 1), padding="same",
								aux_weight_decay=aux_weight_decay)(branch3x3)
	branch3x3 = Concatenate(axis=axis)([branch3x3_1, branch3x3_2])

	branch3x3dbl = _conv_bn_relu(filters=160, kernel_size=(1, 1), strides=(1, 1), padding="same",
								 aux_weight_decay=aux_weight_decay)(input)
	branch3x3dbl = _conv_bn_relu(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same",
								 aux_weight_decay=aux_weight_decay)(branch3x3dbl)
	branch3x3dbl_1 = _conv_bn_relu(filters=128, kernel_size=(1, 3), strides=(1, 1), padding="same",
								   aux_weight_decay=aux_weight_decay)(branch3x3dbl)
	branch3x3dbl_2 = _conv_bn_relu(filters=128, kernel_size=(3, 1), strides=(1, 1), padding="same",
								   aux_weight_decay=aux_weight_decay)(branch3x3dbl)
	branch3x3dbl = Concatenate(axis=axis)([branch3x3dbl_1, branch3x3dbl_2])

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
	branch_pool = _conv_bn_relu(filters=108, kernel_size=(3, 1), strides=(1, 1), padding="same",
								aux_weight_decay=aux_weight_decay)(branch_pool)
	input = Concatenate(axis=axis)([branch1x1, branch3x3, branch3x3dbl, branch_pool])
	return input


def interpret_layout(aux_layer, input, id=-1, aux_weight_decay=3e-4):
	"""
	Interprets a layer from a given layout for an aux network and generates the specified layers.
	:param aux_layer: aux_layout from aux_layouts list (see cifar100aux.py)
	:param input: input layer of aux network.
	:return: output layer of specified layout (further layers may be attached later)
	"""
	if aux_layer[0] == 'cbr':
		filters = 256
		kernel_size = (3, 3)
		strides = (1, 1)
		if len(aux_layer) > 1:
			filters = aux_layer[1]
		if len(aux_layer) > 2:
			kernel_size = (aux_layer[2], aux_layer[2])
		if len(aux_layer) > 3:
			strides = (aux_layer[3], aux_layer[3])
		input = _conv_bn_relu(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
							  aux_weight_decay=aux_weight_decay)(input)
	elif aux_layer[0] == 'se':
		input = squeeze_and_excitation_layer(input, aux_weight_decay=aux_weight_decay)
	elif aux_layer[0] == 'inception':
		input = inception_block_high_level(input, aux_weight_decay=aux_weight_decay)
	elif aux_layer[0] == 'gap':
		input = GlobalAveragePooling2D(name='gap' + str(id))(input)
	elif aux_layer[0] == 'dense':
		units = 768
		if len(aux_layer) > 1:
			units = aux_layer[1]
		input = Dense(units, activation='relu')(input)
	else:
		raise ValueError
	return input


def aux_net(input, num_coarse_classes, id=0, aux_layout=None, aux_weight_decay=3e-4):
	"""
	Constructs an auxiliary network with a classifier.

	:param input: Layer that acts as input for the auxiliar network.
	:param num_coarse_classes: Number of categories in the auxiliary classifier. List, if multiple aux classifiers.
	:param id: Identifier for the aux net, should start with 0 for first aux net.
	:param standard: Use same network on any depth.
	:return: Output layer of the auxiliary network.

	@author: Philipp Engler
	"""
	for layer in aux_layout:
		input = interpret_layout(layer, input, id=id, aux_weight_decay=aux_weight_decay)

	input = Dense(num_coarse_classes, name='aux_output' + str(id), activation='softmax')(input)
	return input


def apply_aux(input, major_id, minor_id, curr_id=0, aux_depth=None, num_coarse_classes=None, aux_layouts=None,
			  aux_weight_decay=3e-4, initialization='he_normal'):
	global init
	init = initialization
	curr_id_temp = curr_id
	aux_list_temp = []
	if aux_depth is not None and num_coarse_classes is not None and aux_layouts is not None and len(
			num_coarse_classes) == len(aux_depth):
		for depth, nb_coarse, layout in zip(aux_depth, num_coarse_classes, aux_layouts):
			if depth[0] == major_id and depth[1] == minor_id:
				aux_list_temp.append(
					aux_net(input, nb_coarse,
							id=curr_id_temp, aux_layout=layout, aux_weight_decay=aux_weight_decay))
				curr_id_temp += 1
				print('Auxiliary network', str(curr_id_temp), 'added at', str(major_id) + '.' + str(minor_id),
					  'with input dimension:', str(input.get_shape().as_list()))
				# prevent list index out of range
				if curr_id_temp >= len(aux_depth):
					curr_id_temp = 0
	else:
		print('No auxiliary network added.')
	return aux_list_temp, curr_id_temp
