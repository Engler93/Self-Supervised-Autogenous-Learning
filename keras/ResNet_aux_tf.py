"""
DESCRIPTION: Implementation of ResNet-18.
Adapted from "github.com/raghakot/keras-resnet", added auxiliary network option and options for different modifications.

COPYRIGHT of original code:

All contributions by Raghavendra Kotikalapudi:
Copyright (c) 2016, Raghavendra Kotikalapudi.
All rights reserved.

All other contributions:
Copyright (c) 2016, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from __future__ import division

import tensorflow as tf

import six
from tensorflow.python.keras.models import Model
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
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2
import tensorflow as K

from aux_nets_tf import *


def _bn_relu(input):
	"""Helper to build a BN -> relu block
	"""
	norm = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.65)(input)
	return Activation("relu")(norm)


def _conv_bn_relu(name=None, weight_decay=1e-4, **conv_params):
	"""Helper to build a conv -> BN -> relu block
	"""
	filters = conv_params["filters"]
	kernel_size = conv_params["kernel_size"]
	strides = conv_params.setdefault("strides", (1, 1))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(weight_decay))

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


def _bn_relu_conv(weight_decay=1e-4, **conv_params):
	"""Helper to build a BN -> relu -> conv block.
	This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
	"""
	filters = conv_params["filters"]
	kernel_size = conv_params["kernel_size"]
	strides = conv_params.setdefault("strides", (1, 1))
	kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
	padding = conv_params.setdefault("padding", "same")
	kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(weight_decay))

	def f(input):
		activation = _bn_relu(input)
		return Conv2D(filters=filters, kernel_size=kernel_size,
					  strides=strides, padding=padding,
					  kernel_initializer=kernel_initializer,
					  kernel_regularizer=kernel_regularizer)(activation)

	return f


def _shortcut(input, residual, weight_decay=1e-4):
	"""Adds a shortcut between input and residual block and merges them with "sum"
	"""
	# Expand channels of shortcut to match residual.
	# Stride appropriately to match residual (width, height)
	# Should be int if network architecture is correctly configured.
	input_shape = input.get_shape().as_list()
	residual_shape = residual.get_shape().as_list()
	stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))  # Problem for variable input
	stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
	equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

	shortcut = input
	# 1 X 1 conv if shape is different. Else identity.
	if stride_width > 1 or stride_height > 1 or not equal_channels:
		shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
						  kernel_size=(1, 1),
						  strides=(stride_width, stride_height),
						  padding="valid",
						  kernel_initializer="he_normal",
						  kernel_regularizer=l2(weight_decay))(input)

	return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False, keep_size=False, block_id=None,
					curr_id=0, aux_depth=None, num_coarse_classes=None, aux_layouts=None):
	"""Builds a residual block with repeating bottleneck blocks.
	"""

	def f(input):
		curr_id_temp = curr_id
		aux_list_temp = []
		for i in range(repetitions):
			# add aux net, if correct position reached
			aux_list_temp_temp, curr_id_temp = apply_aux(input, block_id, i, curr_id=curr_id_temp, aux_depth=aux_depth,
														 num_coarse_classes=num_coarse_classes,
														 aux_layouts=aux_layouts, aux_weight_decay=1e-4)
			aux_list_temp = aux_list_temp + aux_list_temp_temp

			init_strides = (1, 1)
			if i == 0 and not is_first_layer and not keep_size:
				init_strides = (2, 2)
			input = block_function(filters=filters, init_strides=init_strides,
								   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)

		aux_list_temp_temp, curr_id_temp = apply_aux(input, block_id, -1, curr_id=curr_id_temp, aux_depth=aux_depth,
													 num_coarse_classes=num_coarse_classes,
													 aux_layouts=aux_layouts, aux_weight_decay=1e-4)
		aux_list_temp = aux_list_temp + aux_list_temp_temp
		return input, aux_list_temp, curr_id_temp

	return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
	"""Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
	Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
	"""

	def f(input):

		if is_first_block_of_first_layer:
			# don't repeat bn->relu since we just did bn->relu->maxpool
			conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
						   strides=init_strides,
						   padding="same",
						   kernel_initializer="he_normal",
						   kernel_regularizer=l2(1e-4))(input)
		else:
			conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
								  strides=init_strides)(input)

		residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
		return _shortcut(input, residual)

	return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
	"""Bottleneck architecture for > 34 layer resnet.
	Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
	Returns:
		A final conv layer of filters * 4
	"""

	def f(input):

		if is_first_block_of_first_layer:
			# don't repeat bn->relu since we just did bn->relu->maxpool
			conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
							  strides=init_strides,
							  padding="same",
							  kernel_initializer="he_normal",
							  kernel_regularizer=l2(1e-4))(input)
		else:
			conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
									 strides=init_strides)(input)

		conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
		residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
		# residual = squeeze_and_excitation_layer(residual)
		return _shortcut(input, residual)

	return f


def _handle_dim_ordering():
	global ROW_AXIS
	global COL_AXIS
	global CHANNEL_AXIS
	if 'tf' == 'tf':
		ROW_AXIS = 1
		COL_AXIS = 2
		CHANNEL_AXIS = 3


def _get_block(identifier):
	if isinstance(identifier, six.string_types):
		res = globals().get(identifier)
		if not res:
			raise ValueError('Invalid {}'.format(identifier))
		return res
	return identifier


class ResnetBuilder(object):
	@staticmethod
	def build(input_shape, num_classes, block_fn, repetitions, aux_depth=[], num_coarse_classes=[],
			  min_size=8, aux_layouts=None):
		"""Builds a custom ResNet like architecture with optional aux network.
		Args:
			input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
			num_classes: The number of outputs at final softmax layer
			block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
				The original paper used basic_block for layers < 50
			repetitions: Number of repetitions of various block units.
				At each block unit, the number of filters are doubled and the input size is halved
			aux_depth: Position of aux network in the ResNet. 0 before first residual block, n>0 after n-th residual
				block. List, if multiple aux nets.
			num_coarse_classes: Number of categories in the auxiliary classifier. List, if multiple aux classifiers.
		Returns:
			The keras `Model`.
		"""
		_handle_dim_ordering()
		if len(input_shape) != 3:
			raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

		# Load function from str if needed.
		block_fn = _get_block(block_fn)

		input = Input(shape=input_shape)  # (None,None,3) for variable input
		conv1 = _conv_bn_relu(filters=64, kernel_size=(5, 5), strides=(1, 1))(input)

		# pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1) # todo remove again
		block = conv1
		# block = pool1 # todo swap again
		filters = 64

		curr_id = 0
		aux_list = []

		for i, r in enumerate(repetitions):
			if min_size is None or block.get_shape().as_list()[1] >= 2 * min_size:
				block, aux_list_temp, curr_id = _residual_block(block_fn, filters=filters, repetitions=r,
																is_first_layer=(i == 0),
																keep_size=False, block_id=i + 1, curr_id=curr_id,
																aux_depth=aux_depth,
																num_coarse_classes=num_coarse_classes,
																aux_layouts=aux_layouts)(block)
			else:
				block, aux_list_temp, curr_id = _residual_block(block_fn, filters=filters, repetitions=r,
																is_first_layer=(i == 0),
																keep_size=True, block_id=i + 1, curr_id=curr_id,
																aux_depth=aux_depth,
																num_coarse_classes=num_coarse_classes,
																aux_layouts=aux_layouts)(block)
			aux_list = aux_list + aux_list_temp
			filters *= 2

		# Last activation
		block = _bn_relu(block)
		# Classifier block
		gap = GlobalAveragePooling2D(name='gap')(block)

		dense = Dense(units=num_classes, kernel_initializer="he_normal",
					  activation="softmax", name='output')(gap)
		if len(aux_depth) > 0:
			model = Model(inputs=input, outputs=[dense] + aux_list)
		else:
			model = Model(inputs=input, outputs=dense)

		return model

	@staticmethod
	def build_resnet_18(input_shape, num_classes, aux_depth=[], num_coarse_classes=[], min_size=8, aux_layouts=None):
		"""
		Build ResNet-18 with specified parameters and modifications.
		Args:
			input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols).
			num_classes: The number of outputs at final softmax layer.
			aux_depth: Position of aux network in the ResNet. 0 before first residual block, n>0 after n-th residual
				block. List, if multiple aux nets.
			num_coarse_classes: Number of categories in the auxiliary classifier. List, if multiple aux classifiers.
			min_size: Defines minimal width and height the conv layers must have, currently only in version without
				modifiers.
		"""

		return ResnetBuilder.build(input_shape, num_classes, basic_block, [2, 2, 2, 2], aux_depth=aux_depth,
								   num_coarse_classes=num_coarse_classes, min_size=min_size, aux_layouts=aux_layouts)

	@staticmethod
	def build_resnet_50(input_shape, num_classes, aux_depth=[], num_coarse_classes=[], min_size=8, aux_layouts=None):
		"""
		Build ResNet-50 with specified parameters and modifications.
		Args:
			input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols).
			num_classes: The number of outputs at final softmax layer.
			aux_depth: Position of aux network in the ResNet. 0 before first residual block, n>0 after n-th residual
				block. List, if multiple aux nets.
			num_coarse_classes: Number of categories in the auxiliary classifier. List, if multiple aux classifiers.
			min_size: Defines minimal width and height the conv layers must have, currently only in version without
				modifiers.
		"""
		return ResnetBuilder.build(input_shape, num_classes, bottleneck, [3, 4, 6, 3], aux_depth=aux_depth,
								   num_coarse_classes=num_coarse_classes, min_size=min_size,
								   aux_layouts=aux_layouts)
