'''
Adapted for use of auxiliary networks.
Original implementation from Keras:
DenseNet and DenseNet-FCN models for Keras.
DenseNet is a network architecture where each layer is directly connected
to every other layer in a feed-forward fashion (within each dense block).
For each layer, the feature maps of all preceding layers are treated as
separate inputs whereas its own feature maps are passed on as inputs to
all subsequent layers. This connectivity pattern yields state-of-the-art
accuracies on CIFAR10/100 (with or without data augmentation) and SVHN.
On the large scale ILSVRC 2012 (ImageNet) dataset, DenseNet achieves a
similar accuracy as ResNet, but using less than half the amount of
parameters and roughly half the number of FLOPs.
DenseNets support any input image size of 32x32 or greater, and are thus
suited for CIFAR-10 or CIFAR-100 datasets. There are two types of DenseNets,
one suited for smaller images (DenseNet) and one suited for ImageNet,
called DenseNetImageNet. They are differentiated by the strided convolution
and pooling operations prior to the initial dense block.
The following table describes the size and accuracy of DenseNetImageNet models
on the ImageNet dataset (single crop), for which weights are provided:
------------------------------------------------------------------------------------
    Model type      | ImageNet Acc (Top 1)  |  ImageNet Acc (Top 5) |  Params (M)  |
------------------------------------------------------------------------------------
|   DenseNet-121    |    25.02 %            |        7.71 %         |     8.0      |
|   DenseNet-169    |    23.80 %            |        6.85 %         |     14.3     |
|   DenseNet-201    |    22.58 %            |        6.34 %         |     20.2     |
|   DenseNet-161    |    22.20 %            |         -   %         |     28.9     |
------------------------------------------------------------------------------------
DenseNets can be extended to image segmentation tasks as described in the
paper "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for
Semantic Segmentation". Here, the dense blocks are arranged and concatenated
with long skip connections for state of the art performance on the CamVid dataset.
# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic
   Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
This implementation is based on the following reference code:
 - https://github.com/gpleiss/efficient_densenet_pytorch
 - https://github.com/liuzhuang13/DenseNet
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils.layer_utils import convert_all_kernels_in_model
from tensorflow.python.keras.utils.data_utils import get_file

import tensorflow as K

from aux_nets_tf import *
from helper import obtain_input_shape

global init
global aux_initia

"""
Try DenseNet-BC with L=250,k=24 or L=190,k=40, theta 0.5
"""
def DenseNet(input_shape=None,
             depth=40,  # L parameter / Depth parameter
             nb_dense_block=3,
             growth_rate=12,  # k parameter
             nb_filter=-1,
             nb_layers_per_block=-1,
             bottleneck=False,  # True for DenseNet-B
             reduction=0.0,  # 1-theta (1-compression), >0.0 for DenseNet-C
             dropout_rate=0.2,
             weight_decay=1e-4,
             subsample_initial_block=False,  # Keep false for CIFAR
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling='avg',
             classes=10,
             activation='softmax',
             transition_pooling='avg', aux_depth=None,
                                               num_coarse_classes=None, aux_layouts=None, initialization='he_normal',
                                                aux_init='he_normal'):
    '''Instantiate the DenseNet architecture.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` dim ordering)
            or `(3, 224, 224)` (with `channels_first` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 8.
            E.g. `(224, 224, 3)` would be one valid value.
        depth: number or layers in the DenseNet
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. -1 indicates initial
            number of filters will default to 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be nb_dense_block
        bottleneck: flag to add bottleneck blocks in between dense blocks
        reduction: reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression.
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling2D before the initial dense block.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization) or
            'imagenet' (pre-training on ImageNet)..
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        activation: Type of activation at the top layer. Can be one of
            'softmax' or 'sigmoid'. Note that if sigmoid is used,
             classes must be 1.
        transition_pooling: `avg` for avg pooling (default), `max` for max pooling,
            None for no pooling during scale transition blocks. Please note that this
            default differs from the DenseNetFCN paper in accordance with the DenseNet
            paper.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    input_shape = obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format='channels_last',
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    global init
    init = initialization
    global aux_initia
    aux_initia = aux_init

    model = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck,
                           reduction, dropout_rate, weight_decay,
                           subsample_initial_block, pooling, activation,
                           transition_pooling, aux_depth=aux_depth,
                                               num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts)

    return model


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and relu),
    and optionally a bottleneck layer.
    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming
     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        output tensor of block
    '''
    with K.name_scope('ConvBlock'):
        concat_axis = 1 if 'channels_last' == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            x = Conv2D(inter_channel, (1, 1), kernel_initializer=init,
                       padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay),
                       name=name_or_none(block_prefix, '_bottleneck_conv2D'))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name=name_or_none(block_prefix, '_bottleneck_bn'))(x)
            x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer=init, padding='same', kernel_regularizer=l2(weight_decay),
                   use_bias=False, name=name_or_none(block_prefix, '_conv2D'))(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
                  return_concat_list=False, block_prefix=None, block_id=None, curr_id=-1, aux_depth=None,
                                               num_coarse_classes=None, aux_layouts=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if 'channels_last' == 'channels_first' else -1

        x_list = [x]

        curr_id_temp = curr_id
        aux_list_temp = []

        for i in range(nb_layers):
            # Aux net
            aux_list_temp_temp, curr_id_temp = apply_aux(x, major_id=block_id+1, minor_id=i,
                                               curr_id=curr_id_temp, aux_depth=aux_depth,
                                               num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts, initialization=aux_initia)
            aux_list_temp = aux_list_temp + aux_list_temp_temp

            cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list, aux_list_temp, curr_id_temp
        else:
            return x, nb_filter, aux_list_temp, curr_id_temp


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4,
                       block_prefix=None, transition_pooling='max'):
    '''
    Adds a pointwise convolution layer (with batch normalization and relu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.
    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.
    # Returns
        a keras tensor
    '''
    with K.name_scope('Transition'):
        concat_axis = 1 if 'channels_last' == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                               name=name_or_none(block_prefix, '_bn'))(ip)
        x = Activation('relu')(x)
        x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer=init,
                   padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                   name=name_or_none(block_prefix, '_conv2D'))(x)
        if transition_pooling == 'avg':
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        elif transition_pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        return x


def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4,
                          block_prefix=None):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.
    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.
    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):

        if type == 'upsampling':
            x = UpSampling2D(name=name_or_none(block_prefix, '_upsampling'))(ip)
        else:
            x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same',
                                strides=(2, 2), kernel_initializer=init,
                                kernel_regularizer=l2(weight_decay),
                                name=name_or_none(block_prefix, '_conv2DT'))(ip)
        return x


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3,
                       growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
                       bottleneck=False, reduction=0.0, dropout_rate=None,
                       weight_decay=1e-4, subsample_initial_block=False, pooling=None,
                       activation='softmax', transition_pooling='avg', aux_depth=None,
                                               num_coarse_classes=None, aux_layouts=None):
    ''' Build the DenseNet model
    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number
            of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is
            inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling2D before the initial dense block.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        activation: Type of activation at the top layer. Can be one of 'softmax' or
            'sigmoid'. Note that if sigmoid is used, classes must be 1.
        transition_pooling: `avg` for avg pooling (default), `max` for max pooling,
            None for no pooling during scale transition blocks. Please note that this
            default differs from the DenseNetFCN paper in accordance with the DenseNet
            paper.
    # Returns
        a keras tensor
    # Raises
        ValueError: in case of invalid argument for `reduction`
            or `nb_dense_block`
    '''
    with K.name_scope('DenseNet'):
        concat_axis = 1 if 'channels_last' == 'channels_first' else -1

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        if depth == 121:
            nb_layers_per_block = [6,12,24,16]
        elif depth == 169:
            nb_layers_per_block = [6, 12, 32, 32]
        elif depth == 201:
            nb_layers_per_block = [6, 12, 48, 32]
        elif depth == 264:
            nb_layers_per_block = [6, 12, 64, 32]

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != nb_dense_block:
                raise ValueError('If `nb_dense_block` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, ('Depth must be 3 N + 4 '
                                              'if nb_layers_per_block == -1')
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block


        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7)
            initial_strides = (2, 2)
        else:
            initial_kernel = (3, 3)
            initial_strides = (1, 1)

        x = Conv2D(nb_filter, initial_kernel, kernel_initializer=init,
                   padding='same', name='initial_conv2D', strides=initial_strides,
                   use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name='initial_bn')(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        curr_id = 0
        aux_list = []

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            # adds AUXi.0 to AUX<nb_dense_block-1>.<nb_layers-1>
            x, nb_filter, aux_list_temp, curr_id = __dense_block(x, nb_layers[block_idx], nb_filter,
                                         growth_rate, bottleneck=bottleneck,
                                         dropout_rate=dropout_rate,
                                         weight_decay=weight_decay,
                                         block_prefix='dense_%i' % block_idx, block_id=block_idx, curr_id=curr_id, aux_depth=aux_depth,
                                               num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts)

            aux_list = aux_list + aux_list_temp
            # adds AUXi.<nb_layers>
            aux_list_temp, curr_id = apply_aux(x, major_id=block_idx+1, minor_id=nb_layers[block_idx], curr_id=curr_id, aux_depth=aux_depth,
                                               num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts, initialization=aux_initia)
            aux_list = aux_list + aux_list_temp
            aux_list_temp, curr_id = apply_aux(x, major_id=block_idx + 1, minor_id=-1,
                                               curr_id=curr_id, aux_depth=aux_depth,
                                               num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts, initialization=aux_initia)
            aux_list = aux_list + aux_list_temp

            # add transition_block
            x = __transition_block(x, nb_filter, compression=compression,
                                   weight_decay=weight_decay,
                                   block_prefix='tr_%i' % block_idx,
                                   transition_pooling=transition_pooling)
            nb_filter = int(nb_filter * compression)

        # add AUX<nb_dense_block>.0 (AUX3.0 for CIFAR)
        aux_list_temp, curr_id = apply_aux(x, major_id=nb_dense_block, minor_id=0, curr_id=curr_id,
                                           aux_depth=aux_depth,
                                           num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts, initialization=aux_initia)
        aux_list = aux_list + aux_list_temp

        # The last dense_block does not have a transition_block
        x, nb_filter, aux_list_temp, curr_id = __dense_block(x, final_nb_layer, nb_filter, growth_rate,
                                     bottleneck=bottleneck, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay,
                                     block_prefix='dense_%i' % (nb_dense_block - 1), block_id=nb_dense_block, curr_id=curr_id, aux_depth=aux_depth,
                                               num_coarse_classes=num_coarse_classes, aux_layouts=aux_layouts)

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='final_bn')(x)
        x = Activation('relu')(x)
        if include_top:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

            x = Dense(nb_classes, activation=activation, name='output', kernel_regularizer=l2(weight_decay))(x)
            if len(aux_depth) > 0:
                model = Model(inputs=img_input, outputs=[x] + aux_list)
            else:
                model = Model(inputs=img_input, outputs=x)
        else:
            raise NotImplementedError

        return model


