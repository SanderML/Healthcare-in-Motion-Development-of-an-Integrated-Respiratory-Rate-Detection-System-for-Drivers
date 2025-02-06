# Original source: https://github.com/Sakib1263/TF-1D-2D-ResNetV1-2-SEResNet-ResNeXt-SEResNeXt
# ResNet models for Keras.
# Reference for ResNets - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf))

import enum

import keras
import tensorflow as tf

from .blocks import Conv_1D_Block


def _stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    # First Convolutional layer, where pooled feature maps will be reduced by 75%
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    pool_size = 1 if conv.shape[1] <= 2 else 2
    pool = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=2, padding="valid")(conv)
    return pool


def conv_block(inputs, num_filters):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    conv = Conv_1D_Block(inputs, num_filters, 3, 2)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)

    return conv


def residual_block(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = inputs
    #
    conv = Conv_1D_Block(inputs, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    # conv = tf.keras.layers.SpatialDropout1D(0.2)(conv)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation("relu")(conv)

    return out


def residual_group(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block(out, num_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out


def residual_block_bottleneck(inputs, num_filters):
    # Construct a Residual Block of Convolutions
    # x        : input into the block
    # n_filters: number of filters
    shortcut = Conv_1D_Block(inputs, num_filters * 4, 1, 1)
    #
    conv = Conv_1D_Block(inputs, num_filters, 1, 1)
    conv = Conv_1D_Block(conv, num_filters, 3, 1)
    conv = Conv_1D_Block(conv, num_filters * 4, 1, 1)
    conv = tf.keras.layers.Add()([conv, shortcut])
    out = tf.keras.layers.Activation("relu")(conv)

    return out


def residual_group_bottleneck(inputs, num_filters, n_blocks, conv=True):
    # x        : input to the group
    # n_filters: number of filters
    # n_blocks : number of blocks in the group
    # conv     : flag to include the convolution block connector
    out = inputs
    for i in range(n_blocks):
        out = residual_block_bottleneck(out, num_filters)

    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    if conv:
        out = conv_block(out, num_filters * 2)

    return out


class ResNetVariant(enum.Enum):
    r18 = [2, 2, 1, 4, 1, 8, 1]
    r34 = [3, 2, 3, 4, 5, 8, 2]
    r50 = [3, 2, 3, 4, 5, 8, 2]
    r101 = [3, 2, 3, 4, 22, 8, 2]
    r152 = [3, 2, 7, 4, 35, 8, 2]

    @property
    def stem(self):
        return _stem

    def learner(self, inputs, num_filters):
        v = self.value
        x = residual_group(inputs, num_filters, v[0])  # First Residual Block Group of 64 filters
        x = residual_group(x, num_filters * v[1], v[2])  # Second Residual Block Group of 128 filters
        x = residual_group(x, num_filters * v[3], v[4])  # Third Residual Block Group of 256 filters
        out = residual_group(x, num_filters * v[5], v[6], False)  # Fourth Residual Block Group of 512 filters
        return out


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation="softmax")(inputs)

    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs       : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation="linear")(inputs)

    return out


@keras.api.utils.register_keras_serializable(package="Custom", name="ResNet")
class ResNet:
    def __init__(
        self, input_shape, num_filters, problem_type="Classification", output_nums=1, pooling="max", dropout_rate=False
    ):
        self.length, self.num_channel = input_shape
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "length": self.length,
            "num_channel": self.num_channel,
            "num_filters": self.num_filters,
            "problem_type": self.problem_type,
            "output_nums": self.output_nums,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
        }
        return config

    def MLP(self, x):
        if self.pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten(name="flatten")(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name="Dropout")(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation="linear")(x)
        if self.problem_type == "Classification":
            outputs = tf.keras.layers.Dense(self.output_nums, activation="sigmoid")(x)

        return outputs

    def _ResNetFactory(self, variant: ResNetVariant):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_ = variant.stem(inputs, self.num_filters)  # The Stem Convolution Group
        x = variant.learner(stem_, self.num_filters)  # The learner
        outputs = self.MLP(x)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def ResNet18(self):
        return self._ResNetFactory(ResNetVariant.r18)

    def ResNet34(self):
        return self._ResNetFactory(ResNetVariant.r34)

    def ResNet50(self):
        return self._ResNetFactory(ResNetVariant.r50)

    def ResNet101(self):
        return self._ResNetFactory(ResNetVariant.r101)

    def ResNet152(self):
        return self._ResNetFactory(ResNetVariant.r152)
