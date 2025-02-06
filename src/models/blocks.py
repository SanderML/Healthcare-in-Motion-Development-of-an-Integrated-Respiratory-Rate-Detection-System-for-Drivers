from keras.api.layers import Activation, BatchNormalization, Conv1D


def Conv_1D_Block(x, filters, kernel, strides):
    x = Conv1D(filters, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x
