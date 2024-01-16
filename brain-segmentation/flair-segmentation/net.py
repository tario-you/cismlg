from __future__ import print_function

from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model

from keras.layers import UpSampling2D
from typing import Tuple, Any

from data import channels
from data import image_cols
from data import image_rows
from data import modalities

batch_norm = False
smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def conv_layer(input, filters: int, kernel_size: Tuple[int, int], padding: str, axis: int, activation: str):
    convn = Conv2D(filters, kernel_size, padding=padding)(input)
    if batch_norm:
        convn = BatchNormalization(axis=axis)(convn)
    return Activation(activation)(convn)

def larm(input, filters: int, kernel_size: Tuple[int, int], padding: str, axis: int, activation: str, pool_size: Tuple[int, int]):
    convn = conv_layer(input, filters, kernel_size, padding, axis, activation)
    convn = conv_layer(convn, filters, kernel_size, padding, axis, activation)
    pooln = MaxPooling2D(pool_size=pool_size)(convn)
    return convn, pooln

def rarm(input, concat, filters: int, kernel_size_t: Tuple[int, int], kernel_size: Tuple[int, int], strides: Tuple[int, int], padding: str, axis: int, activation: str):
    upn = Conv2DTranspose(filters, kernel_size_t, strides=strides, padding=padding)(input)
    #upn = concatenate([upn, (upn if concat is None else concat)], axis=axis)
    upn = concatenate([upn, concat], axis=axis)
    convn = conv_layer(upn, filters, kernel_size, padding, axis, activation)
    convn = conv_layer(convn, filters, kernel_size, padding, axis, activation)
    return convn

def change_dim(input, src_dim: int, dest_dim: int):
    if src_dim < dest_dim:
        size = (dest_dim // src_dim, dest_dim // src_dim)
        return UpSampling2D(data_format='channels_last', size=size, interpolation='nearest')(input)
    elif src_dim > dest_dim:
        size = (src_dim // dest_dim, src_dim // dest_dim)
        return MaxPooling2D(data_format='channels_last', pool_size=size)(input)
    else: #src_dim == dest_dim
        return input

def unet(to_conv6: str, to_conv7: str, to_conv8: str, to_conv9: str):
    inputs = Input((image_rows, image_cols, channels * modalities))

    conv1, pool1 = larm(inputs, 32, (3, 3), 'same', 3, 'relu', (2, 2))
    conv2, pool2 = larm(pool1, 64, (3, 3), 'same', 3, 'relu', (2, 2))
    conv3, pool3 = larm(pool2, 128, (3, 3), 'same', 3, 'relu', (2, 2))
    conv4, pool4 = larm(pool3, 256, (3, 3), 'same', 3, 'relu', (2, 2))

    conv5 = conv_layer(pool4, 512, (3, 3), 'same', 3, 'relu')
    conv5 = conv_layer(conv5, 512, (3, 3), 'same', 3, 'relu')

    #to_conv6 = change_dim(conv4, 32, 32, self=False)
    #to_conv7 = change_dim(conv3, 64, 64, self=False)
    #to_conv8 = change_dim(conv2, 128, 128, self=False)
    #to_conv9 = change_dim(conv1, 256, 256, self=False)
    #to_conv6 = change_dim(conv4, 32, 32, self=True)
    #to_conv7 = change_dim(conv4, 32, 64, self=False)
    #to_conv8 = change_dim(conv3, 64, 128, self=False)
    #to_conv9 = change_dim(conv2, 128, 256, self=False)
    concat_dict = {'conv4': (conv4, 32), 'conv3': (conv3, 64), 'conv2': (conv2, 128), 'conv1': (conv1, 256)}
    to_conv6 = change_dim(concat_dict[to_conv6][0], concat_dict[to_conv6][1], 32)
    to_conv7 = change_dim(concat_dict[to_conv7][0], concat_dict[to_conv7][1], 64)
    to_conv8 = change_dim(concat_dict[to_conv8][0], concat_dict[to_conv8][1], 128)
    to_conv9 = change_dim(concat_dict[to_conv9][0], concat_dict[to_conv9][1], 256)

    conv6 = rarm(conv5, to_conv6, 256, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')
    conv7 = rarm(conv6, to_conv7, 128, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')
    conv8 = rarm(conv7, to_conv8, 64, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')
    conv9 = rarm(conv8, to_conv9, 32, (2, 2), (3, 3), (2, 2), 'same', 3, 'relu')

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

"""
def unet_backup_2():
    inputs = Input((image_rows, image_cols, channels * modalities))

    conv1 = conv_layer(inputs, 32, (3, 3), 'same', 3, 'relu')
    conv1 = conv_layer(conv1, 32, (3, 3), 'same', 3, 'relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_layer(pool1, 64, (3, 3), 'same', 3, 'relu')
    conv2 = conv_layer(conv2, 64, (3, 3), 'same', 3, 'relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_layer(pool2, 128, (3, 3), 'same', 3, 'relu')
    conv3 = conv_layer(conv3, 128, (3, 3), 'same', 3, 'relu')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_layer(pool3, 256, (3, 3), 'same', 3, 'relu')
    conv4 = conv_layer(conv4, 256, (3, 3), 'same', 3, 'relu')
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_layer(pool4, 512, (3, 3), 'same', 3, 'relu')
    conv5 = conv_layer(conv5, 512, (3, 3), 'same', 3, 'relu')
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)

    conv6 = conv_layer(up6, 256, (3, 3), 'same', 3, 'relu')
    conv6 = conv_layer(conv6, 256, (3, 3), 'same', 3, 'relu')
    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)

    conv7 = conv_layer(up7, 128, (3, 3), 'same', 3, 'relu')
    conv7 = conv_layer(conv7, 128, (3, 3), 'same', 3, 'relu')
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)

    conv8 = conv_layer(up8, 64, (3, 3), 'same', 3, 'relu')
    conv8 = conv_layer(conv8, 64, (3, 3), 'same', 3, 'relu')
    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)

    conv9 = conv_layer(up9, 32, (3, 3), 'same', 3, 'relu')
    conv9 = conv_layer(conv9, 32, (3, 3), 'same', 3, 'relu')

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def unet_backup():
    inputs = Input((image_rows, image_cols, channels * modalities))

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    if batch_norm:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    if batch_norm:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    if batch_norm:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    if batch_norm:
        conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    if batch_norm:
        conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    if batch_norm:
        conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    if batch_norm:
        conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    if batch_norm:
        conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    if batch_norm:
        conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    if batch_norm:
        conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    if batch_norm:
        conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)

    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    if batch_norm:
        conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    if batch_norm:
        conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)

    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    if batch_norm:
        conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    if batch_norm:
        conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)

    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    if batch_norm:
        conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    if batch_norm:
        conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
"""

