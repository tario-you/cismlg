from __future__ import print_function

import os
import sys

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#from keras import backend as K
#from keras.callbacks import TensorBoard
#from keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import TensorBoard
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras.optimizers import Adam

from tensorflow.compat.v1.keras.utils import plot_model

from data import load_data
from data import oversample
from net import dice_coef
from net import dice_coef_loss
from net import unet

#train_images_path = "./data/train/""..\..\archive\test"
#valid_images_path = "./data/valid/"
#init_weights_path = "./weights_128.h5"
train_images_path = "../../archive/train"
valid_images_path = "../../archive/validate"
#init_weights_path = "weights_64.h5"
weights_path = "../../out/weights"
log_folder_path = "../../out/logs"
#architecture_path = "../../out/architectures"
history_path = "../../out/history.txt"

#gpu = "0"

#epochs = 128
#batch_size = 32
base_lr = 1e-5

def train(augment: bool, verbose: bool):
    #print('aasu: loading and pre-processing data...', end='')
    imgs_train, imgs_mask_train, _ = load_data(train_images_path)
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std
    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path)
    imgs_valid -= mean
    imgs_valid /= std
    #imgs_train, imgs_mask_train = oversample(imgs_train, imgs_mask_train)
    imgs_train, imgs_mask_train = oversample(imgs_train, imgs_mask_train, augment=augment)
    #print('completed')

    #model = unet()
    #d = {'to_conv6': 'conv4', 'to_conv7': 'conv3', 'to_conv8': 'conv2', 'to_conv9': 'conv1'}
    #model = unet('conv4', 'conv3', 'conv2', 'conv1')

    #models.append(unet(to_conv6, to_conv7, to_conv8, to_conv9))
    #fnames.append(f'{to_conv6}_{to_conv7}_{to_conv8}_{to_conv9}')
    #print(f'aasu: initializing model {fname} ...', end='')
    model = unet(to_conv6, to_conv7, to_conv8, to_conv9)
    model.compile(optimizer=Adam(lr=base_lr), loss=dice_coef_loss, metrics=[dice_coef])
    #plot_model(
    #    model=model,
    #    to_file=os.path.join(architecture_path, f'model_{fname}.png'),
    #    show_shapes=True,
    #    show_layer_names=True,
    #    rankdir='LR',
    #    expand_nested=False,
    #    dpi=96,
    #)
    
    #model = unet(architecture['to_conv6'], architecture['to_conv7'], architecture['to_conv8'], architecture['to_conv9'])
    #model.summary()
    #fname = '_'.join(val for val in architecture.values())

    #if os.path.exists(init_weights_path):
    #    #model.load_weights(init_weights_path)
    #    model.load_weights(init_weights_path, by_name=True, skip_mismatch=True)

    #for model in unets.keys():
    #    model.compile(optimizer=Adam(lr=base_lr), loss=dice_coef_loss, metrics=[dice_coef])
    #optimizer = Adam(lr=base_lr)
    #model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])

    #training_log = TensorBoard(log_dir=log_path)

    #print('aasu: start training? (y/[n])', end='')
    #if input() != 'y':
    #    return
    #model.fit(
    #    imgs_train,
    #    imgs_mask_train,
    #    validation_data=(imgs_valid, imgs_mask_valid),
    #    batch_size=batch_size,
    #    epochs=epochs,
    #    shuffle=True,
    #    callbacks=[training_log],
    #)

    #print(f'...completed\naasu: training model {fname}: ')
    print(fname)
    train_history = model.fit(
        imgs_train,
        imgs_mask_train,
        validation_data=(imgs_valid, imgs_mask_valid),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        callbacks=[
            TensorBoard(log_dir=log_path, histogram_freq=1, write_graph=True, write_grads=True, write_images=True),
            EarlyStopping(monitor='val_loss', verbose=1, patience=2)
        ],
        verbose=verbose,
        use_multiprocessing=False
    )
    #print('aasu: training completed\naasu: outputting...', end='')
    #model.save_weights(os.path.join(weights_path, "weights_{}.h5".format(epochs)))
    model.save_weights(os.path.join(weights_path, f'weights_{fname}.h5'))
    with open(history_path, 'a') as fp:
        fp.write(f'{fname}: {str(train_history.history)}\n')
    print('completed')


if __name__ == "__main__":
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.allow_soft_placement = True
    #sess = tf.Session(config=config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    #if len(sys.argv) > 1:
    #    gpu = sys.argv[1]
    #device = "/gpu:" + gpu
    device = sys.argv[1]
    if sys.argv[2] == 'small':
        train_images_path += "_small"
        valid_images_path += "_small"
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    augment = True if sys.argv[5] == 'True' else False
    verbose = int(sys.argv[6])

    if not os.path.exists("../../out"):
        os.mkdir("../../out")
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    #if not os.path.exists(architecture_path):
    #    os.mkdir(architecture_path)
    #history_path = os.path.join(architecture_path, 'history.txt')
    with open(history_path, 'w') as fp:
        fp.write(f'<={epochs} epochs\n')
    
    try:
        with tf.device(device):
            fname, log_path = None, None
            #for to_conv9 in ['conv4', 'conv3', 'conv2', 'conv1']:
            for to_conv9 in ['conv2', 'conv1']:
                for to_conv8 in ['conv4', 'conv3', 'conv2', 'conv1']:
                    for to_conv7 in ['conv4', 'conv3', 'conv2', 'conv1']:
                        for to_conv6 in ['conv4', 'conv3', 'conv2', 'conv1']:
                                fname = f'{to_conv6}_{to_conv7}_{to_conv8}_{to_conv9}'
                                log_path = f'{log_folder_path}/{fname}'
                                if not os.path.exists(log_path):
                                    os.mkdir(log_path)
                                train(augment, verbose)
    except KeyboardInterrupt:
        print('\naasu: interrupted')
