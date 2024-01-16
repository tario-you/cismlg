from __future__ import print_function

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import warnings

warnings.filterwarnings("ignore")

#from keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from scipy.io import savemat
from skimage.io import imsave

from data import load_data
from net import unet

from re import match

#weights_path = "./weights_128.h5"
#train_images_path = "./data/train/"
#test_images_path = "./data/valid/"
#predictions_path = "./predictions/"
weights_folder_path = "../../out/weights"
train_images_path = "../../archive/train"
test_images_path = "../../archive/test"
predictions_folder_path = "../../out/predictions"
dsc_folder_path = "../../out/DSCs"

#gpu = "0"


#def predict(mean=20.0, std=43.0):
def predict(weights_path: str, mean=20.0, std=43.0):
    # load and normalize data
    if mean == 0.0 and std == 1.0:
        imgs_train, _, _ = load_data(train_images_path)
        mean = np.mean(imgs_train)
        std = np.std(imgs_train)
    imgs_test, imgs_mask_test, names_test = load_data(test_images_path)
    original_imgs_test = imgs_test.astype(np.uint8)
    imgs_test -= mean
    imgs_test /= std
    # load model with weights
    #model = unet()
    model = unet(to_conv6, to_conv7, to_conv8, to_conv9)
    model.load_weights(weights_path)

    # make predictions
    imgs_mask_pred = model.predict(imgs_test, verbose=verbose)
    # save to mat file for further processing

    matdict = {
        "pred": imgs_mask_pred,
        "image": original_imgs_test,
        "mask": imgs_mask_test,
        "name": names_test,
    }
    #savemat(os.path.join(predictions_path, "predictions.mat"), matdict)
    savemat(os.path.join(predictions_folder_path, f"predictions_{fname}.mat"), matdict)

    predictions_path = f'{predictions_folder_path}/{fname}'
    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)
    # save images with segmentation and ground truth mask overlay
    for i in range(len(imgs_test)):
        pred = imgs_mask_pred[i]
        image = original_imgs_test[i]
        mask = imgs_mask_test[i]

        # segmentation mask is for the middle slice
        image_rgb = gray2rgb(image[:, :, 1])

        # prediction contour image
        pred = (np.round(pred[:, :, 0]) * 255.0).astype(np.uint8)
        #pred, contours, _ = cv2.findContours(
        contours, _ = cv2.findContours(
            pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        pred = np.zeros(pred.shape)
        cv2.drawContours(pred, contours, -1, (255, 0, 0), 1)

        # ground truth contour image
        mask = (np.round(mask[:, :, 0]) * 255.0).astype(np.uint8)
        #mask, contours, _ = cv2.findContours(
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        mask = np.zeros(mask.shape)
        cv2.drawContours(mask, contours, -1, (255, 0, 0), 1)

        # combine image with contours
        pred_rgb = np.array(image_rgb)
        annotation = pred_rgb[:, :, 1]
        annotation[np.maximum(pred, mask) == 255] = 0
        pred_rgb[:, :, 0] = pred_rgb[:, :, 1] = pred_rgb[:, :, 2] = annotation
        pred_rgb[:, :, 2] = np.maximum(pred_rgb[:, :, 2], mask)
        pred_rgb[:, :, 0] = np.maximum(pred_rgb[:, :, 0], pred)

        #imsave(os.path.join(predictions_path, names_test[i] + ".png"), pred_rgb)
        imsave(os.path.join(predictions_path, names_test[i].decode("utf-8") + ".png"), pred_rgb)

    return imgs_mask_test, imgs_mask_pred, names_test


def evaluate(imgs_mask_test, imgs_mask_pred, names_test):
    test_pred = zip(imgs_mask_test, imgs_mask_pred)
    name_test_pred = zip(names_test, test_pred)
    #name_test_pred.sort(key=lambda x: x[0])
    name_test_pred = sorted(name_test_pred, key=lambda x: x[0])

    patient_ids = []
    dc_values = []

    i = 0  # start slice index
    for p in range(len(name_test_pred)):
        # get case id (names are in format <case_id>_<slice_number>)
        #p_id = "_".join(name_test_pred[p][0].split("_")[:-1])
        p_id = "_".join(name_test_pred[p][0].decode("utf-8").split("_")[:-1])

        # if this is the last slice for the processed case
        #if p + 1 >= len(name_test_pred) or p_id not in name_test_pred[p + 1][0]:
        if p + 1 >= len(name_test_pred) or p_id not in name_test_pred[p + 1][0].decode("utf-8"):
            # ground truth segmentation:
            p_slices_mask = np.array(
                [im_m[0] for im_id, im_m in name_test_pred[i : p + 1]]
            )
            # predicted segmentation:
            p_slices_pred = np.array(
                [im_m[1] for im_id, im_m in name_test_pred[i : p + 1]]
            )

            patient_ids.append(p_id)
            dc_values.append(dice_coefficient(p_slices_pred, p_slices_mask))
            print(p_id + ":\t" + str(dc_values[-1]))

            i = p + 1

    return dc_values, patient_ids


def dice_coefficient(prediction, ground_truth):
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return (
        np.sum(prediction[ground_truth == 1])
        * 2.0
        / (np.sum(prediction) + np.sum(ground_truth))
    )


def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


def plot_dc(labels, values):
    y_pos = np.arange(len(labels))

    fig = plt.figure(figsize=(12, 8))
    plt.barh(y_pos, values, align="center", alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xticks(np.arange(0.5, 1.0, 0.05))
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.axes().xaxis.grid(color="black", linestyle="-", linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0.5, 1.0])
    plt.tight_layout()
    axes.axvline(np.mean(values), color="green", linewidth=2)

    #plt.savefig("DSC.png", bbox_inches="tight")
    plt.savefig(os.path.join(dsc_folder_path, f"DSC_{fname}.png"), bbox_inches="tight")
    plt.close(fig)


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
        test_images_path += "_small"
        predictions_folder_path += "_small"
    verbose = int(sys.argv[3])
    
    if not os.path.exists(predictions_folder_path):
        os.mkdir(predictions_folder_path)
    if not os.path.exists(dsc_folder_path):
        os.mkdir(dsc_folder_path)

    try:
        fname = None
        imgs_mask_test, imgs_mask_pred, names_test = None, None, None
        values, labels = None, None
        with tf.device(device):
            for to_conv9 in ['conv4', 'conv3', 'conv2', 'conv1']:
                for to_conv8 in ['conv4', 'conv3', 'conv2', 'conv1']:
                    for to_conv7 in ['conv4', 'conv3', 'conv2', 'conv1']:
                        for to_conv6 in ['conv4', 'conv3', 'conv2', 'conv1']:
                            fname = f'{to_conv6}_{to_conv7}_{to_conv8}_{to_conv9}'
                            #imgs_mask_test, imgs_mask_pred, names_test = predict()
                            imgs_mask_test, imgs_mask_pred, names_test = predict(os.path.join(weights_folder_path, f'weights_{fname}.h5'))
                            values, labels = evaluate(imgs_mask_test, imgs_mask_pred, names_test)
                            #print("\nAverage DSC: " + str(np.mean(values)))
                            print(f"Average DSC of {fname}: {str(np.nanmean(values))}")
                            plot_dc(labels, values)
    except KeyboardInterrupt:
        print('\naasu: interrupted')
