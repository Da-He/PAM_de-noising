'''
PAM de-noising tool, version_20221122, by Da He
---------------
NOTE: 
*.png file datasets: 0~255, uint8.
'''

from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import cv2
import matplotlib.image as mpimg # mpimg 

import matplotlib

# Force matplotlib to not use any Xwindows backend.

matplotlib.use('Agg')


import matplotlib.pyplot as plt
import sys

from pam_denoise_main import SRGAN

import numpy as np
import os

import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True
sess = tf.Session(config=gpu_config)
KTF.set_session(sess)


def cov(y_true, y_pred, mu_y, mu_x):
    x = y_pred - mu_x
    y = y_true - mu_y
    cov_result = tf.cast(K.sum(tf.multiply(x,y)),dtype=tf.float32) / tf.cast((tf.size(x)-1), dtype=tf.float32)
    return cov_result

def ssim(y_true, y_pred):
    y_true = y_true[0]
    y_pred = y_pred[0]
    y_pred = 255.0 * (0.5 * y_pred + 0.5)
    y_true = 255.0 * (0.5 * y_true + 0.5)
    R = 255.0
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    var_x = K.var(y_pred)
    var_y = K.var(y_true)
    cov_xy = cov(y_true, y_pred, mu_y, mu_x)
    c1 = tf.constant((0.01 * R)**2)
    c2 = tf.constant((0.03 * R)**2)
    ssim = (2*mu_x*mu_y+c1)*(2*cov_xy+c2)/((K.pow(mu_x,2)+K.pow(mu_y,2)+c1)*(var_x+var_y+c2))
    return ssim

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    y_true = y_true[0]
    y_pred = y_pred[0]
    #if NORM_MODE == 6:
    # max_pixel = 1.0
    y_pred = 255.0 * (0.5 * y_pred + 0.5)
    y_true = 255.0 * (0.5 * y_true + 0.5)
    max_pixel = 255.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def explore_input_dir(input_dir, input_name_list):
    same_h, same_w = None, None
    for idx, input_name in enumerate(input_name_list):
        input_path = os.path.join(input_dir, input_name)
        noisy_img = cv2.imread(input_path, -1)
        if(idx == 0):
            same_h, same_w = noisy_img.shape[:2]
        else:
            temp_h, temp_w = noisy_img.shape[:2]
            if(temp_h != same_h or temp_w != same_w):
                return False
    return [same_h, same_w]

def cal_minimal_padding(ori_h, ori_w):
    temp_h, temp_w = ori_h, ori_w
    valid_tag = True
    while(temp_h % 16 != 0):
        temp_h += 1
        if(temp_h > 2 * ori_h):
            valid_tag = False
            break
    while(temp_w % 16 != 0):
        temp_w += 1
        if(temp_w > 2 * ori_w):
            valid_tag = False
            break
    if(not valid_tag):
        return None, None, None, None, valid_tag
    padding_up = (temp_h - ori_h)//2 if(ori_h % 2 == 0) else (temp_h - ori_h - 1)//2
    padding_left = (temp_w - ori_w)//2 if(ori_w % 2 == 0) else (temp_w - ori_w - 1)//2
    return temp_h, temp_w, padding_up, padding_left, valid_tag


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='PAM de-noising tool (Ver.20221122)')
    parser.add_argument('--input_dir', default='', type=str, metavar='PATH',
                        help='the directory of the noisy images (default: none)')
    args = parser.parse_args()

    weight_path = './generator_59700_G-PSNR-44.91450881958008_weights.hdf5'
    
    assert (args.input_dir != ''), 'Please input the directory of the input images.'
    input_dir = args.input_dir[:-1] if(args.input_dir[-1]=='/') else args.input_dir
    input_name_list = next(os.walk(input_dir))[2]
    assert (len(input_name_list) != 0), 'There is no noisy image in the given directory.'
    output_dir = os.path.join(input_dir, 'denoised_out')
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    same_size = explore_input_dir(input_dir, input_name_list)

    if(same_size == False):
        print('****** Low speed inference mode (images with different shapes) ******')
        for input_name in input_name_list:
            input_path = os.path.join(input_dir, input_name)
            noisy_img = cv2.imread(input_path, -1).astype(np.float32)
            noisy_img = noisy_img[:,:,0] if(len(noisy_img.shape)==3) else noisy_img
            noisy_img = (noisy_img - np.min(noisy_img)) / (np.max(noisy_img) - np.min(noisy_img))
            ori_h, ori_w = noisy_img.shape[:]
            padded_h, padded_w, padding_up, padding_left, valid_tag = cal_minimal_padding(ori_h, ori_w)
            assert (valid_tag == True), 'The image shape of {} is hard to handle.'.format(input_name)
            padded_input = np.zeros((padded_h, padded_w))
            padded_input[padding_up:padding_up+ori_h, padding_left:padding_left+ori_w] = noisy_img
            padded_input = padded_input.reshape((1,padded_h,padded_w,1))

            test = SRGAN(lr_height = padded_h, lr_width=padded_w, no_dataloader=True)
            test.generator.load_weights(weight_path)

            padded_output = test.generator.predict(padded_input)
            padded_output = np.squeeze(padded_output)
            output = padded_output[padding_up:padding_up+ori_h, padding_left:padding_left+ori_w]
            output_png = (output - np.min(output)) / (np.max(output) - np.min(output)) * 255.0
            output_path = os.path.join(output_dir, input_name.split('.png')[0]+'_denoised.png')
            cv2.imwrite(output_path, output_png.astype(np.uint8))
            print('Saved {}'.format(output_path))

    else:
        print('****** High speed inference mode (images with the same shape) ******')
        ori_h, ori_w = same_size[:]
        padded_h, padded_w, padding_up, padding_left, valid_tag = cal_minimal_padding(ori_h, ori_w)
        assert (valid_tag == True), 'The image shape of {} is hard to handle.'.format(same_size)
        test = SRGAN(lr_height = padded_h, lr_width=padded_w, no_dataloader=True)
        test.generator.load_weights(weight_path)

        for input_name in input_name_list:
            input_path = os.path.join(input_dir, input_name)
            noisy_img = cv2.imread(input_path, -1).astype(np.float32)
            noisy_img = noisy_img[:,:,0] if(len(noisy_img.shape)==3) else noisy_img
            noisy_img = (noisy_img - np.min(noisy_img)) / (np.max(noisy_img) - np.min(noisy_img))
            padded_input = np.zeros((padded_h, padded_w))
            padded_input[padding_up:padding_up+ori_h, padding_left:padding_left+ori_w] = noisy_img
            padded_input = padded_input.reshape((1,padded_h,padded_w,1))

            padded_output = test.generator.predict(padded_input)
            padded_output = np.squeeze(padded_output)
            output = padded_output[padding_up:padding_up+ori_h, padding_left:padding_left+ori_w]
            output_png = (output - np.min(output)) / (np.max(output) - np.min(output)) * 255.0
            output_path = os.path.join(output_dir, input_name.split('.png')[0]+'_denoised.png')
            cv2.imwrite(output_path, output_png.astype(np.uint8))
            print('Saved {}'.format(output_path))





