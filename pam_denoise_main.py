"""
This implementation was developed based on the repository https://github.com/eriklindernoren/Keras-GAN/tree/master/srgan.
--------------------------------------------------------------------------------------
Image format note:
*.npy file datasets: 0.0~1.0, float32/float64.
*.png file datasets: 0~255, uint8.
"""

from __future__ import print_function, division
import scipy
from keras.layers import *
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate
from keras.layers import merge, ZeroPadding2D, UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.regularizers import l2
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import argparse
import logging
import time, glob
import PIL.Image as Image
import pandas as pd
#from keras import backend as K
#import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from skimage.measure import compare_psnr, compare_ssim
import skimage
# import models
import datetime
import matplotlib
import json
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

from GCblock import global_context_block

import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True
sess = tf.Session(config=gpu_config)
KTF.set_session(sess)

def smooth_l1_loss(y_true, y_pred):

    diff = K.abs(y_true - y_pred)    
    less_than_one = K.cast(K.less(diff, 1.0), "float32")    
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)    
    return loss

def total_variation_loss(y_true, y_pred):
    
    x=y_pred
    assert K.ndim(x) == 4
    
    img_nrows=x.shape[1]
    img_ncols=x.shape[2]
    
    a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    loss = K.sum(K.pow(a + b, 1.25))
    return loss


def cov(y_true, y_pred, mu_y, mu_x):
    x = y_pred - mu_x
    y = y_true - mu_y
    cov_result = tf.cast(K.sum(tf.multiply(x,y)),dtype=tf.float32) / tf.cast((tf.size(x)-1), dtype=tf.float32)
    return cov_result

def ssim(y_true, y_pred):
    '''
    The PSNR and SSIM values calculated during model.evaluate are not accurate and can only used as a reference.
    Accurate PSNR and SSIM calculation (as shown in the paper) is additionally needed after the online evaluation
    '''
    y_true = y_true[0]
    y_pred = y_pred[0]
    #ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    #print('TYPE : {},{}'.format(type(y_true), type(y_pred)))
    #if NORM_MODE == 6:
    # R = 1.0
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

    #print(tf.shape(y_pred))
    return ssim

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    '''
    The PSNR and SSIM values calculated during model.evaluate are not accurate and can only used as a reference.
    Accurate PSNR and SSIM calculation (as shown in the paper) is additionally needed after the online evaluation
    '''
    y_true = y_true[0]
    y_pred = y_pred[0]
    #if NORM_MODE == 6:
    # max_pixel = 1.0
    y_pred = 255.0 * (0.5 * y_pred + 0.5)
    y_true = 255.0 * (0.5 * y_true + 0.5)
    max_pixel = 255.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
class SRGAN():
    def __init__(self, lr_height=256, lr_width=256, no_dataloader=False):
        # Input shape
        self.lr_channels = 1
        self.hr_channels = 1
        self.lr_height = lr_height                                          # lr -> noisy input
        self.lr_width = lr_width                  
        self.lr_shape = (self.lr_height, self.lr_width, self.lr_channels)
        self.hr_height = self.lr_height                                     # hr -> clean ground truth
        self.hr_width = self.lr_width    
        self.hr_shape = (self.hr_height, self.hr_width, self.hr_channels)
        self.loss_type = 'vgg'

        # Number of residual blocks in the generator
        # self.n_residual_blocks = 16

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        if(self.loss_type == 'vgg' and not no_dataloader):
            self.vgg = self.build_vgg()
            self.vgg.trainable = False
            self.vgg.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])

        # Configure data loader
        self.datasethr_name = 'clean'
        self.datasetlr_name = 'noisy'
        if(not no_dataloader):
            self.data_loader = DataLoader(datasethr_name=self.datasethr_name, datasetlr_name=self.datasetlr_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        # patch = round(self.hr_height / 2**4) ##########if input is 250 use this
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        
        # Build the generator
        self.generator = self.build_generator()

        # Build and compile the discriminator
        if(not no_dataloader):
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])

            # clean ground truth and noisy input data. img_hr -> ground truth; img_lr -> noisy input
            img_hr = Input(shape=self.hr_shape)
            img_lr = Input(shape=self.lr_shape)

            # Generate de-noised version from noisy input
            fake_hr = self.generator(img_lr)
            if(self.loss_type == 'vgg'):
                if(self.hr_channels == 3):
                    vgg_input = fake_hr
                elif(self.hr_channels == 1):
                    vgg_input = concatenate([fake_hr, fake_hr, fake_hr], axis=-1)     
                # Extract image features of the generated img
                fake_features = self.vgg(vgg_input)

                # For the combined model we will only train the generator
                self.discriminator.trainable = False

                # Discriminator determines validity of generated de-noised images
                validity = self.discriminator(fake_hr)

                self.combined = Model([img_lr, img_hr], [validity, fake_features, fake_hr,fake_hr])
                self.combined.compile(loss=['binary_crossentropy', 'mse', smooth_l1_loss, total_variation_loss],
                                loss_weights=[1e-3, 1, 0,0],metrics=[ssim, PSNR],
                                optimizer=optimizer)
                self.combined.summary()
                self.discriminator.summary()
            elif(self.loss_type == 'pixel_wise'):
                self.combined = Model([img_lr, img_hr], fake_hr)
                self.combined.compile(loss='mse', metrics=[ssim, PSNR],
                                    optimizer=optimizer)
                #self.combined.summary()


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")         # vgg19_weights_tf_dim_ordering_tf_kernels.h5
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=(self.hr_height, self.hr_width, 3))
        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        # def residual_block(layer_input, filters):
        #     """Residual block described in paper"""
        #     d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        #     # d = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(d)
        #     # d = LeakyReLU(alpha=0.2)(d)
        #     d = Activation('relu')(d)
        #     d = BatchNormalization(momentum=0.8)(d)
        #     d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        #     d = BatchNormalization(momentum=0.8)(d)
        #     d = Add()([d, layer_input])
        #     return d

        # def deconv2d(layer_input):
        #     """Layers used during upsampling"""
        #     u = UpSampling2D(size=2)(layer_input)
        #     u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        #     u = Activation('relu')(u)
        #     return u

        def standard_unit(layer_input, stage, nb_filter, kernel_size=3):
            x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(layer_input)
            x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.5, name='dp'+stage+'_1')(x)
            x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Dropout(0.5, name='dp'+stage+'_2')(x)      

            return x

        ########################################

        """
        Standard U-Net [Ronneberger et.al, 2015]
        """
        
        nb_filter = [32,64,128,256,512]

        # Handle Dimension Ordering for different backends
        global bn_axis
        # if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        # Low resolution image input
        img_input = Input(shape=(self.lr_height, self.lr_width, self.lr_channels), name='main_input')
        # else:
        #     bn_axis = 1
        #     img_input = Input(shape=(self.lr_channels, self.lr_height, self.lr_width, ), name='main_input')

        conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
        GC1 = global_context_block(conv1_1)


        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(GC1)

        conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
        GC2 = global_context_block(conv2_1)


        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(GC2)

        conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
        GC3 = global_context_block(conv3_1)


        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(GC3)

        conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
        GC4 = global_context_block(conv4_1)


        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(GC4)

        conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
        GC5 = global_context_block(conv5_1)

        up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(GC5)
        conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])
        # GC6 = global_context_block(conv4_2)

        up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
        conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])
        # GC7 = global_context_block(conv3_3)

        up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
        conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])
        # GC8 = global_context_block(conv2_4)

        up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
        conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])
        # GC9 = global_context_block(conv1_5)

        # unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
        # gen_hr = Conv2D(self.hr_channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
    
        gen_hr = Conv2D(self.hr_channels, (1, 1), activation='relu', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    
        return Model(img_input, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=100, times_d = 1, times_g = 1):

        start_time = datetime.datetime.now()
        whole_log = []
        whole_valid_log = []

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------
            for i in range(times_d):
            # Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

                # From noisy image generate de-noised version
                fake_hr = self.generator.predict(imgs_lr)

                valid = np.ones((batch_size,) + self.disc_patch)
                fake = np.zeros((batch_size,) + self.disc_patch)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            for i in range(times_g):
                # ------------------
                #  Train Generator
                # ------------------

                # Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
                if(self.loss_type == 'vgg'):
                # Extract ground truth image features using pre-trained VGG19 model
                    if(self.hr_channels == 1):
                            vgg_input2 = np.concatenate((imgs_hr, imgs_hr, imgs_hr), axis=-1)
                    elif(self.hr_channels == 3):
                            vgg_input2 = imgs_hr
                # The generators want the discriminators to label the generated images as real
                    valid = np.ones((batch_size,) + self.disc_patch)
                # Extract ground truth image features using pre-trained VGG19 model
                    image_features = self.vgg.predict(vgg_input2)

                # Train the generators
                    g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features, imgs_hr, imgs_hr])
                
                elif(self.loss_type == 'pixel_wise'):
                        g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], imgs_hr)  
            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))
            if(self.loss_type == 'vgg'):
                print("%d [G loss: %f, G_ssim: %f, G_PSNR: %f]" % (epoch, g_loss[0], g_loss[9], g_loss[10]))
            elif(self.loss_type == 'pixel_wise'):
                print("%d [G loss: %f, G_ssim: %f, G_PSNR: %f]" % (epoch, g_loss[0], g_loss[1], g_loss[2]))
            
            g_loss = [float(i) for i in g_loss]
            epoch_log = dict({'epoch': epoch, 'g_loss': g_loss})
            whole_log.append(epoch_log)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # self.sample_images(epoch)
                valid_log = self.validate(epoch,batch_size)
                self.save_model(epoch, g_loss, valid_log)
                #self.sample_images(epoch)
                self.save_log(whole_log)
                whole_valid_log.append(valid_log)
                self.save_valid_log(whole_valid_log)
        
        self.save_log(whole_log)
        valid_log = self.validate(epoch,batch_size)
        self.save_model(epoch, g_loss, valid_log)
        self.save_valid_log(whole_valid_log)


    def validate(self, epoch, batch_size):
        imgs_hr, imgs_lr = self.data_loader.load_data(is_testing=True, batch_size=batch_size)
        valid = np.ones((batch_size,) + self.disc_patch)
        if(self.hr_channels == 1):
            vgg_input = np.concatenate((imgs_hr, imgs_hr, imgs_hr), axis=-1)
        elif(self.hr_channels == 3):
            vgg_input = imgs_hr
        if(self.loss_type == 'vgg'):
            image_features = self.vgg.predict(vgg_input)
            valid_loss = self.combined.evaluate([imgs_lr, imgs_hr], [valid, image_features, imgs_hr, imgs_hr], batch_size=8)
            print("%d [val_loss: %f, val_ssim: %f, val_PSNR: %f]" % (epoch, valid_loss[0], valid_loss[9], valid_loss[10]))
        elif(self.loss_type == 'pixel_wise'):
            valid_loss = self.combined.evaluate([imgs_lr, imgs_hr], imgs_hr, batch_size=8)
            print("%d [val_loss: %f, val_ssim: %f, val_PSNR: %f]" % (epoch, valid_loss[0], valid_loss[1], valid_loss[2]))
        valid_loss = [float(i) for i in valid_loss]
        valid_log = dict({'epoch': epoch, 'val_loss': valid_loss})
       
        return valid_log

    def save_model(self, epoch, g_loss, valid_log):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            if(self.loss_type == 'vgg'):
                weights_path = "saved_model/%s_%s_G-PSNR-%s_weights.hdf5" % (model_name, str(epoch), str(valid_log['val_loss'][10]))
            elif(self.loss_type == 'pixel_wise'):
                weights_path = "saved_model/%s_%s_G-PSNR-%s_weights.hdf5" % (model_name, str(epoch), str(valid_log['val_loss'][2]))
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        # save(self.srresnet, "srresnet")
        save(self.generator, "generator")

    def save_log(self, log):
        json_log = json.dumps(log)
        log_path = "saved_model/log.json"
        with open(log_path, 'w') as f:
            f.write(json_log)
    
    def save_valid_log(sellf, log):
        json_log = json.dumps(log)
        log_path = "saved_model/valid_log.json"
        with open(log_path, 'w') as f:
            f.write(json_log)


    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.datasethr_name, exist_ok=True)
        # r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        for i in range(2):
            #  print(imgs_hr[i].shape)
             imgs_hr_save = np.squeeze(imgs_hr[i])
             scipy.misc.imsave('images/%s/%d_Original%d.jpg' % (self.datasethr_name, epoch, i), imgs_hr_save)

            
             fake_hr_save = np.squeeze(fake_hr[i])
             scipy.misc.imsave('images/%s/%d_Generated%d.jpg' % (self.datasethr_name, epoch, i), fake_hr_save)


if __name__ == '__main__':
    if(not os.path.exists('./saved_model')):
        os.makedirs('./saved_model')
    gan = SRGAN()
    gan.train(epochs=60000, batch_size=8, times_g=1, times_d=1, sample_interval=300)
