import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, datasethr_name, datasetlr_name, img_res=(256, 256)):
        self.datasethr_name = datasethr_name
        self.datasetlr_name = datasetlr_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        if (is_testing==False):
            #path_hr = glob('./datasets/%s/*' % (self.datasethr_name))
            path_lr = glob('./datasets/training_examples/train_aline_forTrain_part/%s/*' % (self.datasetlr_name))

            #batch_images_hr = np.random.choice(path_hr, size=batch_size)
            batch_images_lr = np.random.choice(path_lr, size=batch_size)
            batch_images_hr = []
            for image_lr in batch_images_lr:
                image_name = image_lr.split('/')[-1]
                batch_images_hr.append('./datasets/training_examples/train_aline_forTrain_part/%s/'%(self.datasethr_name) + image_name)

            imgs_hr = []
            imgs_lr = []

            for i in range(batch_size):
                img_hr = np.load(batch_images_hr[i])
                if(img_hr.shape[0] == 250):
                    img_hr2 = np.zeros(self.img_res)
                    img_hr2[3:253,3:253] = img_hr
                img_lr = np.load(batch_images_lr[i])
                if(img_lr.shape[0] == 250):
                    img_lr2 = np.zeros(self.img_res)
                    img_lr2[3:253,3:253] = img_lr
                    imgs_hr.append(img_hr2)
                    imgs_lr.append(img_lr2)
                else:
                    imgs_hr.append(img_hr.astype(np.float32))
                    imgs_lr.append(img_lr.astype(np.float32))
            
            
            imgs_hr = np.array(imgs_hr)

            imgs_hr = imgs_hr.reshape((imgs_hr.shape[0],imgs_hr.shape[1],imgs_hr.shape[2],1))
            # imgs_hr = imgs_hr*2 - 1
            imgs_lr = np.array(imgs_lr)

            imgs_lr = imgs_lr.reshape((imgs_lr.shape[0],imgs_lr.shape[1],imgs_lr.shape[2],1))
            # imgs_lr = imgs_lr*2 - 1

        if (is_testing==True):
            #path_hr = glob('./datasets/%s/*' % (self.datasethr_name))
            path_lr = glob('./datasets/training_examples/test_aline_forTrain/%s/*' % (self.datasetlr_name))

            #batch_images_hr = np.random.choice(path_hr, size=batch_size)
            batch_images_lr = np.random.choice(path_lr, size=batch_size)
            batch_images_hr = []
            for image_lr in batch_images_lr:
                image_name = image_lr.split('/')[-1]
                batch_images_hr.append('./datasets/training_examples/test_aline_forTrain/%s/'%(self.datasethr_name) + image_name)

            imgs_hr = []
            imgs_lr = []

            for i in range(batch_size):
                img_hr = np.load(batch_images_hr[i])
                if(img_hr.shape[0] == 250):
                    img_hr2 = np.zeros(self.img_res)
                    img_hr2[3:253,3:253] = img_hr
                img_lr = np.load(batch_images_lr[i])
                if(img_lr.shape[0] == 250):
                    img_lr2 = np.zeros(self.img_res)
                    img_lr2[3:253,3:253] = img_lr
                    imgs_hr.append(img_hr2)
                    imgs_lr.append(img_lr2)
                else:
                    imgs_hr.append(img_hr.astype(np.float32))
                    imgs_lr.append(img_lr.astype(np.float32))
            
            
            imgs_hr = np.array(imgs_hr)

            imgs_hr = imgs_hr.reshape((imgs_hr.shape[0],imgs_hr.shape[1],imgs_hr.shape[2],1))
            # imgs_hr = imgs_hr*2 - 1
            imgs_lr = np.array(imgs_lr)

            imgs_lr = imgs_lr.reshape((imgs_lr.shape[0],imgs_lr.shape[1],imgs_lr.shape[2],1))
            # imgs_lr = imgs_lr*2 - 1

        return imgs_hr, imgs_lr


    
