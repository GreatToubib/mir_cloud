from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
import cv2
import os
import csv
import math
from skimage.feature import greycomatrix
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import xception
from matplotlib.pyplot import imread

from descriptors import *
from distances import *

if __name__ == '__main__':

    model_vgg16 = vgg16.VGG16(weights='imagenet', include_top=True, pooling='avg')
    #model_vgg19 = vgg19.VGG19(weights='imagenet', include_top=True, pooling='avg')
    model_resnet50 = resnet50.ResNet50(weights='imagenet', include_top=True, pooling='avg')
    #model_inception_v3 = inception_v3.InceptionV3(weights='imagenet', include_top=True, pooling='avg')
    model_mobilenet = mobilenet.MobileNet(weights='imagenet', include_top=True, pooling='avg')
    #model_xception = xception.Xception(weights='imagenet', include_top=True, pooling='avg')
    
    model_dict = {
        #"inception_v3": model_inception_v3,
        "mobilenet": model_mobilenet,
        "resnet50": model_resnet50,
        #"vgg19": model_vgg19,
        "vgg16": model_vgg16
        #"xception": model_xception,
    }

    desc_dict = {
        "sift": cal_SIFT, # all good
        "orb": cal_ORB, # all good
        "glcm": cal_GLCM, # foire a flann, mais tout foire + ou - a flann
        "histo": cal_HISTO, # foire a flann
        "lbp": cal_LBP # foire a euclidean
    }
    topCB = 100
    for filename in ["1_112","1_150","1_193", "14_1475","14_1437","14_1430","26_2606", "26_2681", "26_2609", "39_3970", "39_3904","39_3916", "49_4991", "49_4958", "49_4932"]:
    #for filename in ["1_112"]: # ,"6_650","7_793","2_212","3_350","5_593"
        imagePath = "Corel100_database/" + filename + ".jpg"
        print("=================================== filename :  : ", filename)

        for descriptorChoice1 in desc_dict:
            descriptorChoice2 = None
            desc_name = descriptorChoice1
            if descriptorChoice2 != None:
                desc_name = descriptorChoice1 + "_" + descriptorChoice2
            img = cv2.imread(imagePath)
            print( "======= : ", desc_name)
            des_of_input_img = desc_dict[descriptorChoice1](img)
            if descriptorChoice2 != None:
                des2_of_input_img = desc_dict[descriptorChoice2](img)
                des_of_input_img = combiner_des(des_of_input_img, des2_of_input_img)
            results_dict = get_all_tops(descriptorChoice1, descriptorChoice2, des_of_input_img,filename,topCB,features_type="classic")

        for descriptorChoice1 in model_dict:
            descriptorChoice2 = None
            desc_name = descriptorChoice1
            if descriptorChoice2 != None:
                desc_name = descriptorChoice1 + "_" + descriptorChoice2
            print("=======  : ", desc_name)
            image1 = load_img(imagePath, target_size=(224, 224))
            image1 = img_to_array(image1)
            image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
            des_of_input_img = cal_MODEL(image1, model_dict[descriptorChoice1])
            if descriptorChoice2 != None:
                des2_of_input_img = cal_MODEL(image1, model_dict[descriptorChoice2])
                des_of_input_img = combiner_des(des_of_input_img, des2_of_input_img)
            results_dict = get_all_tops(descriptorChoice1, descriptorChoice2, des_of_input_img, filename, topCB, features_type="deep")


