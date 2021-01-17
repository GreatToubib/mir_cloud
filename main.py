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

# numpy.save(file, arr)
# numpy.load(file)
# Press the green button in the gutter to run the script.
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
        "orb": cal_ORB # all good
        #"glcm": cal_GLCM, # foire a flann, mais tout foire + ou - a flann
        #"histo": cal_HISTO, # foire a flann
        #"lbp": cal_LBP # foire a euclidean
    }
    topCB = 100
    filename = "14_1475" # 1_112 150 14_1475
    imagePath = "Corel100_database/"+filename+".jpg"

    descriptorChoice1 = "vgg16"
    descriptorChoice2 = None
    desc_name = descriptorChoice1
    if descriptorChoice2 != None:
        desc_name = descriptorChoice1 + "_" + descriptorChoice2
    img = cv2.imread(imagePath)
    print(" filename : ",filename)

    print(" desc_name : ", desc_name)
    image1 = load_img(imagePath, target_size=(224, 224))
    image1 = img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))


    if descriptorChoice1 in desc_dict :
        des_of_input_img = desc_dict[descriptorChoice1](img)
    else:
        print("in deepl des")
        des_of_input_img = cal_MODEL(image1,model_dict[descriptorChoice1])

    if descriptorChoice2 != None:
        if descriptorChoice2 in desc_dict:
            des2_of_input_img = desc_dict[descriptorChoice2](img)
        else:
            des2_of_input_img = cal_MODEL(image1, model_dict[descriptorChoice2])
        des_of_input_img = combiner_des(des_of_input_img, des2_of_input_img)

    results_dict = get_all_tops(descriptorChoice1, descriptorChoice2, des_of_input_img, topCB,filename)

    """des_index_image2 = np.load("features/" + descriptorChoice1 + "/" + "14_1475.npy")
    print(flann(des_of_input_img, des_index_image2))
    print(euclidean(des_of_input_img, des_index_image2))
    print(bruteForceMatching(des_of_input_img, des_index_image2))
"""
