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
from metrics import *

def guess_file_class(filename, best_model,best_model2):

    topCB = 100
    #imagePath = "uploaded_images/" + filename
    imagePath = "uploads/" + filename
    distFunction = chi2_distance

    image1 = load_img(imagePath, target_size=(224, 224))
    image1 = img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    des_of_input_img = cal_MODEL(image1, best_model)
    des2_of_input_img = cal_MODEL(image1, best_model2)
    #des_of_input_img = combiner_des(des_of_input_img, des2_of_input_img)
    descriptorChoice1 = "vgg16"
    descriptorChoice2 = None
    list_distances = compute_distances(descriptorChoice1, descriptorChoice2, des_of_input_img, distFunction)
    top_images = get_closest_images(100, list_distances)
    top_classes=[]
    for index in top_images:
        top_classes.append(int(math.floor((index-1) / 100)))
    file_class = max(set(top_classes), key=top_classes.count)
    return file_class


def get_search_results(descriptorChoice1,descriptorChoice2,distName,filename,file_class):
    print(filename)
    print("loading models")
    model_vgg16 = vgg16.VGG16(weights='imagenet', include_top=True, pooling='avg')
    model_vgg19 = vgg19.VGG19(weights='imagenet', include_top=True, pooling='avg')
    model_resnet50 = resnet50.ResNet50(weights='imagenet', include_top=True, pooling='avg')
    model_inception_v3 = inception_v3.InceptionV3(weights='imagenet', include_top=True, pooling='avg')
    model_xception = xception.Xception(weights='imagenet', include_top=True, pooling='avg')

    model_dict = {
        "inception_v3": model_inception_v3,
        "resnet50": model_resnet50,
        "vgg19": model_vgg19,
        "vgg16": model_vgg16,
        "xception": model_xception
    }
    dist_dict={
            "euclidean": euclidean,
            "chi2_distance" : chi2_distance,
            "bhatta" : bhatta
        }

    topCB = 100
    imagePath = "uploads/" + filename
    distFunction = dist_dict[distName]
    if file_class == None:
        print("gonna guess class")
        file_class = guess_file_class(filename, model_vgg16, model_resnet50)
        print("guessed class = ", file_class)

    # extraire les features de l'image en input
    if descriptorChoice2 == 0:
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

    print("gonna get results")
    list_distances = compute_distances(descriptorChoice1, descriptorChoice2, des_of_input_img, distFunction)
    top_images = get_closest_images(100, list_distances)

    AP50, AP100, prec_list, rec_list = compute_Precision_Recall_AP(top_images, file_class, topCB)
    print("== " + distName + " precision: " + str(prec_list[-1]) + " recall: " + str(rec_list[-1]) + " AP50: " + str(
        AP50) + " AP100: " + str(AP100))



    return AP50, AP100, prec_list, rec_list, top_images

 # 0_8 1_173 3_323 4_443 6_638  23_2328  40_4098 54_5498 76_7628
AP50, AP100, prec_list, rec_list, top_images = get_search_results("resnet50",None,"euclidean","54_5498.jpg",None)
print(top_images[0:20])
P50 = prec_list[49]
P100 = prec_list[-1]
R50 = rec_list[49]
R100 = rec_list[-1]

