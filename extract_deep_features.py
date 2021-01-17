# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import xception
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import Model, load_model
from keras.preprocessing.image import load_img
from matplotlib.pyplot import imread
import numpy as np
import cv2
from skimage.feature import greycomatrix
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb

def cal_model(image, model):
    feature = model.predict(image)
    feature = np.array(feature[0])
    return feature

model_vgg16 = vgg16.VGG16(weights='imagenet', include_top=True,pooling='avg')
model_resnet50 = resnet50.ResNet50(weights='imagenet', include_top=True,pooling='avg')
model_mobilenet = mobilenet.MobileNet(weights='imagenet', include_top=True,pooling='avg')

print("starting main")

if not os.path.isdir("features/vgg16"):
    os.mkdir("features/vgg16")
if not os.path.isdir("features/resnet50"):
    os.mkdir("features/resnet50")
if not os.path.isdir("features/mobilenet"):
    os.mkdir("features/mobilenet")
imgFolderPath = "Corel100_database/"
i = 0

for filename in os.listdir(imgFolderPath):
    image1 = load_img(imgFolderPath + "/" + filename, target_size=(224, 224))
    filename = filename.split(".")[0]
    image1 = img_to_array(image1)
    image1 = image1.reshape((1, image1.shape[0], image1.shape[1], image1.shape[2]))
    mvgg16 = cal_model(image1, model_vgg16)
    np.save("features/vgg16/"+filename, mvgg16)
    mresnet50 = cal_model(image1, model_resnet50)
    np.save("features/resnet50/" + filename, mresnet50)
    mmobilenet = cal_model(image1, model_mobilenet)
    np.save("features/mobilenet/" + filename, mmobilenet)

print("")
print("done")