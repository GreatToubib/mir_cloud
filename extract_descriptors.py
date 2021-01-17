from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.models import Model, load_model
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

from descriptors import *

# numpy.save(file, arr)
# numpy.load(file)
# Press the green button in the gutter to run the script.

print("starting main")
imgFolderPath = "Corel100_database/"  # test_images
feat_folder = "features/"  # test_features
if not os.path.isdir(feat_folder+"glcm"):
    os.mkdir(feat_folder+"glcm")
if not os.path.isdir(feat_folder+"sift"):
    os.mkdir(feat_folder+"sift")
if not os.path.isdir(feat_folder+"orb"):
    os.mkdir(feat_folder+"orb")
if not os.path.isdir(feat_folder+"histo"):
    os.mkdir(feat_folder+"histo")
if not os.path.isdir(feat_folder+"lbp"):
    os.mkdir(feat_folder+"lbp")

i = 0
for filename in os.listdir(imgFolderPath):
    i += 1
    #if i == 307: continue
    if i%50 == 0 : print(i)

    image1 = cv2.imread(imgFolderPath+filename)
    filename = filename.split(".")[0]
    sift1 = cal_SIFT(image1)
    np.save(feat_folder+"sift/"+filename, sift1)
    orb1 = cal_ORB(image1)
    np.save(feat_folder+"orb/"+filename, orb1)
    lbp1 = cal_LBP(image1)
    np.save(feat_folder+"lbp/"+filename, lbp1)
    glcm1 = cal_GLCM(image1)
    np.save(feat_folder+"glcm/"+filename, glcm1)
    histo1 = cal_HISTO(image1)
    np.save(feat_folder+"histo/"+filename, histo1)

print("")
print("done")


