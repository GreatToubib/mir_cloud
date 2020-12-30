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

if not os.path.isdir("glcm"):
    os.mkdir("glcm")
if not os.path.isdir("sift"):
    os.mkdir("sift")
if not os.path.isdir("orb"):
    os.mkdir("orb")
if not os.path.isdir("histo"):
    os.mkdir("histo")
if not os.path.isdir("lbp"):
    os.mkdir("lbp")
imgFolderPath = "Corel100_database/"
i = 0
for filename in os.listdir(imgFolderPath):
    i += 1
    #if i == 307: continue
    if i%50==0: print(i)

    image1 = cv2.imread(imgFolderPath+filename)
    filename = filename.split(".")[0]
    sift1 = cal_SIFT(image1)
    np.save("sift/"+filename, sift1)
    orb1 = cal_ORB(image1)
    np.save("orb/"+filename, orb1)
    lbp1 = cal_LBP(image1)
    np.save("lbp/"+filename, lbp1)
    glcm1 = cal_GLCM(image1)
    np.save("glcm/"+filename, glcm1)
    histo1 = cal_Histo2(image1)
    np.save("histo/"+filename, histo1)

print("")
print("done")


