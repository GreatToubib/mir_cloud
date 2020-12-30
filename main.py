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
from distances import *

#numpy.save(file, arr)
#numpy.load(file)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    imagePath = "Corel100_database/3_306.jpg"
    img = cv2.imread(imagePath)
    descriptorChoice = "sift"
    if descriptorChoice == "sift":
        cal_DES = cal_SIFT
    if descriptorChoice == "glcm":
        cal_DES = cal_GLCM
    if descriptorChoice == "histo":
        cal_DES = cal_HISTO
    if descriptorChoice == "lbp":
        cal_DES = cal_LBP
    if descriptorChoice == "orb":
        cal_DES = cal_ORB

    des = cal_DES(img)
    list_distances = compute_distances(descriptorChoice, des, euclidean)
    print(list_distances)

    top_images= top_images(20, list_distances)
    print(top_images)
    """print("starting main")
    image1 = cv2.imread("Corel100_database/3_307.jpg")
    image2 = cv2.imread("Corel100_database/9_950.jpg")

    sift1 = cal_SIFT(image1)
    orb1 = cal_ORB(image1)
    lbp1 = cal_LBP(image1)
    glcm1 = cal_GLCM(image1)
    histo1 = cal_Histo2(image1)

    print("combined:")
    combined_Features1, combined_Features2 = combiner(image1, image2, cal_LBP, cal_Histo2)
    euclidean_LBPHisto = euclidean(combined_Features1, combined_Features2)
    print("euclidean_LBP+Histo: ", euclidean_LBPHisto)
    bhatta_LBPHisto = bhatta(combined_Features1, combined_Features2)
    print("bhatta_LBP+Histo: ", bhatta_LBPHisto)

    combined_Features1, combined_Features2 = combiner(image1, image2, cal_SIFT, cal_ORB)
    brute_Siftorb = bruteForceMatching(combined_Features1, combined_Features2)
    print("brute_Sift+orb: ", brute_Siftorb)
    flann_Siftorb = flann(combined_Features1, combined_Features2)
    print("flann_Sift+orb: ", flann_Siftorb)

    combined_Features1, combined_Features2 = combiner(image1, image2, cal_LBP, cal_GLCM)
    euclidean_LBPGLCM = euclidean(combined_Features1, combined_Features2)
    print("euclidean_LBP+GLCM: ", euclidean_LBPGLCM)
    bhatta_LBPGLCM = bhatta(combined_Features1, combined_Features2)
    print("bhatta_LBP+GLCM: ", bhatta_LBPGLCM)"""
