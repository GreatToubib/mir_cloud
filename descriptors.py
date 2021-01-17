
import numpy as np
import cv2
from skimage.feature import greycomatrix
from skimage.feature import local_binary_pattern

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
import operator
import math
import os
import tensorflow as tf
from keras.models import Model

def cal_MODEL(image, model):
    feature = model.predict(image)
    feature = np.array(feature[0])
    return feature

def histogram_color(image):
    histR = cv2.calcHist([image],[0],None,[256],[0,256])
    histG = cv2.calcHist([image],[1],None,[256],[0,256])
    histB = cv2.calcHist([image],[2],None,[256],[0,256])
    hist = [histR, histG, histB]
    return hist


def cal_HISTO(image):
  hist_features = []
  if image is not None:
    des = histogram_color(image)
    hist_features.append(des)
  hist_featuresA = np.array(hist_features)
  return hist_featuresA


def siftDescriptor(image):
  sift = cv2.xfeatures2d.SIFT_create()
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)        # Convertion en niveau de gris
  keypoints = sift.detect(gray,None)
  keypoints_sift, descriptor_sift = sift.compute(gray, keypoints)
  return keypoints_sift, descriptor_sift


def cal_SIFT(image):
  if image is not None:
    kp, des = siftDescriptor(image)
  else:
    print("image is none")
  if des is not None:
    sift_features = des.tolist()
    sift_featuresA = np.array(sift_features)
    return sift_featuresA
  else :
    print( "des is none")
    return np.array([0])


def orbDescriptor(image):
  orb = cv2.ORB()
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY )        # Convertion au niveau de gris
  keypoints, descriptor_orb = orb.detectAndCompute(gray, None)
  return keypoints, descriptor_orb


def cal_ORB(image):
  orb = cv2.ORB_create()
  if image is not None:
    kp, des = orb.detectAndCompute(image, None)
  else: print ("image is none")
  if des is not None:
    orb_features = des.tolist()
    orb_featuresA = np.array(orb_features)
    return orb_featuresA
  else :
    print( "des is none")
    return np.array([0])


def lbpDescriptor(image):
# settings for LBP
  METHOD = 'uniform'
  radius = 3
  n_points = 8 * radius
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)         # Convertion au niveau de gris
  lbp = local_binary_pattern(gray, n_points, radius, METHOD)
  return lbp


def cal_LBP(image):
  if image is not None:
    des = lbpDescriptor(image)
    lbp_features = des.tolist()
  lbp_featuresA = np.array(lbp_features)
  return lbp_featuresA


def cal_GLCM(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  glcm = greycomatrix(image, distances=[1], angles=[0, np.pi / 4, np.pi / 2],
                      symmetric=True, normed=True)
  return glcm



