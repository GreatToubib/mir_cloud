
import numpy as np
import cv2
import math
import os


def euclidean(l1, l2):
    n = min(len(l1), len(l2))
    return np.sqrt(np.sum((l1[:n] - l2[:n])**2))


def chiSquareDistance(l1, l2): # il faut normaliser le descripteur avant d appeler chiSquare
    n = min(len(l1), len(l2))
    return np.sum((l1[:n] - l2[:n])**2 / l2[:n])


def bhatta(l1, l2):
    n = min(len(l1), len(l2))
    N_1, N_2 = np.sum(l1[:n])/n, np.sum(l2[:n])/n
    score = np.sum(np.sqrt(l1[:n] * l2[:n]))
    num = round(score, 2)
    den = round(math.sqrt(N_1*N_2*n*n), 2)
    return math.sqrt( 1 - num / den )


def flann(a,b):
  a = np.float32(a)
  b = np.float32(b)
  FLANN_INDEX_KDTREE = 1
  INDEX_PARAM_TREES = 5
  SCH_PARAM_CHECKS = 50
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
  sch_params = dict(checks=SCH_PARAM_CHECKS)
  flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
  matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
  return np.mean(matches)


def bruteForceMatching(a, b):
    a = a.astype('uint8')
    b = b.astype('uint8')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)


#2 images + 2 descpteur, + methode de calcule de distance !
def combiner(image1, image2, Descp1,Descp2):
  DescpA1 =Descp1(image1)
  DescpA2 =Descp1(image2)
  DescpB1 =Descp2(image1)
  DescpB2 =Descp2(image2)

  combined_Features1= np.concatenate((DescpA1, DescpB1), axis=None)
  combined_Features2= np.concatenate((DescpA2, DescpB2), axis=None)

  return combined_Features1, combined_Features2

def combiner_2(DescpA1 ,DescpA2 ,DescpB1 ,DescpB2 ):

  combined_Features1= np.concatenate((DescpA1, DescpB1), axis=None)
  combined_Features2= np.concatenate((DescpA2, DescpB2), axis=None)

  return combined_Features1, combined_Features2


def compute_distances(descriptorChoice, des, cal_dist):
    print("starting to compute distances")
    list_distances=[]
    i=0
    for filename in os.listdir(descriptorChoice+"/"):
        i+=1
        if i % 100 ==0: print(i)
        des_index_image = np.load(descriptorChoice+"/"+filename)
        distance = cal_dist(des,des_index_image)
        list_distances.append(distance)

    return list_distances


def top_images(topCB, list_distances):
    lst_index=[]
    i=0
    while i < topCB:
        i+=1
        max_value = max(list_distances)  # get max value
        max_index = list_distances.index(max_value)  # get its index
        lst_index.append(max_index)
        list_distances.pop(max_index)
    return lst_index