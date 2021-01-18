
import numpy as np
import cv2
import math
import os
import json
from pathlib import Path
import operator
from metrics import *


def euclidean(l1, l2):
    n = min(len(l1), len(l2))
    return np.sqrt(np.sum((l1[:n] - l2[:n])**2))


def chiSquareDistance(l1, l2): # il faut normaliser le descripteur avant d appeler chiSquare
    n = min(len(l1), len(l2))
    return np.sum((l1[:n] - l2[:n])**2 / l2[:n])


def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                        for (a, b) in zip(A, B)])
    return chi

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


# 2 images + 2 descripteurs
def combiner_des_cal(image1, image2, Descp1,Descp2):
  DescpA1 =Descp1(image1)
  DescpA2 =Descp1(image2)
  DescpB1 =Descp2(image1)
  DescpB2 =Descp2(image2)

  combined_Features1= np.concatenate((DescpA1, DescpB1), axis=None)
  combined_Features2= np.concatenate((DescpA2, DescpB2), axis=None)

  return combined_Features1, combined_Features2


def combiner_des(DescpA ,DescpB):

  combined_Features= np.concatenate((DescpA, DescpB), axis=None)

  return combined_Features


def compute_distances(descriptorChoice1,descriptorChoice2, des, cal_dist):

    feat_folder = "features/"
    list_distances = []
    i=0
    while i <10000:
        i+=1
        filename1=feat_folder+descriptorChoice1+"/"+str(math.floor((i-1)/100))+"_"+str(i)+".npy"
        my_file = Path(filename1)
        if not my_file.exists():
            continue
        des_index_image = np.load(filename1)
        if descriptorChoice2 != None:
            filename2 = feat_folder + descriptorChoice2 + "/" + str(math.floor((i - 1) / 100)) + "_" + str(i) + ".npy"
            des_index_image2 = np.load(filename2)
            des_index_image = combiner_des(des_index_image, des_index_image2)
        distance = cal_dist(des,des_index_image)
        list_distances.append((distance, i))

    return list_distances


def get_closest_images(topCB, list_distances):
    list_distances.sort(key=operator.itemgetter(0))
    top_images = []
    for i in range(topCB):
        top_images.append(list_distances[i][1])
    return top_images


def get_acc(top_images,file_class,topCB):
    count=0
    for index in top_images:
        if int(math.floor((index-1)/100)) == file_class:
            count+=1
    return count/topCB


def get_all_tops(descriptorChoice1, descriptorChoice2, des_of_input_img, filename, topCB,features_type):
    results_dict = {}
    results_top_dict = {}
    desc_name = descriptorChoice1
    file_class = int(filename.split("_")[0])
    if descriptorChoice2 != None :
        desc_name = descriptorChoice1+"_"+descriptorChoice2
    my_file = Path("data.json")
    if my_file.exists():
        #print("opening existing data.json")
        results_dict = json.load(open("data.json"))
        results_top_dict = json.load(open("data_top_lists.json"))

        if filename not in results_dict:
            results_dict[filename] = {}
            results_dict[filename][desc_name] = {}

            results_top_dict[filename] = {}
            results_top_dict[filename][desc_name] = {}

        if desc_name not in results_dict[filename]:
            results_dict[filename][desc_name] = {}
            results_top_dict[filename][desc_name] = {}
    else:
        results_dict[filename] = {}
        results_dict[filename][desc_name] = {}
        results_top_dict[filename] = {}
        results_top_dict[filename][desc_name] = {}

    dist_dict={
        #"euclidean": euclidean,
        "chi2_distance" : chi2_distance,
        #"bhatta" : bhatta
    }
    if features_type=="classic":
        dist_dict = {
            #"euclidean": euclidean,
            #"bhatta": bhatta
        }


    for distName, distFunction in dist_dict.items():

        list_distances = compute_distances(descriptorChoice1, descriptorChoice2, des_of_input_img, distFunction)
        top_images = get_closest_images(100, list_distances)
        AP50, AP100, prec_list, rec_list = compute_Precision_Recall_AP(top_images, file_class, topCB)
        print("== " + distName +" precision: " + str(prec_list[-1]) + " recall: " + str(rec_list[-1])+ " AP50: " + str(AP50)+ " AP100: " + str(AP100))
        results_dict[filename][desc_name][distName+"_precision50"] = prec_list[49]
        results_dict[filename][desc_name][distName+"_precision100"] = prec_list[-1]
        results_dict[filename][desc_name][distName+"_recall50"] = rec_list[49]
        results_dict[filename][desc_name][distName+"_recall100"] = rec_list[-1]
        results_dict[filename][desc_name][distName+"_AP50"] = AP50
        results_dict[filename][desc_name][distName+"_AP100"] = AP100
        results_top_dict[filename][desc_name][distName] = top_images


    a_file = open("data.json", "w")
    json.dump(results_dict, a_file)
    a_file.close()
    a_file = open("data_top_lists.json", "w")
    json.dump(results_top_dict, a_file)
    a_file.close()

    return results_dict
