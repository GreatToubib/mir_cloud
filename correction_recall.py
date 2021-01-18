import json

results_dict = json.load(open("data.json"))
distName="euclidean"
for filename in ["1_112", "1_150", "1_193", "14_1475", "14_1437", "14_1430", "26_2606", "26_2681", "26_2609", "39_3970",
                 "39_3904", "39_3916", "49_4991", "49_4958", "49_4932"]:

    for desc_name in ["mobilenet","resnet50","vgg16","sift","orb","glcm","histo","lbp" ] :

        results_dict[filename][desc_name][distName+"_recall50"] =  results_dict[filename][desc_name][distName+"_precision50"]/2

a_file = open("data2.json", "w")
json.dump(results_dict, a_file)
a_file.close()

