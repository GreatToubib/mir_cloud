import json
map_dict={}
results_folder= "results_euclid_8basic"
results_dict = json.load(open(results_folder+"/data.json"))
distName="euclidean"
for desc_name in ["mobilenet","resnet50","vgg16","sift","orb","glcm","histo","lbp" ] :
    map_dict[desc_name]={}
    for distName in ["euclidean"]:
        map_value50=0
        map_value100 = 0
        map_dict[desc_name][distName]={}
        for filename in ["1_112", "1_150", "1_193", "14_1475", "14_1437", "14_1430", "26_2606", "26_2681", "26_2609", "39_3970",
                 "39_3904", "39_3916", "49_4991", "49_4958", "49_4932"]:

            map_value50 += results_dict[filename][desc_name][distName+"_AP50"]
            map_value100 += results_dict[filename][desc_name][distName + "_AP100"]

        map_value50 = map_value50/15
        map_value100 = map_value100/15
        map_dict[desc_name][distName]["map50"] = map_value50
        map_dict[desc_name][distName]["map100"] = map_value100
a_file = open(results_folder+"/map.json", "w")
json.dump(map_dict, a_file)
a_file.close()

