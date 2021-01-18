import math
from matplotlib import pyplot as plt

def compute_Precision_Recall_AP (top_images,file_class,topCB):

    AP=0
    prec_list = []
    rec_list= []
    previous_count=0
    for i in range(topCB):
        j=i
        countPertinent = 0
        while j >= 0:
            if int(math.floor((top_images[j]-1) / 100)) == file_class: # si l image trouvée est pertinente
                countPertinent += 1
            j -= 1
        if previous_count < countPertinent:
            AP+= (countPertinent/(i+1))*100
        if i == 49:
            AP50=AP


        precision = (countPertinent/(i+1))*100 # quantite de positif sur la quantite de predictions realisees
        recall = (countPertinent/100)*100 # quantite de positif sur la quantite d images possibles dans une classe
        prec_list.append(precision)
        rec_list.append(recall)

        previous_count = countPertinent

    AP50 = AP50/100
    AP100 = AP/100
    return AP50, AP100, prec_list, rec_list


def plot_rp(top_images, file_class, topCB):
    AP50, AP100, prec_list, rec_list = compute_Precision_Recall_AP(top_images, file_class, topCB)

    plt.plot(prec_list,rec_list,'C1', label="VGG16" )
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title("R/P")
    plt.legend()