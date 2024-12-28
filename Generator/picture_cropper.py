import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from statistics import mean
import json
import re


BILDER = r"images"
BILDER2 = r"train\TXT_pic"

LABEL = r"labels"
LABEL2 =r"train\labels"

# 0 0.14182692307692307 0.6923076923076923 0.10336538461538461 0.375
# class x y (center of rectangle) widht heigt


BILDER_LIST = os.listdir(BILDER)
BILDER_LIST2 = os.listdir(BILDER2)

LABEL_LIST = os.listdir(LABEL)
LABEL_LIST2 = os.listdir(LABEL2)

BILDER_LIST.sort()
LABEL_LIST.sort()

BILDER_LIST2.sort()
LABEL_LIST2.sort()

bilder3 = zip(BILDER_LIST2,LABEL_LIST2)

for eins, zwei in bilder3:
    if re.match(eins, zwei):
        pass
    else:
        print(f"{eins} ungleich {zwei}")


# bilder2=[]
# for i in range(len(BILDER_LIST2)):
#     if ".txt" in BILDER_LIST2[i]:
#         bilder2.append(BILDER_LIST2[i])
#     bilder2.sort()
# for i in range(len(bilder2)):
#     if re.match(bilder2[i], LABEL_LIST2[i]):
#         pass
#     else:
#         print(f"bilder2: {bilder2[i]} <--> {LABEL_LIST2[i]} :label2  <-->  {BILDER_LIST2[i]} bilderliste")
#         print(f"i: {i}")




def read_img_txt(adress):
    with open(adress) as data:
        line = data.readline()
        temp = line.split(",")[:1024]
        # print(temp)
        nd_temp = np.asarray(temp, int)
        nd_temp2 = nd_temp.reshape((32, 32))
        # plt.imshow(nd_temp2)
        # plt.show()
    return nd_temp2

def read_img_jpg(adress):
    image = cv2.imread(adress)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def crop_roi(adress):
    desired_pad = 3

    with open(adress) as f:
        lines = f.readline()
        if ", " in lines:
            split = lines.split(", ")
        else:
            split = lines.split(" ")
    # print(f" adresse ist {adress} und split ist:{split}")
    # print(split)
    label = split[0]

    x = float(split[1])*image.shape[1]
    y = float(split[2])*image.shape[0]
    widht = float(split[3])*image.shape[1]
    height = float(split[4]) *image.shape[0]

    padding = [0,0,0,0]
    for p_o in range(-desired_pad,1):
        for p_r in range(-desired_pad,1):
            for p_u in range(-desired_pad, 1):
                for p_l in range(-desired_pad, 1):
                    if (int(y - height / 2) + p_o >= 0) and (int(x - widht / 2) +p_l >= 0) and (int(y + height / 2) -p_u <= 32) and (int(x + widht / 2) -p_r <= 32):
                        if -p_o >= padding[0]:
                            padding[0] = -p_o
                        if -p_r >= padding[1]:
                            padding[1] = -p_r
                        if -p_u >= padding[2]:
                            padding[2] = -p_u
                        if -p_l >= padding[3]:
                            padding[3] = -p_l
                        break

    person = image[int(y - height / 2) - padding[0]:int(y + height / 2) + padding[2], int(x - widht / 2) - padding[3]:int(x + widht / 2) + padding[1]]

    # start_point = (int(x-widht/2),int(y-height/2))
    # end_point = (int(x+widht/2),int(y+height/2))
    # color = (0,0,255)
    # thickness = 1
    #
    # pic = image.copy()
    # image2 = cv2.rectangle(pic, start_point, end_point, color, thickness)
    # plt.imshow(image2)
    # plt.show()
    # plt.imshow(person)
    # plt.show()

    return label,person,padding

all_persons = []
all_störung = []

all_persons2 = []
all_störung2 = []

for i in range(len(BILDER_LIST)):
    # if ".jpg" in BILDER_LIST[i]:
    #     image = read_img_jpg(BILDER + "/" + BILDER_LIST[i])
    if ".txt" in BILDER_LIST[i]:
        image = read_img_txt(BILDER + "/" + BILDER_LIST[i])
        # print("break")
        # plt.imshow(image)
        # plt.show()



        label,person,padding = crop_roi(LABEL + "/" + LABEL_LIST[i])

        # ##############################breite*höhe
        # p_height,p_widht = person.shape
        # start_lo = (padding[3],padding[0])
        # end_ru = (p_widht-padding[1],p_height-padding[2])
        # person = cv2.rectangle(person,start_lo , end_ru, (0,0,255), 1)
        #
        # fig, axs = plt.subplots(1, 2, figsize=(14, 8))
        # plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
        #
        # axs[0].imshow(image)  # .astype(np.uint8)
        # axs[1].imshow(person)  # .astype(np.uint8)
        # plt.show()

        if label =="1":
            all_persons.append([person, np.asarray(padding)])
            all_persons2.append([person.tolist(), padding])
        else:
            all_störung.append([person, np.asarray(padding)])
            all_störung2.append([person.tolist(), padding])

# for person in all_persons:
    # print(type(person))
    # print(person[0])
    # print(np.amax(person[0]))
    # break
    # print(mean(person[0]))
    # print(max(person[0]))

for n in range(len(BILDER_LIST2)):
    if ".txt" in BILDER_LIST2[n]:
        image = read_img_txt(BILDER2 + "/" + BILDER_LIST2[n])

        label, person, padding = crop_roi(LABEL2 + "/" + LABEL_LIST2[n])

        if label == "0":
            all_persons.append([person, np.asarray(padding)])
            all_persons2.append([person.tolist(), padding])
        else:
            all_störung.append([person, np.asarray(padding)])
            all_störung2.append([person.tolist(), padding])


np.save('persons2.npy', all_persons, allow_pickle=True)
np.save("störung2.npy", all_störung, allow_pickle=True)

with open("persons.json", "w") as data:
    json.dump(all_persons2,data)

with open("interference.json", "w") as data:
    json.dump(all_störung2,data)