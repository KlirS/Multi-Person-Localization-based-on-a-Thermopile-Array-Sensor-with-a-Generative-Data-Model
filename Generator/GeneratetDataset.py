import numpy as np
import random
import pandas as pd
import os
import json
import cmcrameri as cm

# eigene
from funktionen import save_in_folder, save_COCO
from background import generate_samples
from GeneratetImage import GeneratetImage


class GeneratetDataSet:

    def __init__(self, personen, störungen, bg, bildgröße:(int,int), einzelbilder:int=0, serien:int=0, temp_range=(2902,2972)):
        self.data_list_g = []
        self.data_list = []
        self.bg_arr = generate_samples(np.load(bg),einzelbilder+serien)
        self.person_list = personen
        self.störung_list = np.load(störungen,allow_pickle=True)
        self.bg_list = bg
        self.available_bg = [x for x in range(len(self.bg_arr))]

        print(f" {einzelbilder} Einzelbilder werden erstellt")
        for i in range(einzelbilder):
            self._add_data(height=bildgröße[0], widht=bildgröße[1], temp_range=temp_range)
        print(f" {serien} Bildserien werden erstellt")
        for i in range(serien):
            self._add_series(height=bildgröße[0], widht=bildgröße[1], temp_range=temp_range)

    def _add_data(self, height:int, widht:int, temp_range:(int,int)):
        temp_dk = random.choice([x for x in range(temp_range[0],temp_range[1]+1)])
        choice = self.available_bg.pop(random.choice([x for x in range(len(self.available_bg))]))
        image = GeneratetImage(personen= self.person_list,störungen=self.störung_list,bg=self.bg_arr[choice], height=height, widht=widht, temp_dk=temp_dk)
        self.data_list.append([image.image,image.label])

    def _add_series(self, height:int, widht:int, temp_range:(int,int)):
        temp_dk = random.choice([x for x in range(temp_range[0], temp_range[1] + 1)])
        choice = self.available_bg.pop(random.choice([x for x in range(len(self.available_bg))]))
        series = GeneratetImage(personen=self.person_list, störungen=self.störung_list,bg=self.bg_arr[choice], height=height, widht=widht, temp_dk=temp_dk, series=True)
        for image in series.image_list:
            self.data_list.append(image)

    def delete_data(self):
        self.data_list = []

    def len_data(self):
        print(len(self.data_list))

    def save_dataset(self,speicherort,name:str):

        df = pd.DataFrame(self.data_list)
        train, val, test = np.split(df.sample(frac=1), [int(.7 * len(df)), int(.9 * len(df))])

        train = train.to_numpy()
        val = val.to_numpy()
        test = test.to_numpy()

        img_dir = speicherort + "/train/images/"
        os.makedirs(img_dir, exist_ok=True)
        label_dir = speicherort + "/train/labels/"
        os.makedirs(label_dir, exist_ok=True)
        save_in_folder(train,img_dir,label_dir, name+"_train")
        save_COCO(train, img_dir, name + "_train")

        img_dir = speicherort + "/val/images/"
        os.makedirs(img_dir, exist_ok=True)
        label_dir = speicherort + "/val/labels/"
        os.makedirs(label_dir, exist_ok=True)
        save_in_folder(val,img_dir,label_dir, name+"_val")
        save_COCO(val, img_dir, name + "_val")

        img_dir = speicherort + "/test/images/"
        os.makedirs(img_dir, exist_ok=True)
        label_dir = speicherort + "/test/labels/"
        os.makedirs(label_dir, exist_ok=True)
        save_in_folder(test,img_dir,label_dir, name+"_test")
        save_COCO(val, img_dir, name + "_test")



# import time
#
#
# start = time.time()
#
# DATASETLOCATION = r"generatet"
# IMAGESIZE = (32,32)
#
# EINZELBILDER = 20
# SERIEN = 0
#
# data_set_1 = GeneratetDataSet(personen="persons.npy", störungen="störung.npy", bg = "bg.npy", bildgröße=IMAGESIZE, einzelbilder=EINZELBILDER, serien=SERIEN)
#
# data_set_1.save_dataset(DATASETLOCATION, name="set2")
#
#
#
# end = time.time()
# delta = end - start
# print("took %.2f seconds to process" % delta)
#
# #           python train.py --img 32 --batch 64 --epochs 200 --weights C:\Users\fglt\Desktop\code\best.pt  --name genrated_1 --data C:\Users\fglt\Desktop\code\generatet\data.yaml