import numpy as np
import csv
import random
import sys
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scaling import scaler
from scipy import ndimage
from funktionen import pick_start,pick_obj,paste_obj,do_label,pick_series_start,save_in_folder,paste_obj2, mutate_obj
import os
from background import generate_samples


class GeneratetDataSet:

    def __init__(self, personen, störungen, bg,bildgröße:(int,int), einzelbilder:int=0, serien:int=0, temp_range=(2902,2972)):
        self.data_list = []
        self.bg_arr = generate_samples(np.load(bg),einzelbilder+serien)
        self.person_list = personen
        self.störung_list = störungen
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

        img_dir = speicherort + "/val/images/"
        os.makedirs(img_dir, exist_ok=True)
        label_dir = speicherort + "/val/labels/"
        os.makedirs(label_dir, exist_ok=True)
        save_in_folder(val,img_dir,label_dir, name+"_val")

        img_dir = speicherort + "/test/images/"
        os.makedirs(img_dir, exist_ok=True)
        label_dir = speicherort + "/test/labels/"
        os.makedirs(label_dir, exist_ok=True)
        save_in_folder(test,img_dir,label_dir, name+"_test")


class GeneratetImage:

    def __init__(self, personen, störungen,bg, height:int,widht:int, temp_dk:int, series=False):
        self.pic_height = height
        self.pic_widht = widht
        self.temp = temp_dk  #for dezi Kelvin
        self.size = (height,widht)
        self.bg = self.generate_bg(bg,self.temp)
        self.persons = self.generate_person(personen=personen)
        self.störung = self.generate_störung(störungen=störungen)
        if series == False:
            self.image,self.label = self.generate_image()
        if series == True:
            self.image_list = self.generate_imageseries()

    def generate_bg(self,data,temp):
        # print(data)
        mean_temp = np.mean(data)
        temp_diff = temp - mean_temp
        for i in range(len(data)):
            data[i] = data[i] + temp_diff

        return data.reshape(self.pic_height,self.pic_widht)

    def generate_person(self, personen):
        person_list = []
        path = personen
        b = np.load(path, allow_pickle=True)

        for x in range(5):
            i = random.choice([a for a in range(len(b))])
            person_list.append(b[i])
        return person_list

    def generate_störung(self, störungen):
        störung_list =[]
        path = störungen
        c = np.load(path, allow_pickle=True)

        for x in range(5):
            if len(c) == 0:
                return störung_list
            i = random.choice([a for a in range(len(c))])
            störung_list.append(c[i])
        return störung_list


    def generate_image(self):
        bg = self.bg
        max_obj = 3

        breaker = False
        label_list = []
        old_h, old_w = 0,0
        for i in range(max_obj):
            person_list = self.persons
            störung_list = self.störung

            if breaker == True:
                break
            if i == 0:
                objekt = pick_obj(person_list)
                obj, padding = mutate_obj(objekt[0],objekt[1])
                start_h,start_w = pick_start(objekt=obj,pic_widht=self.pic_widht,pic_height=self.pic_height)
                label = 0
                label_list.append(
                    do_label(person=obj, pic_height=self.pic_height, pic_widht=self.pic_widht, start_h=start_h,
                             start_w=start_w, padding=padding, obj_class=label))
            else:
                zufall = random.choices(["n", "p", "s"],[0.6,0.2,0.2])
                # print(zufall)
                if zufall[0] == "p":
                    objekt = pick_obj(person_list)
                    obj, padding = mutate_obj(objekt[0],objekt[1])
                    start_h, start_w = pick_start(objekt=obj,pic_widht=self.pic_widht,pic_height=self.pic_height)
                    label = 0
                    label_list.append(do_label(person=obj, pic_height=self.pic_height, pic_widht=self.pic_widht,
                                               start_h=start_h, start_w=start_w, padding=padding, obj_class=label))
                elif zufall[0] == "s":
                    # print("in s")
                    if len(störung_list) == 0:
                        continue
                    objekt = pick_obj(störung_list)
                    obj, padding = mutate_obj(objekt[0],objekt[1])
                    start_h, start_w = pick_start(objekt=obj,pic_widht=self.pic_widht,pic_height=self.pic_height)
                    label = 0
                else:
                    continue

            height,widht = obj.shape
            bg = paste_obj(bg, obj, start_h,start_w)
            bg = paste_obj2(bg=bg, object=obj, start_h=start_h, start_w=start_w, padding=padding, temp_dk=self.temp)

            # plt.imshow(bg)
            # plt.show()
            old_h=height+old_h
            old_w=widht+old_w
            if height * widht >= bg.shape[0] * bg.shape[1] * 0.6:
                breaker = True
                break
            if (height+old_h) * (widht+old_w) >= bg.shape[0] * bg.shape[1] * 0.8:
                breaker = True
                break

        return bg,label_list


    def generate_imageseries(self):
        # print("start series")
        # bg = self.bg
        person_list = self.persons
        objekt = pick_obj(person_list)
        person,padding = objekt[0],objekt[1]
        schrittweite = 6

        start_p, end_p = pick_series_start(objekt=person,pic_height=self.pic_height,pic_widht=self.pic_widht)

        schritte_x = (self.pic_widht + person.shape[1]) / schrittweite
        schritte_y = (self.pic_height + person.shape[0]) / schrittweite
        entf = ((end_p[0] - start_p[0]) / schritte_x, (end_p[1] - start_p[1]) / schritte_y)

        # print(start_p,end_p)
        # print(entf)
        pic_list = []
        start_x = start_p[0]
        start_y = start_p[1]

        # print("start if else")
        if end_p[0] < start_p[0]:
            while (start_x > end_p[0]):
                label_list = []
                # print("start while")
                bg2 = self.bg.copy()
                # print(start_x,end_p[0])
                start_h=int(start_y)
                start_w=int(start_x)
                image = paste_obj(bg2, person, start_h=start_h, start_w=start_w)
                label_list.append(do_label(person=person, pic_height=self.pic_height, pic_widht=self.pic_widht,
                                           start_h=start_h, start_w=start_w, padding=padding, obj_class=0))
                pic_list.append([image, label_list])
                start_x = start_x + entf[0]
                start_y = start_y + entf[1]

        else:
            while (start_x < end_p[0]):
                label_list=[]
                # print("start while")
                bg2 = self.bg.copy()
                # print(start_x, end_p[0])
                start_h = int(start_y)
                start_w = int(start_x)
                image = paste_obj(bg2, person, start_h=start_h, start_w=start_w)
                label_list.append(do_label(person=person, pic_height=self.pic_height, pic_widht=self.pic_widht,
                                 start_h=start_h, start_w=start_w, padding=padding, obj_class=0))
                pic_list.append([image, label_list])
                start_x = start_x + entf[0]
                start_y = start_y + entf[1]

        return pic_list