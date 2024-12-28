import numpy as np
import random
import matplotlib.pyplot as plt

#eigene
from funktionen import pick_start,pick_obj,paste_obj,do_label,pick_series_start,paste_obj2, mutate_obj, gen_stoer, mutate_temp_p, gen_hot_st



class GeneratetImage:

    def __init__(self, personen, störungen,bg, height:int,widht:int, temp_dk:int, series=False):
        self.pic_height = height
        self.pic_widht = widht
        self.temp = temp_dk  #for dezi Kelvin
        self.size = (height,widht)
        self.bg = self.generate_bg(bg,self.temp)  #np.zeros((32,32)) #
        self.störungen = störungen
        self.persons = self.generate_person(personen=personen)
        self.generate_störung(störungen=self.störungen,anzahl=3)

        # plt.imshow(self.bg)
        # plt.show()
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
        res = data.reshape(self.pic_height,self.pic_widht)
        return res

    def generate_person(self, personen):
        person_list = []
        path = personen
        b = np.load(path, allow_pickle=True)

        for x in range(5):
            i = random.choice([a for a in range(len(b))])

            person = mutate_temp_p(b[i][0], self.temp)

            person_list.append([person,b[i][1]])
        return person_list

    def generate_störung(self, störungen, anzahl):
        # plt.imshow(self.bg)
        # plt.show()
        störung_list =[]
        anzahl = anzahl
        c = störungen
        temp = random.choice([x for x in range(20*10+2732,45*10+2732)])
        for x in range(anzahl):
            if len(c) == 0:
                print("keine störung")
                # return störung_list
            else:
                i = random.choice([a for a in range(len(c))])
                stör = c[i]
                störung_list.append(stör)

                zufall = random.choices(["n", "s", "h", "p"], [0.3, 0.3, 0.4, 0.1])[0]
                if zufall == "s":
                    objekt_list = []
                    objekt = pick_obj(störung_list)
                    objekt_list.append(mutate_obj(mutate_temp_p(objekt[0],temp), objekt[1]))
                    start_list = pick_start(objekt_list=objekt_list, pic_widht=self.pic_widht, pic_height=self.pic_height)
                    # print(f"start_list 's': {start_list}")
                    self.bg = paste_obj2(bg=self.bg, object=start_list[0][0], start_h=start_list[0][3], start_w=start_list[0][2], padding=start_list[0][1],
                                    temp_dk=self.temp)

                if zufall=="h":
                    objekt_list = []
                    objekt = gen_hot_st(self.bg)
                    objekt_list.append(mutate_obj(objekt[0], objekt[1]))
                    start_list = pick_start(objekt_list=objekt_list, pic_widht=self.pic_widht,
                                           pic_height=self.pic_height)
                    # print(f"start_list 'h': {start_list}")
                    self.bg = paste_obj2(bg=self.bg, object=start_list[0][0], start_h=start_list[0][3], start_w=start_list[0][2],
                                         padding=start_list[0][1],
                                         temp_dk=self.temp)

                # if zufall == "p": # and anzahl >1:
                #     # plt.imshow(self.bg)
                #     # plt.show()
                #     # print(self.persons)
                #     ppl = random.choice(self.persons)
                #     objekt_list=[]
                #     objekt = gen_stoer(ppl[0])
                #     # print(f"objekt: {len(objekt)}")
                #     objekt_list.append(objekt)
                #     start_list = pick_start(objekt_list=objekt_list, pic_widht=self.pic_widht, pic_height=self.pic_height)
                #     # print(f"start_list 'gen_person': {start_list}")
                #     self.bg = paste_obj2(bg=self.bg, object=start_list[0][0], start_h=start_list[0][3], start_w=start_list[0][2],
                #                          padding=start_list[0][1],
                #                          temp_dk=random.choices([temp, self.temp], [0.6, 0.4])[0])
                #     # plt.imshow(self.bg)
                #     # plt.show()

                objekt_list = []
                objekt = gen_hot_st(self.bg)
                objekt_list.append(mutate_obj(objekt[0], objekt[1]))
                start_list = pick_start(objekt_list=objekt_list, pic_widht=self.pic_widht,
                                        pic_height=self.pic_height)
                # print(f"start_list 'h': {start_list}")
                self.bg = paste_obj2(bg=self.bg, object=start_list[0][0], start_h=start_list[0][3],
                                     start_w=start_list[0][2],
                                     padding=start_list[0][1],
                                     temp_dk=self.temp)

    def generate_image(self):
        bg = self.bg
        max_obj = 3
        objekt_liste=[]
        label_list = []

        for zähler in range(max_obj):
            if zähler == max_obj-1:
                zufall = 1
            else:
                zufall = random.choices([0, 1, 2], [0.15, 0.8, 0.05])[0]

            if zufall == 1:
                person = pick_obj(self.persons)
                objekt_liste.append(mutate_obj(person[0], person[1]))
            if zufall == 2:
                self.generate_störung(störungen=self.störungen, anzahl=1)

        start_infos = pick_start(objekt_list=objekt_liste,pic_widht=self.pic_widht,pic_height=self.pic_height)

        for item in start_infos:
            label_list.append(
                do_label(person=item[0], pic_height=self.pic_height, pic_widht=self.pic_widht, start_h=item[3],
                         start_w=item[2], padding=item[1], obj_class=0))

            bg = paste_obj2(bg=bg, object=item[0], start_h=item[3], start_w=item[2], padding=item[1], temp_dk=self.temp)

        # print(label_list)

        return bg,label_list


    def generate_imageseries(self):

        person_list = self.persons
        objekt = pick_obj(person_list)
        person,padding = objekt[0],objekt[1]
        schrittweite = 6

        start_p, end_p = pick_series_start(objekt=person,pic_height=self.pic_height,pic_widht=self.pic_widht)

        schritte_x = (self.pic_widht + person.shape[1]) / schrittweite
        schritte_y = (self.pic_height + person.shape[0]) / schrittweite
        entf = ((end_p[0] - start_p[0]) / schritte_x, (end_p[1] - start_p[1]) / schritte_y)

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