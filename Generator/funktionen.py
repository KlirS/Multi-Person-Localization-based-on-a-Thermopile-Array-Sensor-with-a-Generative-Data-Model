import random
import numpy as np
import json
import statistics
import scipy

#bilder
import cv2
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.image as mpimg
from cmcrameri import cm
import matplotlib.pyplot as plt

#eigene
from scaling import scaler


def pick_obj(objekte):
    zufall = random.randint(0, len(objekte) - 1)
    objekt = objekte.pop(zufall)
    # print(objekt)
    return objekt


def paste_obj(bg, person, start_h, start_w):

    height,widht = person.shape
    for i in range(height):
        for j in range(widht):
            try:
                if (start_h + i < 0) or (start_w + j < 0):
                    pass
                elif (i == 0) or (i == height - 1) or (j == 0) or (j == widht - 1):
                    bg[start_h + i][start_w + j] = (bg[start_h + i][start_w + j] * 1.8 + person[i][
                        j] * 0.2) / 2
                elif (i == 1) or (i == height - 2) or (j == 1) or (j == widht - 2):
                    bg[start_h + i][start_w + j] = (bg[start_h + i][start_w + j] * 1.4 + person[i][
                        j] * 0.6) / 2
                elif (i == 2) or (i == height - 3) or (j == 2) or (j == widht - 3):
                    bg[start_h + i][start_w + j] = (bg[start_h + i][start_w + j] * 1 + person[i][
                        j] * 1) / 2
                elif (i == 3) or (i == height - 4) or (j == 3) or (j == widht - 4):
                    bg[start_h + i][start_w + j] = (bg[start_h + i][start_w + j] * 0.5 + person[i][
                        j] * 1.5) / 2
                else:
                    bg[start_h + i][start_w + j] = (bg[start_h + i][start_w + j] * 0.1 + person[i][
                        j] * 1.9) / 2
            except:
                pass
    return bg

def do_label(person,pic_height, pic_widht, start_h, start_w, padding, obj_class):
    # 0 0.14182692307692307 0.6923076923076923 0.10336538461538461 0.375
    # class x y (center of rectangle) widht heigt

    start_point = [start_w + padding[3], start_h + padding[0]]
    end_point = [
        start_w + person.shape[1] - padding[1], start_h + person.shape[0] - padding[2]]

    if start_point[0]<0:
        start_point[0]=0
    if start_point[1]<0:
        start_point[1]=0
    if start_point[0] > pic_widht:
        start_point[0] = pic_widht
    if start_point[1] > pic_height:
       start_point[1] = pic_height

    if end_point[0] > pic_widht:
        end_point[0] = pic_widht
    if end_point[1]>pic_height:
        end_point[1]=pic_height
    if end_point[0]<0:
        end_point[0]=0
    if end_point[1]<0:
        end_point[1]=0


    xCenter = (start_point[0] + end_point[0]) / 2 / pic_widht
    yCenter = (start_point[1] + end_point[1]) / 2 / pic_height

    label_height = (end_point[1]-start_point[1]) / pic_height
    label_widht = (end_point[0]-start_point[0]) / pic_widht

    for i in [obj_class, xCenter,yCenter,label_widht,label_height]:
        if i < 0:
            print(f"start: {start_point}")
            print(f"end: {end_point}")
            print([obj_class, xCenter,yCenter,label_widht,label_height])

    return [obj_class, xCenter,yCenter,label_widht,label_height]


def pick_start(objekt_list, pic_widht, pic_height):
    # print(f"objekt_list aus pick_start: {objekt_list}, len: {len(objekt_list)}") #{objekt_list}
    res_list = []
    for i in range(len(objekt_list)):
        # print(f"len res_list: {len(res_list)}")
        # print(f"objekt_list[i]: {objekt_list[i]}")
        objekt = objekt_list[i][0]
        padding = objekt_list[i][1]
        # print(objekt.shape)
        height_o, widht_o = objekt.shape
        if height_o>=pic_height:
            height_o=pic_height-1
        if widht_o>=pic_widht:
            widht_o = pic_widht-1

        if i == 0:
            # print(f"i = 0")
            start_w = random.choice([x for x in range(pic_widht -int(widht_o/2))])
            start_h = random.choice([y for y in range(pic_height - int(widht_o/2))])
            res_list.append([objekt, padding, start_w, start_h])
        else:
            index = len(res_list)-1
            # print("else")
            free_x = [x for x in range(pic_widht) if (x < res_list[index][2] - widht_o) or (x > res_list[index][2] + res_list[index][0].shape[1])]
            free_y = [y for y in range(pic_height) if (y < res_list[index][3] - height_o) or (y > res_list[index][3] + res_list[index][0].shape[0])]
            # print(f"free x,y shape:  {free_x}, {free_y} , {(height_o,widht_o)}")
            choice1 = [x for x in free_x if x<pic_widht-widht_o/2]
            choice2 = [y for y in free_y if y<pic_height-height_o/2]
            # print(f"choices: {choice1} and {choice2}")

            if choice1!=[] and choice2!=[]:
                start_w = random.choice(choice1)
                start_h = random.choice(choice2)
                res_list.append([objekt, padding, start_w, start_h])

    # print(f"res_list aus pick start: {res_list}")
    return res_list


def pick_series_start(objekt, pic_height, pic_widht):
    height_o, widht_o = objekt.shape

    startside = random.choice(["o", "r", "u", "l"])

    if startside == "o" :
        start_x = random.choice([x+int(widht_o/2) for x in range(pic_widht-widht_o)])
        start_y = int(-height_o/2)
        end_x = random.choice([x+int(widht_o/2) for x in range(pic_widht-widht_o)])
        end_y = pic_height - int(height_o/2)
    elif startside == "r" :
        start_x = pic_widht - int(widht_o/2)
        start_y = random.choice([y+int(height_o/2) for y in range(pic_height-height_o)])
        end_x = int(widht_o/2)
        end_y = random.choice([y+int(height_o/2) for y in range(pic_height-height_o)])
    elif startside == "u":
        start_x = random.choice([x+int(widht_o/2) for x in range(pic_widht-widht_o)])
        start_y = pic_height - int(height_o/2)
        end_x = random.choice([x+int(widht_o/2) for x in range(pic_widht-widht_o)])
        end_y = int(-height_o/2)
    else:
        start_x = int(widht_o/2)
        start_y = random.choice([y+int(height_o/2) for y in range(pic_height-height_o)])
        end_x = pic_widht - int(widht_o/2)
        end_y = random.choice([y+int(height_o/2) for y in range(pic_height-height_o)])


    start_p = (start_x, start_y)
    end_p = (end_x, end_y)

    return start_p, end_p

def save_in_folder(data, img_dir, label_dir, name, cmap=cm.grayC_r, color=True):
    if color:
        for i in range(len(data)):
            # np.savetxt(speicherort + "\images/"+name+str(i)+".txt", self.data_list[i][0])
            mpimg.imsave(img_dir + name + str(i) + ".jpg", scaler(data[i][0]), cmap=cmap)
            # cv2.imwrite(img_dir + name + str(i) + ".jpg", scaler(data[i][0]))
            # if self.data_list[i][0].all() == np.loadtxt(speicherort + "\images/"+name+str(i)+".txt").all():
            #     print("supi")
            # break
            with open(label_dir + name + str(i) + ".txt", 'w') as f:
                for line in data[i][1]:
                    if len(data[i][1]) == 1:
                        s = " ".join(map(str, line))
                        f.write(s)
                    else:
                        s = " ".join(map(str, line))
                        f.write(s + '\n')
    else:
        for i in range(len(data)):
            img = scaler(data[i][0])
            cv2.imwrite(img_dir + name + str(i) + ".jpg", img)
            with open(label_dir + name + str(i) + ".txt", 'w') as f:
                for line in data[i][1]:
                    if len(data[i][1]) == 1:
                        s = " ".join(map(str, line))
                        f.write(s)
                    else:
                        s = " ".join(map(str, line))
                        f.write(s + '\n')
    print(f"{len(data)} Bilder in {name} gespeichert")


def save_COCO(data, img_dir, name, cmap=cm.grayC_r):
    categories = []
    categories.append({'id': 0, 'name': "people", 'supercategory': "people"})

    write_json_context = dict()
    write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '',
                                  'date_created': '2021-02-12 11:00:08.5'}
    write_json_context['licenses'] = [{'id': 0, 'name': None, 'url': None}] # image id =  0 wegen lightning
    write_json_context['categories'] = categories
    write_json_context['images'] = []
    write_json_context['annotations'] = []

    for i in range(len(data)):
        img_name = name + str(i)  # name of the file without the extension
        # mpimg.imsave(img_dir + img_name + ".jpg", scaler(data[i][0]), cmap=cmap)
        height, width = data[i][0].shape

        img_context = {}
        img_context['file_name'] = img_name + ".jpg"
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = '2022'
        img_context['id'] = i  # image id =  0 wegen lightning
        img_context['license'] = 1
        img_context['coco_url'] = ''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)

        bbox_dict = {}
        for label in data[i][1]:
            class_id = int(label[0])
            x_yolo = float(label[1])
            y_yolo = float(label[2])
            width_yolo = float(label[3])
            height_yolo = float(label[4])

            bbox_dict['id'] = i             # image id =  0 wegen lightning
            bbox_dict['image_id'] = i       # image id =  0 wegen lightning
            bbox_dict['category_id'] = class_id     # image id =  0 wegen lightning
            bbox_dict['iscrowd'] = 0  # There is an explanation before
            h, w = abs(height_yolo * height), abs(width_yolo * width)
            bbox_dict['area'] = h * w
            x_coco = round(x_yolo * width - (w / 2))
            y_coco = round(y_yolo * height - (h / 2))
            if x_coco < 0:  # check if x_coco extends out of the image boundaries
                x_coco = 0
            if y_coco < 0:  # check if y_coco extends out of the image boundaries
                y_coco = 0
            bbox_dict['bbox'] = [x_coco, y_coco, w, h]
            bbox_dict['segmentation'] = []
            write_json_context['annotations'].append(bbox_dict)

    # Finally done, save!
    coco_format_save_path =img_dir + name + '.json'
    with open(coco_format_save_path, 'w') as fw:
        json.dump(write_json_context, fw)
    print("COCO is saved")

def paste_obj2(object, bg, start_h, start_w, padding, temp_dk):

    img_bg = Image.fromarray(bg)
    img_ob = Image.fromarray(object)

    mask_im = Image.new("L", img_ob.size, 0)
    draw = ImageDraw.Draw(mask_im)

    start_point = (0 + padding[3], 0 + padding[0])
    end_point = (0 + object.shape[1] - padding[1], 0 + object.shape[0] - padding[2])
    draw.rectangle((start_point, end_point), fill=temp_dk)
    # mask_im.save('mask_circle.tiff', quality=95)

    back_im = img_bg.copy()
    back_im.paste(img_ob, (0, 0), mask_im)
    # back_im.save('rocket_pillow_paste_mask_circle.tiff', quality=95)

    mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(3))
    # mask_im_blur.save('mask_circle_blur.tiff', quality=95)

    back_im = img_bg.copy()
    back_im.paste(img_ob, (start_w, start_h), mask_im_blur)
    # back_im.save('rocket_pillow_paste_mask_circle_blur.tiff', quality=95)

    return np.asarray(back_im)


def mutate_obj(obj,padding=np.asarray([1,1,1,1])):
    choice = random.choices([0,1,2,3,4,5],[0.2,0.2,0.2,0.2,0.2,0.2])[0]
    if choice == 0:
        res = np.flipud(obj)
        res_p = [padding[2],padding[1],padding[0],padding[3]]
    elif choice == 1:
        res = np.fliplr(obj)
        res_p = [padding[0],padding[3],padding[2],padding[1]]
    elif choice == 2:
        res = np.rot90(obj, 1)
        res_p = np.roll(padding, 1)
    elif choice == 3:
        res = np.rot90(obj, 2)
        res_p = np.roll(padding, 2)
    elif choice == 4:
        res = np.rot90(obj, 3)
        res_p = np.roll(padding, 3)
    else:
        res = obj
        res_p = padding

    return res ,res_p


def makeGaussian2(x_center=0, y_center=0, theta=0, sigma_x = 10, sigma_y=10, x_size=32, y_size=32):
    # x_center und y_center mitte vom Ausschlag, theta drehung
    # sigma_x und sigma_y größe Kreis
    # x_size und y_size Bildgröße

    theta = 2*np.pi*theta/360
    x = np.arange(0,x_size, 1, float)
    y = np.arange(0,y_size, 1, float)

    # print(f"x: {x}   y: {y}")

    y = y[:,np.newaxis]
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    # # rotation
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*x0 -np.sin(theta)*y0
    b0=np.sin(theta)*x0 +np.cos(theta)*y0

    return np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))

def gen_hot_st(bg):
    flat = bg.flatten().tolist()
    meant = statistics.mean(flat)
    stdt = statistics.stdev(flat)
    zufall = random.choice([3,4,5,6])
    stoe = np.zeros((zufall,zufall))
    temp = random.choice([x for x in range(20*10+2732,80*10+2732)])
    temp_g = np.random.normal(loc=temp, scale=10, size=100)
    gauss = np.random.normal(loc=meant, scale=stdt, size=100)

    for i in range(zufall):
        for j in range(zufall):
            if i>=1 and j>=1 and i <zufall-1 and j <zufall-1:
                stoe[i][j] = random.choice(temp_g)
            else:
                stoe[i][j] = random.choice(gauss)
    pad = np.asarray([1,1,1,1])
    # plt.imshow(stoe)
    # plt.show()
    return [stoe, pad]

def gen_stoer(ppl):
    size = (random.choice([x+3 for x in range(13)]), random.choice([x+3 for x in range(13)]))
    pad = random.choices([0,1,2],k=4)
    maxt = max(ppl.flatten())
    meant = statistics.mean(ppl.flatten())
    stdt = statistics.stdev(ppl.flatten())
    gauss = np.random.normal(loc=meant+(maxt-meant)/2, scale=stdt, size=size)

    x_center = random.choice([x for x in range(int(size[1] * 0.1), int(size[1] * 0.9))])
    y_center = random.choice([y for y in range(int(size[0] * 0.1), int(size[0] * 0.9))])
    sigma_x = random.choice([x for x in range(1, int(size[1] * 0.9))])
    sigma_y = random.choice([y for y in range(1, int(size[0] * 0.9))])
    theta = random.choice([t for t in range(1, 181)])

    gauss2 = makeGaussian2(x_center=x_center, y_center=y_center, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, x_size=size[1], y_size=size[0])

    temp = random.choice([x for x in range(25*10+2732,35*10+2732)])
    gauss3 = np.random.normal(loc=temp, scale=random.choice([x for x in range(20,100)]), size=size[0] * size[1])
    gauss3_max = max(gauss3)
    gauss3_min = min(gauss3)

    start = gauss3_min
    end = gauss3_max
    width = end - start

    res = (gauss2 - gauss2.min()) / gauss2.ptp() * width + start

    height, widht = gauss.shape
    stoer = np.zeros(size)

    for i in range(height):
        for j in range(widht):
            stoer[i][j] = (gauss[i][j] * 0.05 + res[i][j] * 1.95) / 2

    s = 0.5
    w = 3
    t = (((w - 1) / 2) - 0.5) / s
    filtered_data = scipy.ndimage.filters.gaussian_filter(stoer, sigma=s, truncate=t)
    # print(f"filtered_data type: {type(filtered_data)}")
    filtered_data = np.asarray(filtered_data)
    # print(filtered_data.shape)

    if random.choice([True,False]):
        filtered_data = mutate_temp_p(filtered_data,temp).reshape(size)
    # plt.imshow(filtered_data)
    # plt.show()
    # print(pad)
    return [filtered_data, pad]

def mutate_temp_p(ppl,temp):
    shape = ppl.shape
    newp = ppl.flatten()
    medi_l = statistics.median_low(newp)
    for idx,item in enumerate(newp):
        if item < medi_l:
            newp[idx]= item+temp-medi_l

    return np.reshape(newp,shape)

########################### Tiff speichern und laden
#
# from numpy import *
#
# data = np.load("/content/bg.npy")[0].reshape((32,32))
# img1 = Image.fromarray(data)
# img1.save('test.tiff')
# img2 = Image.open('test.tiff')
#
# f1 = list(img1.getdata())
# f2 = list(img2.getdata())
# print(f1 == f2)
# print(f1)

#
# list_p=[]
#
# a = np.asarray([[1,0],[0,0],[0,0]])
# padding = [1,2,3,4]
#
# for i in range(6):
#     img,pad = mutate_obj(a,padding)
#     print(f"padding: {padding} und pat {pad}")
#     list_p.append(img)
#
# länge = len(list_p)
# fig, axs = plt.subplots(2,länge)
# plt.subplots_adjust(
# top=0.976,
# bottom=0.062,
# left=0.037,
# right=0.972,
# hspace=0.0,
# wspace=0.462)
#
# # print(list_p)
#
# itemcount=0
# for j in range(länge):
#     img = list_p[itemcount]
#     axs[0,j].imshow(img)
#     axs[1, j].hist(img)
#     axs[1, j].set_xlim(xmax = 3000, xmin = 2800)
#     axs[1, j].set_ylim(ymax=60, ymin=0)
#     itemcount+=1
#
# plt.show()