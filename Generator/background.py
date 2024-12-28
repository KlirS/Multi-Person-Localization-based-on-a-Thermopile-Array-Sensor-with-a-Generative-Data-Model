import random
import os
import numpy as np
import scipy

# Needed for plotting
import matplotlib.colors
import matplotlib.pyplot as plt

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def read_img_txt(adress):
    with open(adress) as data:
        line = data.readline()
        temp = line.split(",")[:1024]
        # print(temp)
        nd_temp = np.asarray(temp, float)
        # print(f" mean:  {np.mean(nd_temp)-2732}\n std: {np.std(nd_temp)}")
        nd_temp2 = nd_temp.reshape((32, 32))
    return nd_temp

#define the seed so that results can be reproduced
# seed = 11
# rand_state = 11

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

def generate_samples(data,anzahl):

    # Fit a kernel density model using GridSearchCV to determine the best parameter for bandwidth
    bandwidth_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
    grid_search = GridSearchCV(KernelDensity(), bandwidth_params)
    grid_search.fit(data)
    kde = grid_search.best_estimator_

    # Generate/sample "anzahl" new objects from this dataset
    new_bgs = kde.sample(anzahl) #random_state=rand_state

    return new_bgs

def read_and_save_bg(pfad,save=False):

    list_dir = os.listdir(pfad)
    list_p = []
    for i in list_dir:
        adresse = pfad + "/" + i
        list_p.append(read_img_txt(adresse))

    x = np.asarray(list_p)
    list_obj=[]
    for obj in x:
        obj = obj.reshape(32,32)
        list_obj.append(obj.reshape(-1))
        list_obj.append(np.flipud(obj).reshape(-1))
        list_obj.append(np.fliplr(obj).reshape(-1))
        list_obj.append(np.rot90(obj, 1).reshape(-1))
        list_obj.append(np.rot90(obj, 2).reshape(-1))
        list_obj.append(np.rot90(obj, 3).reshape(-1))
        list_obj.append(np.rot90(obj, 4).reshape(-1))

    random.shuffle(list_obj)
    arr = np.asarray((list_obj))

    if save == True:
        np.save('bg.npy', arr)

    return arr

#
# PATH=r""
#
# x = read_and_save_bg(PATH,True)
#
# bgs = generate_samples(x,8)
#
# print(type(bgs))

#
# def do_bg(size:(int,int), temp,sigma,größegauss):
#     data = np.random.normal(loc=temp, scale=14, size=size[0] * size[1])
#     print(f"first mean:  {np.mean(data)} std: {np.std(data)}")
#     data2= np.random.normal(loc=temp*10+2732, scale=0.9, size=size[0] * size[1])
#     generator = np.random.default_rng()
#     print(f"low {int(np.min(data2))-2732}  and max: {int(np.max(data2))-2732}")
#     data2 = generator.uniform(low=int(np.min(data2)), high=int(np.max(data2)), size=size[0] * size[1])
#     # print(f"data2   {data2}")
#     for i in range(len(data)):
#         data[i]=((data[i]*10+2732)*0.5+(np.round(data2)[i])*1.5)/2
#     print(f"second mean:  {np.mean(data)-2732} std: {np.std(data)}")
#     data = data.reshape(size)
#
#     s = sigma
#     w = größegauss
#     t = (((w - 1) / 2) - 0.5) / s
#     for i in range(2):
#         data = scipy.ndimage.filters.gaussian_filter(data, sigma=s, truncate=t)
#
#
#     # blurred_data = ndimage.gaussian_filter(data, sigma=2)
#     # blurred_data = ndimage.median_filter(data, 1)
#     # cv2.imshow("win",blurred_data.astype(np.uint8))
#     # cv2.waitKey()
#
#     # Generate Distribution:
#     randomNums = np.random.normal(loc=2932, scale=0.9, size=[32,32])
#     data = np.round(data)
#
#     print(f"third mean:  {np.mean(data) - 2732} std: {np.std(data)}")
#
#     return data
