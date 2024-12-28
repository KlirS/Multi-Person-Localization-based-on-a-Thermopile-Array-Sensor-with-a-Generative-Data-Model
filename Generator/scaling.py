import sklearn.preprocessing as pre
import numpy as np

def scaler(objekt):
    scaler = pre.StandardScaler()
    nd_temp_scale = scaler.fit_transform(objekt)
    return np.interp(nd_temp_scale, (nd_temp_scale.min(), nd_temp_scale.max()), (0, 255))


# def scaler(data):   # _between_min_max
#     s_min = 5 * 10 + 2732
#     s_max = 50 * 10 + 2732
#
#     start = 0
#     end = 255
#     width = end - start
#
#     dat = data.flatten()
#     dat = np.append(dat, [s_min, s_max])
#     dat = np.clip(dat, s_min, s_max)
#
#     scaled = (dat - dat.min()) / dat.ptp() * width + start
#     scaled = scaled.astype(np.uint8)
#     scaled = scaled[:-2]
#
#     return scaled.reshape((32,32))