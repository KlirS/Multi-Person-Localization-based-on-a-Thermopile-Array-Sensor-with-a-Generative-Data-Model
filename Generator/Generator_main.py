import time

from GeneratetDataset import GeneratetDataSet

start = time.time()
print("start Dataset generation")

DATASETLOCATION = r"new_generatet5"
IMAGESIZE = (32,32)

FRAMES = 16000
SERIES = 1000

data_set_1 = GeneratetDataSet(personen="persons2.npy", störungen="störung3.npy", bg="bg.npy", bildgröße=IMAGESIZE, einzelbilder=FRAMES, serien=SERIES)

data_set_1.save_dataset(DATASETLOCATION, name="new_generatet_set_newstoer")


end = time.time()
delta = end - start
print("end of Dataset generation\n took %.2f seconds to process" % delta)
