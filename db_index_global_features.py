#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:55:29 2017

@author: yves
"""
import cv2
import numpy as np

from timeit import default_timer as timer
import pandas as pd

''' 
    Usage :
    ./db_indexing.py -d "database_name"
'''

######## Program parameters
import argparse
parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING")


args = parser.parse_args()
## Set paths [TO UPDATE]
img_dir="/home/yves/tel/Text_Image_Mining/project/Images/" + args.db_name + "/"
imagesNameFile = img_dir + "_liste_" + args.db_name
output_dir="./results/"

print(img_dir)
print('\n')
print(output_dir)


####################
#### Compute descriptors of the whole database
####################
dataBaseDescriptors = []
imageBaseIndex = []
imageBasePaths = []
im_nb = 0
des_nb=0




imagesNameList = open(imagesNameFile)
for imageName in imagesNameList:
    imagePath = img_dir + imageName[:-1] + ".jpg"
    print "Compute descriptors for : " + imagePath

    image = cv2.imread(imagePath)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])


    for descriptor in hist:
        dataBaseDescriptors.append(descriptor)
        imageBaseIndex.append(im_nb)
        des_nb+=1

    imageBasePaths.append(imagePath)
    im_nb = im_nb + 1

print str(im_nb) + " images in the DB"
print str(des_nb) + " descriptors in the DB"

out_put = pd.DataFrame(dataBaseDescriptors)
out_put.to_csv('/home/yves/tel/Text_Image_Mining/project/t.csv')

np.save(output_dir+ args.db_name + "_DB_Descriptors_col_hist.npy",dataBaseDescriptors)
np.save(output_dir+ args.db_name +"_imagesIndex.npy",imageBaseIndex)
np.save(output_dir+ args.db_name +"_imagesPaths.npy",imageBasePaths)



####################
### Index database descriptors
####################

## Load descriptors (if not in memory)
#dataBaseDescriptors = np.load(output_dir+ args.db_name + "_DB_Descriptors_col_hist.npy")

# Algorithms
# 0 : FLANN_INDEX_LINEAR,
# 1 : FLANN_INDEX_KDTREE,

FLANN_INDEX_ALGO=1
index_params = dict(algorithm = FLANN_INDEX_ALGO,trees = 5)


start = timer()
fl = cv2.flann_Index(np.asarray(dataBaseDescriptors,np.float32),index_params)
end = timer()
print "index descriptors: " + str(end - start)

fl.save(output_dir+ args.db_name +"_flann_index" + str(FLANN_INDEX_ALGO)+".dat")