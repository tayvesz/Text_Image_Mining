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
    ./db_indexing.py -d "database_name" -f "features_types" -a "algoindex"
'''

######## Program parameters
import argparse
parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING")

parser.add_argument("-f", "--features", dest="features",
                    help="type of features to extract", metavar="STRING")

## index algorithm type
parser.add_argument("-a", "--algoindex", dest="algo_index",  type=int,
                    help="index algo type", metavar="INTEGER")


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


if (args.features == "surf"):
    feat_extractor = cv2.SURF(400)
if (args.features == "sift"):
    feat_extractor = cv2.SIFT()
      

imagesNameList = open(imagesNameFile)
for imageName in imagesNameList:
    imagePath = img_dir + imageName[:-1] + ".jpg"
    print "Compute descriptors for : " + imagePath

    image = cv2.imread(imagePath)
    kp, des = feat_extractor.detectAndCompute(image,None)

    if(des != None):
        for descriptor in des:
            dataBaseDescriptors.append(descriptor)
            imageBaseIndex.append(im_nb)
            des_nb+=1

    imageBasePaths.append(imagePath)
    im_nb = im_nb + 1

print str(im_nb) + " images in the DB"
print str(des_nb) + " descriptors in the DB"

out_put = pd.DataFrame(dataBaseDescriptors)
out_put.to_csv('/home/yves/tel/Text_Image_Mining/project/t.csv')

np.save(output_dir+ args.db_name + "_DB_Descriptors.npy",dataBaseDescriptors)
np.save(output_dir+ args.db_name +"_imagesIndex.npy",imageBaseIndex)
np.save(output_dir+ args.db_name +"_imagesPaths.npy",imageBasePaths)



####################
### Index database descriptors
####################

## Load descriptors (if not in memory)
dataBaseDescriptors = np.load(output_dir+ args.db_name + "_DB_Descriptors.npy")

# Algorithms
# 0 : FLANN_INDEX_LINEAR,
# 1 : FLANN_INDEX_KDTREE,

if (args.algo_index == 1):
    FLANN_INDEX_ALGO=1
    index_params = dict(algorithm = FLANN_INDEX_ALGO, trees = 5)

    index_params = dict(algorithm = FLANN_INDEX_ALGO)

if (args.algo_index == 0):
    FLANN_INDEX_ALGO=0
    index_params = dict(algorithm = FLANN_INDEX_ALGO)

start = timer()
fl = cv2.flann_Index(np.asarray(dataBaseDescriptors,np.float32),index_params)
end = timer()
print "index descriptors: " + str(end - start)

fl.save(output_dir+ args.db_name +"_flann_index" + str(FLANN_INDEX_ALGO)+".dat")