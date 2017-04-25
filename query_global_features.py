#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 01:09:40 2017

@author: yves
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from timeit import default_timer as timer

'''
    Usage :
    ./query_global.py -d "database_name" -q "query_imagename" -r "relevant_images_number"
'''


######## Program parameters
import argparse
parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING")


## Query image name
parser.add_argument("-q", "--query", dest="query_name",
                    help="query image name", metavar="STRING")


## Number of relevant images in the database, considering the query
parser.add_argument("-r", "--relevant", dest="relevant",  type=int,
                    help="relevant image number", metavar="INTEGER")


args = parser.parse_args()


## Set paths [TO UPDATE]
img_dir="/home/yves/tel/Text_Image_Mining/project/Images/" + args.db_name + "/"
#queryNameFile = img_dir + "_liste_" + args.db_name + "_queries.txt"
output_dir="./results/"
if (args.db_name == "COREL"):
    sep='_'
elif (args.db_name == "NISTER"):
    sep='-'


## Load query image
query_filename=img_dir + args.query_name + ".jpg"
queryImage = cv2.imread(query_filename)
queryId = args.query_name.split(sep)[1]

print queryId

cv2.imshow('Image requete',queryImage)
cv2.waitKey(0)
cv2.destroyAllWindows()



      

## Compute query descriptors

hsv = cv2.cvtColor(queryImage,cv2.COLOR_BGR2HSV)
#q_desc is the histogram of color or graylevel
qdesc = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

print "Number of query descriptors :", qdesc.shape[0]




#######################
## Search for similar descriptors in the database
#######################

## Load database descriptors
start = timer()
print "loading db"
dataBaseDescriptors = np.load(output_dir+ args.db_name + "_DB_Descriptors_col_hist.npy")

print 'Number of database descriptors', dataBaseDescriptors.shape[0]
imageBasePaths = np.load(output_dir+ args.db_name +"_imagesPaths.npy")
imageBaseIndex = np.load(output_dir+ args.db_name +"_imagesIndex.npy")
end = timer()
print "load descriptors: " + str(end - start)



## Load database index (computed offline)
start = timer()
FLANN_INDEX_ALGO=1
index_params = dict(algorithm = 254, filename = output_dir+ args.db_name +"_flann_index" + str(FLANN_INDEX_ALGO)+".dat")

d = np.asarray(dataBaseDescriptors,np.float32)
print 'd type', type(d)
print 'd size', d.shape
fl = cv2.flann_Index(np.asarray(dataBaseDescriptors,np.float32), index_params)
end = timer()
print "load index: " + str(end - start)


## Search on the database index
start = timer()
knn = 50
#idx = np.zeros((1,5))
search_params = dict(checks=50)   # or pass empty dictionary
idx,dist = fl.knnSearch(np.asarray(qdesc,np.float32),knn,params={})
end = timer()
print "knn search: " + str(end - start)


print idx.shape
print type(imageBaseIndex)
print imageBaseIndex.shape


#######################
## Compute image scores (voting mechanism)
#######################

scores = np.zeros(len(imageBasePaths))
for index in idx:
    scores[imageBaseIndex[index]] += 1


sortedScore = zip(scores, imageBasePaths)
sortedScore.sort(reverse=True)

# filter out the images with null score
filtered_scores = filter(lambda x: x[0] > 0,  sortedScore   )
print len(filtered_scores), "images retained amongst ", len(sortedScore)

# save results in file
resfile = open(output_dir + args.query_name + "_ranked_list.txt", 'w')
resfile.writelines(["%f %s\n" % item  for item in filtered_scores])
resfile.close()



#######################
## Visualize the top images
#######################
top=10
for i in range(top):
    img = cv2.imread(filtered_scores[i][1])
    plt.subplot(2,5,i+1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('rank '+str(i+1)), plt.xticks([]), plt.yticks([])

plt.savefig(output_dir + args.query_name + "_top" + str(top) +".png")
plt.show()



#######################
## Evaluation : TO COMPLETE
#######################

rpFile = open(output_dir + args.query_name + "_rp.dat", 'w')
total_retrieved = [k[1].split('/')[-1].split(sep)[1]  for k in filtered_scores]
total_nb_relevant = sum([queryId == r_id for r_id in total_retrieved])
nbRelevantImage = args.relevant
precision = np.zeros(nbRelevantImage, dtype=float)
recall = np.zeros(nbRelevantImage, dtype=float)
tp = np.zeros(nbRelevantImage, dtype=float)

for i in range(1,nbRelevantImage):
    nb_retrieved = i
    t = filtered_scores[:nb_retrieved]
    retrieved_id = [k[1].split('/')[-1].split(sep)[1]  for k in t]
    # true positif
    tp[i] = sum([queryId == r_id for r_id in retrieved_id])
    precision[i] = tp[i] /float(nb_retrieved)
    recall[i] = tp[i]/float(total_nb_relevant)
    rpFile.write(str(precision[i]) + '\t' + str(recall[i]) +  '\n')

precision = precision[1:]
recall = recall[1:]   
print 'precision:',precision
print 'recall:',recall

rpFile.close()

# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, lw=2, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall for '+args.query_name)
plt.legend(loc="upper right")
plt.savefig(output_dir + args.query_name + "_rp.png")
#plt.savefig(output_dir + args.query_name + "_rp.pdf", format='pdf')
plt.show()



