import cv2 as cv
import os

faces_path = '/home/ave/Documents/10kfaces/10k_US_Adult_Faces_Database/Face_Images/'
max_width = 0
min_width = 1000

for file in os.listdir(faces_path):
    if file.endswith(".jpg"): 
        img = cv.imread(faces_path + file)
        if img.shape[1] < min_width:
        	min_width = img.shape[1]
        if img.shape[1] > max_width:
        	max_width = img.shape[1]

print "Max Width:", max_width
print "Min Width:", min_width


#   Results
#Max Width: 394
#Min Width: 151

#Note: All heights are 256!