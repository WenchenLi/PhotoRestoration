import cv2 as cv
import os

faces_path = '/home/ave/Documents/10kfaces/10k_US_Adult_Faces_Database/Face_Images/'
dest_path  = '/home/ave/Documents/10kfaces/10k_US_Adult_Faces_Database/Face_Images_Grayscale/'

for file in os.listdir(faces_path):
    if file.endswith(".jpg"): 
		img = cv.imread(faces_path + file, 1)
		gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		cv.imwrite(dest_path + file, gray_img)
