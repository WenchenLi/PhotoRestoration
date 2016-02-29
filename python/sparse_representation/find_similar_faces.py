import cv2 as cv
from scipy.sparse import csc_matrix, lil_matrix
from scipy.optimize import minimize
import numpy as np
import os

faces_path = '/home/ave/Documents/10kfaces/10k_US_Adult_Faces_Database/Face_Images_Grayscale/'
query_img_path = "/home/ave/Downloads/avery.jpg"
query_img = cv.imread(query_img_path,0)
query_img = query_img[query_img.shape[0]/2-128:query_img.shape[0]/2+128,
	query_img.shape[1]/2-75:query_img.shape[1]/2+75]

img_paths = [path for path in os.listdir(faces_path) if path.endswith(".jpg")]
num_faces = len(img_paths)
#38400 is len of cropped image vector, original image is 256 x 150
vectorized_images = np.zeros((38400,num_faces))

#cannot get sparse representation to work with minimize for some reason
#coeffecient_vec = csc_matrix((num_faces),dtype = np.float64)
coeffecient_vec = np.zeros(num_faces)
min_func = lambda x: abs(x).sum(axis=0).sum()
print min_func(coeffecient_vec)
for i in range(len(img_paths)):
	face_img = cv.imread(faces_path + img_paths[i], 0)
	#Smallest pic only has 150 width, so lets try just grabbing that much to start
	center = face_img.shape[1]/2
	cropped_face = face_img[:, center-75:center+75].flatten()

	try:
		vectorized_images[:,i] = cropped_face
	
	except:
		print "Image Sizer Error"
		print "Image path:", img_paths[i]
		print "Image num:", i
		print "Uncropped Shape:", face_img.shape
		print "Cropped Shape:", cropped_face.shape

min_func = lambda x: abs(x).sum(axis=0).sum()
cons = ({'type': 'eq', 'fun': lambda x: vectorized_images * coeffecient_vec - query_img})
res = minimize(min_func, coeffecient_vec, method='SLSQP',options={'disp': True})