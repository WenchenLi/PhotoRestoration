import os
from utils import *
from glob import glob

data = glob(os.path.join("./data", "celebA", "*.jpg"))
crop_directory = os.path.join("./data","celebACropped")
print crop_directory
for file in data:
	image = cv2.imread(file)
	curr_path = os.getcwd()
        #print 'curr_path',curr_path
        cascPath = curr_path.replace("python/DCGAN-tensorflow", "/data/cascade/haarcascade_frontalface_default.xml")
        #print 'cascPath',cascPath
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
	    cropped = image[y:y+h,x:x+w]
	    resized = resize(cropped,64,64)
	    cv2.imwrite(crop_directory + file[file.rindex("/"):],resized)
	    
