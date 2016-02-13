#!/usr/bin/python
import numpy as np
import cv2
import sys


def main(argv):
	if len(sys.argv) < 3:
		print "Not enough params"
		sys.exit()
	img = cv2.imread(sys.argv[1])
	mask = cv2.imread(sys.argv[2],0)
	res = cv2.inpaint(img,mask,3,cv2.INPAINT_NS) 
	cv2.imshow('res',res)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
	sys.exit()

if __name__ == "__main__":
   main(sys.argv[1:])