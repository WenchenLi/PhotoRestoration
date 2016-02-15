import Tkinter as tk
import os
import numpy as np
import cv2
import Tkinter
import Image, ImageTk
from Tkinter import Tk
from tkFileDialog import askopenfilename
from inpainting import *

def filePicker():
    Tk().withdraw()
    filename = askopenfilename()
    return filename
def SelectAndShowImage(root):
    img = cv2.imread(filePicker())
    b,g,r = cv2.split(img)#Rearrang the color channel
    img = cv2.merge((r,g,b))
    im = Image.fromarray(img)# Convert the Image object into a TkPhoto object
    imgtk = ImageTk.PhotoImage(image=im)
    Tkinter.Label(root, image=imgtk).pack()#http://stackoverflow.com/questions/21979172/tkinter-display-other-windows
    root.mainloop()

global p_image,p_mask
root = tk.Tk()

import_button = tk.Button(root, text="select photo", command=(lambda : SelectAndShowImage(root))).pack(fill=tk.X, side = tk.TOP, )
pick_button = tk.Button(root,text="pick area").pack(fill=tk.X, side = tk.TOP)
restore_button = tk.Button(root,text="restore",command=(lambda : inpaint(p_image,p_mask))).pack(fill=tk.X, side = tk.TOP)

root.mainloop()