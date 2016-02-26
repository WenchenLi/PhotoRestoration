import Tkinter as tk
from PIL import Image, ImageTk, ImageOps, ImageDraw
import os
from Tkconstants import LEFT, RIGHT, Y, END, BOTH, VERTICAL

import numpy as np
import cv2

# import Image, ImageTk
from tkFileDialog import askopenfilename
from inpainting import *

import platform

def filePicker():
    tk.Tk().withdraw()
    filename = askopenfilename(initialdir=os.path.join(os.getcwd(), '../data/images/original/'))
    return filename

# class ToolBar():
#     def __init__(self, master):
#         self.master = master
#         self.frame = tk.Frame(self.master)
#         self.canvas = tk.Canvas(master, width=130, height=300)
#         self.canvas.pack()
#         self.master.title("Toolbar")
#         self.master.maxsize(130, 300)
#         self.master.minsize(130, 300)

#         self.brush_selected_image = tk.PhotoImage(file="../assets/paint_selected.png")
#         self.brush_not_selected_image = tk.PhotoImage(file="../assets/paint_not_selected.png")
#         self.rectangle_selected_image = tk.PhotoImage(file="../assets/rectangle_selected.png")
#         self.rectangle_not_selected_image = tk.PhotoImage(file="../assets/rectangle_not_selected.png")

#         self.brush_button = tk.Button(self.canvas, image=self.brush_selected_image, command=(self.select_paintbrush))
#         self.brush_button.pack(side=tk.LEFT, padx=5, pady=5)
#         self.rectangle_button = tk.Button(self.canvas, image=self.rectangle_not_selected_image,
#                                           command=(self.select_rectangle_tool))
#         self.rectangle_button.pack(side=tk.RIGHT, padx=5, pady=5)

#         self.brush_size = 5

#         self.brush_slider = tk.Scale(self.master, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.brush_size,
#                                      command=(lambda: self.change_slider_value))
#         self.brush_slider.set(5)
#         self.brush_slider.pack()

#         self.brush_size_image = tk.PhotoImage(file="../assets/brushsize_4.png")
#         self.brush_size_label = tk.Label(self.master, image=self.brush_size_image)
#         self.brush_size_label.pack()

#         self.rectangle_selected = False
#         self.brush_selected = True

#         self.save_mask_button = tk.Button(self.master, text="Save Mask")
#         self.save_mask_button.pack(side=tk.BOTTOM, fill=tk.X)
#         self.restore_button = tk.Button(self.master, text="Restore")
#         self.restore_button.pack(side=tk.BOTTOM, fill=tk.X)

#     def select_paintbrush(self):
#         if not self.brush_selected:
#             self.brush_button["image"] = self.brush_selected_image
#             self.brush_selected = True

#             self.toggle_on_paintbrush_slider()

#             self.rectangle_button["image"] = self.rectangle_not_selected_image
#             self.rectangle_selected = False

#     def select_rectangle_tool(self):
#         if not self.rectangle_selected:
#             self.rectangle_button["image"] = self.rectangle_selected_image
#             self.rectangle_selected = True

#             self.toggle_off_paintbrush_slider()

#             self.brush_button["image"] = self.brush_not_selected_image
#             self.brush_selected = False

#     def select_paintbrush_size(self):
#         pass

#     def toggle_off_paintbrush_slider(self):
#         self.brush_slider.pack_forget()
#         self.brush_size_label.pack_forget()

#     def toggle_on_paintbrush_slider(self):
#         self.brush_slider.pack()
#         self.brush_size_label.pack()

#     def change_slider_value(self, val):
#         self.brush_size_image = tk.PhotoImage(file="../assets/brushsize_" + str(val) + ".png")
#         self.brush_size_label["image"] = self.brush_size_image

#     def close_window(self):
#         self.master.destroy()


class App():
    def __init__(self, master):
        self.master = master
        self.curr_path= os.getcwd()
        self.platform = platform.system()
        scrollbar = tk.Scrollbar(self.master,orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.frame = tk.Frame(self.master)
        self.frame.pack()
        self.canvas = tk.Canvas(master, width=130, height=300)
        self.canvas.pack()
        
        self.save_button = tk.Button(self.master, text="save masks", command=(self.save_mask)).pack(fill=tk.X,side=tk.TOP)                                                        
        self.restore_button = tk.Button(self.master, text="restore",
                                   command=(lambda: inpaint(self.file, self.mask_path))).pack(fill=tk.X, side=tk.TOP)

        self.brush_selected_image = tk.PhotoImage(file="../assets/paint_selected.png")
        self.brush_selected_image = tk.PhotoImage(file="../assets/paint_selected.png")
        self.brush_not_selected_image = tk.PhotoImage(file="../assets/paint_not_selected.png")
        self.rectangle_selected_image = tk.PhotoImage(file="../assets/rectangle_selected.png")
        self.rectangle_not_selected_image = tk.PhotoImage(file="../assets/rectangle_not_selected.png")

        self.brush_button = tk.Button(self.canvas, image=self.brush_selected_image, command=(self.select_paintbrush))
        self.brush_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.rectangle_button = tk.Button(self.canvas, image=self.rectangle_not_selected_image,
                                          command=(self.select_rectangle_tool))
        self.rectangle_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.brush_size = 5

        self.brush_slider = tk.Scale(self.master, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.brush_size,
                                     command=(self.change_slider_value))
        self.brush_slider.set(5)
        self.brush_slider.pack()

        self.brush_size_image = tk.PhotoImage(file="../assets/brushsize_4.png")
        self.brush_size_label = tk.Label(self.master, image=self.brush_size_image)
        self.brush_size_label.pack()

        self.rectangle_selected = False
        self.brush_selected = True


        self.file = filePicker()
        self.crop_face()
        self.orig_img = Image.open(self.file)
        self.tk_img = ImageTk.PhotoImage(self.orig_img)
        w, h = self.orig_img.size
        self.mask = Image.new('L', self.orig_img.size, "black")
        self.mask_draw = ImageDraw.Draw(self.mask)

        self.mask_path = None
        self.canvas_pic = tk.Canvas(master, width=w, height=h)
        self.canvas_pic.pack()
        self.x = 0
        self.y = 0
        self.select_windows = []
        self.current_window = []

        self.rect = None
        self.start_x = None
        self.start_y = None
        self.canvas_pic.bind("<ButtonPress-1>", self.on_button_press_on_brush_mode)
        # self.canvas.bind("<B1-Motion>", self.on_move_press)
        # self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self._draw_image()
        self.face_coordinate = None
        self.points_pool = []
       
        scrollbar.config(command=self.canvas_pic.yview_moveto(100))

    def photoimage(self,imgFile):
        #mac need to convert image format on imagefilePath
        if self.platform =='Darwin':
            print "Darwin"
            img = Image.open(imgFile).convert('RGB')
            return tk.PhotoImage(img)
        elif self.platform =='Linux':
            return tk.PhotoImage(file=imgFile)
        else:pass
    def select_paintbrush(self):
        if not self.brush_selected:
            self.brush_button["image"] = self.brush_selected_image
            self.brush_selected = True

            self.toggle_on_paintbrush_slider()

            self.rectangle_button["image"] = self.rectangle_not_selected_image
            self.rectangle_selected = False

    def select_rectangle_tool(self):
        if not self.rectangle_selected:
            self.rectangle_button["image"] = self.rectangle_selected_image
            self.rectangle_selected = True

            self.toggle_off_paintbrush_slider()

            self.brush_button["image"] = self.brush_not_selected_image
            self.brush_selected = False

    def select_paintbrush_size(self):
        pass

    def toggle_off_paintbrush_slider(self):
        self.brush_slider.pack_forget()
        self.brush_size_label.pack_forget()

    def toggle_on_paintbrush_slider(self):
        self.brush_slider.pack()
        self.brush_size_label.pack()

    def change_slider_value(self, val):
        self.brush_size_image = tk.PhotoImage(file="../assets/brushsize_" + str(val) + ".png")
        self.brush_size_label["image"] = self.brush_size_image
        print "change brush size to:",self.brush_slider.get()
        self.brush_slider.set(self.brush_slider.get())
        self.brush_size = self.brush_slider.get()

    def close_window(self):
        self.master.destroy()
    def crop_face(self):
        # Get user supplied values
        curr_path = self.curr_path
        imagePath = self.file
        cascPath = curr_path.replace("/python", "/data/cascade/haarcascade_frontalface_default.xml")
        face_path = curr_path.replace("/python", "/data/images/face_cropped/")
        oldfileName = (self.file)[(self.file).rfind("/") + 1:-4]
        face_filename = face_path + oldfileName + "_face.png"
        print "saving to " face_filename
        faceCascade = cv2.CascadeClassifier(cascPath)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print "Found {0} faces!".format(len(faces))
        for (x, y, w, h) in faces:
            self.face_coordinate = (x, y, w, h)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(face_filename, image[y:y + h, x:x + w])
        cv2.waitKey(0)

    def save_mask(self):
        curr_path = os.getcwd()
        new_path = curr_path.replace("/python", "/data/images/mask/")
        oldfileName = (self.file)[(self.file).rfind("/") + 1:-4]
        self.mask_path = new_path + oldfileName + "_mask.png"
        self.mask.save(self.mask_path)
        print "mask saved "

    def _draw_image(self):
        self.canvas_pic.create_image(0, 0, anchor="nw", image=self.tk_img)

    def on_button_press_on_brush_mode(self, event):
        print "drawing on", (event.x, event.y)
        self.canvas_pic.create_rectangle(event.x-self.brush_size, event.y-self.brush_size,
                                                event.x+self.brush_size, event.y+self.brush_size, fill="white")
        self.mask_draw.rectangle([event.x-self.brush_size, event.y-self.brush_size,
                                                event.x+self.brush_size, event.y+self.brush_size], fill="white")
    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        # self.current_window.append((self.start_x,self.start_y))
        self.points_pool.append((event.x, event.y))
        print "appending start", (event.x, event.y)
        # create rectangle if not yet exist
        # if not self.rect:
        self.rect = self.canvas_pic.create_line(self.x, self.y, 1,1, fill="white")

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas_pic.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        # self.current_window.append((event.x,event.y))
        # self.select_windows.append(self.current_window)
        # self.current_window=[]
        self.points_pool.append((event.x, event.y))
        print "appending end", (event.x, event.y)

    def switch_draw_method(self):
        pass


root = tk.Tk()
App(root)
root.mainloop()