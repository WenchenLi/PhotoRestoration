import Tkinter as tk
import ImageTk
import Image
import cv2
import os
path ='pyhton/rest-samp-before-2-208x300.jpg'
root = tk.Tk()

def filePicker():
    from Tkinter import Tk
    from tkFileDialog import askopenfilename
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print filename
    return filename
import_button = tk.Button(root,text="select photo",command=(lambda : filePicker())).pack(fill=tk.X, side = tk.TOP, )
pick_button = tk.Button(root,text="pick area").pack(fill=tk.X, side = tk.TOP)
restore_buttion = tk.Button(root,text="restore").pack(fill=tk.X, side = tk.TOP)

embed = tk.Frame(root, width=640, height=400)
embed.pack()
# Tell pygame's SDL window which window ID to use
os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
# The wxPython wiki says you might need the following line on Windows
# (http://wiki.wxpython.org/IntegratingPyGame).
#os.environ['SDL_VIDEODRIVER'] = 'windib'
root.update()

# cropimage = cropImage(path)
# # output_loc = 'out.png'
# cropimage.setup(path)
# left, upper, right, lower = cropimage.mainLoop()
# # ensure output rect always has positive width, height
# if right < left:
#     left, right = right, left
# if lower < upper:
#     lower, upper = upper, lower
# im = Image.open(path)
# im = im.crop(( left, upper, right, lower))
# pygame.display.quit()
# im.save(cropimage)
# img = ImageTk.PhotoImage(Image.open(path))
# panel = tk.Label(root, image = img)
# panel.pack(side = "bottom", fill = "both", expand = "yes")
root.mainloop()