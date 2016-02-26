class ToolBar():
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.canvas = tk.Canvas(master, width=130, height=300)
        self.canvas.pack()
        self.master.title("Toolbar")
        self.master.maxsize(130, 300)
        self.master.minsize(130, 300)

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
                                     command=(lambda: self.change_slider_value))
        self.brush_slider.set(5)
        self.brush_slider.pack()

        self.brush_size_image = tk.PhotoImage(file="../assets/brushsize_4.png")
        self.brush_size_label = tk.Label(self.master, image=self.brush_size_image)
        self.brush_size_label.pack()

        self.rectangle_selected = False
        self.brush_selected = True

        self.save_mask_button = tk.Button(self.master, text="Save Mask")
        self.save_mask_button.pack(side=tk.BOTTOM, fill=tk.X)
        self.restore_button = tk.Button(self.master, text="Restore")
        self.restore_button.pack(side=tk.BOTTOM, fill=tk.X)

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

    def close_window(self):
        self.master.destroy()