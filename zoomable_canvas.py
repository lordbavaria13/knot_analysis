import tkinter as tk
from PIL import Image, ImageTk
import cv2

class ZoomableCanvas(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.original_image = None
        self.tk_image = None
        self.scale = 1.0
        
        # Panning Variables
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.img_x = 0
        self.img_y = 0

        # Bindings for Mouse
        self.canvas.bind("<MouseWheel>", self.zoom)      
        self.canvas.bind("<Button-4>", self.zoom)         
        self.canvas.bind("<Button-5>", self.zoom)        
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.bind("<Configure>", self.on_resize)

    def set_image(self, cv_img):
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        self.original_image = Image.fromarray(cv_img)
        self.update_idletasks()
        
        # initial Zoom
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            scale_w = cw / self.original_image.width
            scale_h = ch / self.original_image.height
            self.scale = min(scale_w, scale_h) * 0.95
        else:
            self.scale = 1.0
            
        self.img_x = max(cw, 2) // 2
        self.img_y = max(ch, 2) // 2
        self.redraw()

    def zoom(self, event):
        if not self.original_image: return
        if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0:
            self.scale *= 1.2 
        elif getattr(event, 'num', 0) == 5 or getattr(event, 'delta', 0) < 0:
            self.scale *= 0.8
        
        self.scale = max(0.1, min(self.scale, 20.0))
        self.redraw()

    def start_pan(self, event):
        self.canvas.config(cursor="fleur")
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        if not self.original_image: return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.img_x += dx
        self.img_y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.redraw()

    def on_resize(self, event):
        if getattr(self, 'original_image', None):
            self.redraw()

    def redraw(self):
        if not self.original_image: return
        
        # new pic size based on current scale
        new_w = int(self.original_image.width * self.scale)
        new_h = int(self.original_image.height * self.scale)
        if new_w <= 0 or new_h <= 0: return
        
        resized = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.CENTER, image=self.tk_image)
        self.canvas.config(cursor="")