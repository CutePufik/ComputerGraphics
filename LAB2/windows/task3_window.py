import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk


class Task3Window:
    def __init__(self, root: tk.Toplevel, parent):
        self.root = root
        self.parent = parent
        self.output_path = self.parent.create_and_get_folder_path(
            os.path.join(self.parent.output_path, "task3")
        )
        self.image = Image.open(self.parent.path_entry.get())
        self.root.configure(bg=parent.back_ground)

        self.width, self.height = self.image.size
        self.root.geometry(f"{self.width + 100}x{self.height}")
        self.root.title("task3 ")

        self.img_array = np.array(self.image)
        self.img_display = ImageTk.PhotoImage(self.image)
        self.img_label = tk.Label(self.root, image=self.img_display)
        self.img_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(self.root, bg=parent.back_ground)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.hue_slider = tk.Scale(
            controls_frame, from_=-360, to=360, orient=tk.HORIZONTAL, label="Hue"
        )
        self.hue_slider.pack(pady=5)
        self.saturation_slider = tk.Scale(
            controls_frame, from_=-100, to=100, orient=tk.HORIZONTAL, label="Saturation"
        )
        self.saturation_slider.pack(pady=5)
        self.value_slider = tk.Scale(
            controls_frame, from_=-100, to=100, orient=tk.HORIZONTAL, label="Value"
        )
        self.value_slider.pack(pady=5)

        self.process_button = tk.Button(
            controls_frame, text="Process", command=self.process_image
        )
        self.process_button.pack(pady=5)
        self.save_button = tk.Button(
            controls_frame, text="Save", command=self.save_image
        )
        self.save_button.pack(pady=5)

    def process_image(self):
        self.process_button.configure(state=tk.DISABLED)
        h_shift = self.hue_slider.get()
        s_shift = self.saturation_slider.get() / 100.0
        v_shift = self.value_slider.get() / 100.0

        self.image = Image.open(self.parent.path_entry.get())
        self.img_array = np.array(self.image.convert("RGB"))

        hsv_img = np.array(
            [self.rgb_to_hsv(pixel) for pixel in self.img_array.reshape(-1, 3)]
        )
        hsv_img[:, 0] = np.clip(hsv_img[:, 0] + h_shift, 0, 360)
        hsv_img[:, 1] = np.clip(hsv_img[:, 1] + s_shift, 0, 1)
        hsv_img[:, 2] = np.clip(hsv_img[:, 2] + v_shift, 0, 1)

        rgb_img = np.array(
            [self.hsv_to_rgb(pixel) for pixel in hsv_img], dtype=np.uint8
        )
        self.image = Image.fromarray(rgb_img.reshape((self.height, self.width, 3)))

        self.img_display = ImageTk.PhotoImage(self.image)
        self.img_label.config(image=self.img_display)
        self.process_button.configure(state=tk.NORMAL)

    def save_image(self):
        original_filename = os.path.basename(self.parent.path_entry.get())
        filename, ext = os.path.splitext(original_filename)
        save_filename = f"{filename}_processed{ext}"

        save_path = os.path.join(self.output_path, save_filename)

        self.image.convert("RGB").save(save_path)
        print(f"Image saved to {save_path}")

    def rgb_to_hsv(self, rgb):
        r, g, b = rgb / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        h = s = v = (mx + mn) / 2.0
        c = mx - mn

        if c == 0:
            h = 0
        else:
            if mx == r:
                h = (g - b) / c
            elif mx == g:
                h = (b - r) / c + 2
            else:
                h = (r - g) / c + 4
            h *= 60
            if h < 0:
                h += 360

        s = 0 if mx == 0 else (mx - mn) / mx
        v = mx
        return np.array([h, s, v])

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r = (r + m) * 255
        g = (g + m) * 255
        b = (b + m) * 255
        return np.array([r, g, b], dtype=np.uint8)