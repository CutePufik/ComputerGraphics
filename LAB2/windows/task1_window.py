import os
import tkinter as tk
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


class Task1Window:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.configure(bg=parent.back_ground)
        self.root.geometry("230x40+500+20")
        self.root.title("task1")

        self.start_button = tk.Button(
            root,
            text="start",
            command=self.start,
            width=20,
            bg="#555",
            fg="white",
        )
        self.root.grid_columnconfigure(0, weight=1)

        self.start_button.grid(
            row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

    def start(self):
        image = Image.open(self.parent.path_entry.get())
        image_array = np.array(image)

        output_folder = os.path.join(self.parent.output_path, "task1")
        os.makedirs(output_folder, exist_ok=True)

        gray_image_ntsc = self.rgb2gray_ntsc(image_array)
        gray_image_hdtv = self.rgb2gray_hdtv(image_array)

        Image.fromarray(gray_image_ntsc).save(
            os.path.join(output_folder, "grayscale_ntsc.jpg")
        )
        Image.fromarray(gray_image_hdtv).save(
            os.path.join(output_folder, "grayscale_hdtv.jpg")
        )

        difference_image = np.abs(gray_image_ntsc.astype(np.int16) - gray_image_hdtv.astype(np.int16)).astype(np.uint8)
        Image.fromarray(difference_image).save(
            os.path.join(output_folder, "grayscale_difference.jpg")
        )

        self.plot_histograms(gray_image_ntsc, gray_image_hdtv)

    def rgb2gray_ntsc(self, img):
        return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)

    def rgb2gray_hdtv(self, img):
        return (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(np.uint8)

    def plot_histograms(self, gray_image_ntsc, gray_image_hdtv):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(gray_image_ntsc.ravel(), bins=256, color="gray", alpha=0.75)
        axs[0].set_title("Grayscale Histogram (NTSC/PAL Standard)")

        axs[1].hist(gray_image_hdtv.ravel(), bins=256, color="gray", alpha=0.75)
        axs[1].set_title("Grayscale Histogram (HDTV Standard)")

        plt.tight_layout()
        plt.show()