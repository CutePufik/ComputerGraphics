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

        gray_image_average = self.rgb2gray_average(image_array)
        gray_image_weighted = self.rgb2gray_weighted(image_array)

        Image.fromarray(gray_image_average).save(
            os.path.join(output_folder, "grayscale_average.jpg")
        )
        Image.fromarray(gray_image_weighted).save(
            os.path.join(output_folder, "grayscale_weighted.jpg")
        )

        difference_image = np.abs(gray_image_average - gray_image_weighted)
        Image.fromarray(difference_image).save(
            os.path.join(output_folder, "grayscale_difference.jpg")
        )

        self.plot_histograms(gray_image_average, gray_image_weighted)

    def rgb2gray_average(self, img):
        return (0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]).astype(
            np.uint8
        )

    def rgb2gray_weighted(self, img):
        return (0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]).astype(
            np.uint8
        )

    def plot_histograms(self, gray_image_average, gray_image_weighted):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(gray_image_average.ravel(), bins=256, color="gray", alpha=0.75)
        axs[0].set_title("Grayscale Image Histogram (Average Method)")

        axs[1].hist(gray_image_weighted.ravel(), bins=256, color="gray", alpha=0.75)
        axs[1].set_title("Grayscale Image Histogram (Weighted Method)")

        plt.show()
        return