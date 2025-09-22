import os
import tkinter as tk
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


class Task2Window:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.configure(bg=parent.back_ground)
        self.root.geometry("230x40+500+20")
        self.root.title("task2")

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

        R, G, B = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

        R_image = np.zeros_like(image_array)
        G_image = np.zeros_like(image_array)
        B_image = np.zeros_like(image_array)

        R_image[:, :, 0] = R
        G_image[:, :, 1] = G
        B_image[:, :, 2] = B

        output_dir = os.path.join(self.parent.output_path, "task2")
        os.makedirs(output_dir, exist_ok=True)

        Image.fromarray(R_image).save(os.path.join(output_dir, "R_channel.jpg"))
        Image.fromarray(G_image).save(os.path.join(output_dir, "G_channel.jpg"))
        Image.fromarray(B_image).save(os.path.join(output_dir, "B_channel.jpg"))

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        r_hist = [0] * 256
        for value in R.ravel():
            r_hist[value] += 1
        axs[0].bar(range(256), r_hist, color="red", alpha=0.6)
        axs[0].set_title("hexagram for R-сhanel")

        g_hist = [0] * 256
        for value in G.ravel():
            g_hist[value] += 1
        axs[1].bar(range(256), g_hist, color="green", alpha=0.6)
        axs[1].set_title("hexagram for G-сhanel")

        b_hist = [0] * 256
        for value in B.ravel():
            b_hist[value] += 1
        axs[2].bar(range(256), b_hist, color="blue", alpha=0.6)
        axs[2].set_title("hexagram for B-сhanel")

        plt.show()
        return
