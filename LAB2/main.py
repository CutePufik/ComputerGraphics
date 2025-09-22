import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from windows.task1_window import Task1Window



class Lab2:
    def __init__(self, root: tk.Tk):
        self.back_ground = "#333"

        self.window_max = 600

        self.root = root
        self.output_path = self.create_and_get_folder_path("output")
        self.root.title("Lab2")
        self.root.configure(bg=self.back_ground)
        self.root.geometry("400x300+1+1")
        self.path_label = tk.Label(root, text="Image_Path", bg="#333", fg="white")
        self.image_label = tk.Label(root, bg="#333")
        self.path_entry = tk.Entry(root)
        self.load_button = tk.Button(
            root,
            text="Load Image",
            command=self.load_image,
            width=20,
            bg="#555",
            fg="white",
        )

        # Поиск в файлах
        self.browse_button = tk.Button(
            root,
            text="Browse",
            command=self.browse_file,
            bg="#555",
            fg="white",
        )

        # task1
        self.task1_button = tk.Button(
            root,
            text="task1",
            command=self.task1,
            bg="#555",
            fg="white",
        )


        self.path_label.grid(row=0, column=0, columnspan=3, pady=3, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)

        self.path_entry.grid(
            row=1, column=0, columnspan=2, padx=10, pady=3, sticky="nsew"
        )

        self.load_button.grid(row=1, column=2, pady=3)
        self.browse_button.grid(row=2, column=0, pady=3, columnspan=3)
        self.image_label.grid(row=3, column=0, columnspan=3, sticky="nsew")

    def load_image(self):
        path = self.path_entry.get()
        try:
            image = Image.open(path)
            width, height = image.size
            if width > 1200 or height > self.window_max:
                new_height = self.window_max
                height_percent = new_height / float(image.size[1])
                new_width = int((float(image.size[0]) * float(height_percent)))
                image = image.resize((new_width, new_height))

            image = ImageTk.PhotoImage(image)
            self.image_label.config(image=image)
            self.image_label.image = image
            self.root.geometry(f"{image.width()}x{image.height() + 185}")

            self.task1_button.grid(row=5, column=0, pady=2, columnspan=3, sticky="nsew")
            

        except Exception as e:
            print("Error loading image:", e)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if filename:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(tk.END, filename)
            self.load_image()

    def create_and_get_folder_path(self, folder_name):
        root_dir = os.path.abspath(os.getcwd())
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def task1(self):
        child = tk.Tk()
        task1_window = Task1Window(root=child, parent=self)



if __name__ == "__main__":
    root = tk.Tk()
    app = Lab2(root)
    root.mainloop()