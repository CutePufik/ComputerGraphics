import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math


class Task1Window:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.back_ground = parent.back_ground
        self.root.configure(bg=parent.back_ground)
        self.root.geometry("230x120+500+20")
        self.root.title("task1")

        # task1a
        self.task1a_button = tk.Button(
            root,
            text="task1a",
            command=self.task1a,
            bg="#555",
            fg="white",
            width=100,
        )
        self.task1a_button.pack(pady=5, padx=5)

        # task1b
        self.task1b_button = tk.Button(
            root,
            text="task1b",
            command=self.task1b,
            bg="#555",
            fg="white",
            width=100,
        )
        self.task1b_button.pack(pady=5, padx=5)

        # task1c
        self.task1c_button = tk.Button(
            root,
            text="task1c",
            command=self.task1c,
            bg="#555",
            fg="white",
            width=100,
        )
        self.task1c_button.pack(pady=5, padx=5)

    def task1a(self):
        child = tk.Tk()
        task1a_window = Task1aWindow(root=child, parent=self)

    def task1b(self):
        child = tk.Tk()
        task1b_window = Task1bWindow(root=child, parent=self)

    def task1c(self):
        child = tk.Tk()
        task1c_window = Task1cWindow(root=child, parent=self)


class Task1aWindow:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.configure(bg=parent.back_ground)
        self.width = 800
        self.height = 600
        self.root.title("task1a")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)

        # границы,
        self.boarders = set()

        # уже закрашены
        self.passed_val = set()

        self.fill_button = tk.Button(
            root,
            text="fill",
            command=self.fill,
            bg="#555",
            fg="white",
            width=100,
        )
        self.fill_button.pack(side=tk.TOP, pady=10)

        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None

    def fill(self):
        self.canvas.unbind("<B1-Motion>")
        self.canvas.bind("<Button-1>", self.recursive_line_fill)

    def check_validity(self, x, y):
        return(
                (x, y) not in self.boarders
                and (x, y) not in self.passed_val
                and 0 < x < self.width - 2
                and 0 < y < self.height - 2
        )


    def recursive_line_fill(self, event):
        self.canvas.unbind("<Button-1>")
        x, y = event.x, event.y

        def fill_scanline(x, y):
            if not self.check_validity(x, y):
                return

            x_left = x
            while self.check_validity(x_left - 1, y):
                x_left -= 1

            x_right = x
            while self.check_validity(x_right + 1, y):
                x_right += 1

            self.canvas.create_line(x_left, y, x_right, y, fill="red")

            for nx in range(x_left, x_right + 1):
                self.passed_val.add((nx, y))

            self.root.update()  # Обновление для избежания зависаний

            for nx in range(x_left, x_right + 1):
                fill_scanline(nx, y + 1)
                fill_scanline(nx, y - 1)

        fill_scanline(x, y)
        print("Done")
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        self.last_x, self.last_y = event.x, event.y
        oval_size = 10

        self.canvas.create_oval(
            self.last_x - oval_size // 2,
            self.last_y - oval_size // 2,
            self.last_x + oval_size // 2,
            self.last_y + oval_size // 2,
            fill="black",
            outline="black",
        )

        for x in range(self.last_x - oval_size // 2, self.last_x + oval_size // 2 + 1):
            for y in range(
                self.last_y - oval_size // 2, self.last_y + oval_size // 2 + 1
            ):
                if (x - self.last_x) ** 2 + (y - self.last_y) ** 2 <= (
                    oval_size // 2
                ) ** 2:
                    self.boarders.add((x, y))


class Task1bWindow:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.geometry("800x500+500+20")
        self.width = 800
        self.height = 600
        self.root.configure(bg=parent.back_ground)
        self.root.title("task1b")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.image_path = ""
        self.image = None

        self.boarders = set()
        self.passed_val = set()

        self.fill_button = tk.Button(
            root,
            text="fill",
            command=self.fill,
            bg="#555",
            fg="white",
            width=100,
        )
        self.fill_button.pack(side=tk.TOP, pady=5)

        self.browse_button = tk.Button(
            root,
            text="browse",
            command=self.browse_file,
            bg="#555",
            fg="white",
            width=100,
        )
        self.browse_button.pack(side=tk.TOP, pady=5)

        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if filename:
            self.image_path = filename
            self.image = Image.open(self.image_path)

    def fill(self):
        self.canvas.unbind("<B1-Motion>")
        self.canvas.bind("<Button-1>", self.recursive_line_fill)

    def check_validity(self, x, y):
        return(
                (x, y) not in self.boarders
                and (x, y) not in self.passed_val
                and self.width - 2 > x > 0 < y < self.height - 2
        )


    def get_hex_pixel(self, x, y):
        r, g, b = self.image.getpixel((x % self.image.width, y % self.image.height))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def recursive_line_fill(self, event):
        self.canvas.unbind("<Button-1>")
        x, y = event.x, event.y

        if self.image is None:
            print("Загрузите изображение для паттерна")
            self.canvas.bind("<B1-Motion>", self.paint)
            return

        def fill_scanline(x, y):
            if not self.check_validity(x, y):
                return

            x_left = x
            while self.check_validity(x_left - 1, y):
                x_left -= 1

            x_right = x
            while self.check_validity(x_right + 1, y):
                x_right += 1

            for nx in range(x_left, x_right + 1):
                img_color = self.get_hex_pixel(nx, y)
                self.canvas.create_oval(
                    nx, y, nx + 1, y + 1, fill=img_color, outline=img_color
                )
                self.passed_val.add((nx, y))

            self.root.update()

            for nx in range(x_left, x_right + 1):
                fill_scanline(nx, y + 1)
                fill_scanline(nx, y - 1)

        fill_scanline(x, y)
        print("Done")
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        self.last_x, self.last_y = event.x, event.y
        oval_size = 10

        self.canvas.create_oval(
            self.last_x - oval_size // 2,
            self.last_y - oval_size // 2,
            self.last_x + oval_size // 2,
            self.last_y + oval_size // 2,
            fill="black",
            outline="black",
        )

        for x in range(self.last_x - oval_size // 2, self.last_x + oval_size // 2 + 1):
            for y in range(
                self.last_y - oval_size // 2, self.last_y + oval_size // 2 + 1
            ):
                if (x - self.last_x) ** 2 + (y - self.last_y) ** 2 <= (
                    oval_size // 2
                ) ** 2:
                    self.boarders.add((x, y))


class Task1cWindow:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.configure(bg=parent.back_ground)
        self.passed_val = set()
        self.color_for_fill = "#00CCCC"

        self.root.title("Task1c")

        self.image = None

        self.browse_button = tk.Button(
            root,
            text="Browse Image",
            command=self.browse_file,
            bg="#555",
            fg="white",
            width=20,
        )
        self.browse_button.pack(side=tk.TOP, pady=5)

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.boundary_trace)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if filename:
            self.image_path = filename
            self.image = Image.open(self.image_path)

            max_size = 500
            if self.image.width > max_size or self.image.height > max_size:
                ratio = min(max_size / self.image.width, max_size / self.image.height)
                new_width = int(self.image.width * ratio)
                new_height = int(self.image.height * ratio)
                self.image = self.image.resize((new_width, new_height), Image.LANCZOS)

            self.width, self.height = self.image.size
            self.canvas.config(width=self.width, height=self.height)
            self.canvas.delete("all")
            self.draw_image()

    def draw_image(self):
        for x in range(self.width):
            for y in range(self.height):
                r, g, b = self.image.getpixel((x, y))
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                self.canvas.create_rectangle(
                    x, y, x + 1, y + 1, outline=hex_color, fill=hex_color
                )

    # Если она не помечена, внутри картинки и цвет точно совпадает с color
    def check_validity_and_color(self, x, y, color):
        if (
            (x, y) not in self.passed_val
            and 0 <= x < self.width
            and 0 <= y < self.height
        ):
            r, g, b = self.image.getpixel((x, y))
            current_color = f"#{r:02x}{g:02x}{b:02x}"
            return current_color == color
        else:
            return False

    #проверяет, похож ли цвет
    def check_validity_and_color_in_range(self, x, y, color, threshold=50):
        if (
            (x, y) not in self.passed_val
            and 0 <= x < self.width
            and 0 <= y < self.height
        ):
            r, g, b = self.image.getpixel((x, y))
            current_color = (r, g, b)

            target_r = int(color[1:3], 16)
            target_g = int(color[3:5], 16)
            target_b = int(color[5:7], 16)
            color_rgb = (target_r, target_g, target_b)

            distance = math.sqrt(
                (current_color[0] - color_rgb[0]) ** 2
                + (current_color[1] - color_rgb[1]) ** 2
                + (current_color[2] - color_rgb[2]) ** 2
            )

            return distance <= threshold
        else:
            return False

    #Если хоть один сосед имеет другой цвет
    def is_boundary_pixel(self, x, y, color):
        directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                r, g, b = self.image.getpixel((nx, ny))
                n_color = f"#{r:02x}{g:02x}{b:02x}"
                if n_color != color:
                    return True
        return False

    def boundary_trace(self, event):
        x, y = event.x, event.y

        r, g, b = self.image.getpixel((x, y))
        color = f"#{r:02x}{g:02x}{b:02x}"

        if not self.is_boundary_pixel(x, y, color):
            print("Щелчок не на граничном пикселе")
            return

        boundary = []

        directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]

        cur_x, cur_y = x, y

        if not self.check_validity_and_color_in_range(cur_x, cur_y, color):
            print("Щелчок не на границе")
            return

        self.passed_val.add((cur_x, cur_y))
        boundary.append((cur_x, cur_y))

        back_dir = 4  # Начальный back - с запада

        frames_count = 0

        while True:
            found = False
            start_dir = (back_dir + 2) % 8  # Право от back для clockwise

            for i in range(8):
                d = (start_dir + i) % 8
                nx = cur_x + directions[d][0]
                ny = cur_y + directions[d][1]
                if self.check_validity_and_color_in_range(nx, ny, color) and self.is_boundary_pixel(nx, ny, color):
                    boundary.append((nx, ny))
                    self.passed_val.add((nx, ny))
                    cur_x, cur_y = nx, ny
                    back_dir = (d + 4) % 8
                    found = True

                    frames_count += 1
                    if frames_count % 100 == 0:
                        self.root.update()

                    break

            if not found:
                break

            if cur_x == x and cur_y == y and len(boundary) > 2:
                break

        # Прорисовка границы поверх
        for px, py in boundary:
            self.canvas.create_oval(
                px, py, px + 1, py + 1, fill=self.color_for_fill, outline=self.color_for_fill
            )

        print("Done, граница:", len(boundary), "точек")