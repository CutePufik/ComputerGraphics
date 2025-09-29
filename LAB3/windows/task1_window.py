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
        task1a_window = Task1bWindow(root=child, parent=self)

    def task1c(self):
        child = tk.Tk()
        task1a_window = Task1cWindow(root=child, parent=self)


class Task1aWindow:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.configure(bg=parent.back_ground)
        self.width = 800
        self.height = 600
        self.root.title("task1a")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)

        self.frames_for_update = 1000

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
        self.fill_button.pack(side=tk.TOP, pady=10)

        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None

    def fill(self):
        self.canvas.unbind("<B1-Motion>")
        self.canvas.bind("<B1-Motion>", self.stack_fill_lines)

    def check_validity(self, x, y):
        if (
            (x, y) not in self.boarders
            and (x, y) not in self.passed_val
            and x > 0
            and x < self.width - 2
            and y > 0
            and y < self.height - 2
        ):
            return True
        else:
            return False

    def stack_fill(self, event):
        self.canvas.unbind("<B1-Motion>")
        x_start, y_start = event.x, event.y
        stack = [(x_start, y_start)]
        frames_count = 0
        while stack:
            frames_count += 1
            x, y = stack.pop()
            self.canvas.create_oval(x, y, x + 1, y + 1, fill="red", outline="red")
            if frames_count % self.frames_for_update == 0:
                self.root.update()
            if self.check_validity(x + 1, y):
                self.passed_val.add((x + 1, y))
                stack.append((x + 1, y))

            if self.check_validity(x - 1, y):
                self.passed_val.add((x - 1, y))
                stack.append((x - 1, y))

            if self.check_validity(x, y + 1):
                self.passed_val.add((x, y + 1))
                stack.append((x, y + 1))

            if self.check_validity(x, y - 1):
                self.passed_val.add((x, y - 1))
                stack.append((x, y - 1))

        print("Done")
        self.canvas.bind("<B1-Motion>", self.paint)
        return

    def stack_fill_lines(self, event):
        self.canvas.unbind("<B1-Motion>")
        x_start, y_start = event.x, event.y
        stack = [(x_start, y_start)]
        pixel_count = 0

        while stack:
            pixel_count += 1
            x, y = stack.pop()

            x_left = x
            while self.check_validity(x_left - 1, y):
                x_left -= 1

            x_right = x
            while self.check_validity(x_right + 1, y):
                x_right += 1

            self.canvas.create_line(x_left, y, x_right, y, fill="red")

            for nx in range(x_left, x_right + 1):
                self.passed_val.add((nx, y))

            if pixel_count % self.frames_for_update == 0:
                self.root.update()

            for nx in range(x_left, x_right + 1):
                if self.check_validity(nx, y + 1):
                    stack.append((nx, y + 1))

                if self.check_validity(nx, y - 1):
                    stack.append((nx, y - 1))

        print("Done")
        self.canvas.bind("<B1-Motion>", self.paint)

    def rec_fill(self, event):
        passed_val = set()
        self.canvas.unbind("<B1-Motion>")
        x_start, y_start = event.x, event.y

        def f(self: Task1aWindow, x, y):
            self.canvas.create_oval(x, y, x + 1, y + 1, fill="red", outline="red")
            self.canvas.update()

            if self.check_validity(x + 1, y):
                passed_val.add((x + 1, y))
                f(self, x + 1, y)

            if self.check_validity(x - 1, y):
                passed_val.add((x - 1, y))
                f(self, x - 1, y)

            if self.check_validity(x, y + 1):
                passed_val.add((x, y + 1))
                f(self, x, y + 1)

            if self.check_validity(x, y - 1):
                passed_val.add((x, y - 1))
                f(self, x, y - 1)

        f(self=self, x=x_start, y=y_start)
        print("Done")
        self.canvas.bind("<B1-Motion>", self.paint)
        return

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
        self.frames_for_update = 1000

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
        self.canvas.bind("<B1-Motion>", self.stack_fill_lines)

    def check_validity(self, x, y):
        if (
            (x, y) not in self.boarders
            and (x, y) not in self.passed_val
            and x > 0
            and x < self.width - 2
            and y > 0
            and y < self.height - 2
        ):
            return True
        else:
            return False

    def get_hex_pixel(self, x, y):
        r, g, b = self.image.getpixel((x % self.image.width, y % self.image.height))
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def stack_fill(self, event):
        self.canvas.unbind("<B1-Motion>")
        x_start, y_start = event.x, event.y
        stack = [(x_start, y_start)]

        frame_count = 0
        while stack:
            frame_count += 1
            x, y = stack.pop()
            color = self.get_hex_pixel(x, y)
            self.canvas.create_oval(x, y, x + 1, y + 1, fill=color, outline=color)
            if frame_count % self.frames_for_update == 0:
                self.canvas.update()
            if self.check_validity(x + 1, y):
                self.passed_val.add((x + 1, y))
                stack.append((x + 1, y))

            if self.check_validity(x - 1, y):
                self.passed_val.add((x - 1, y))
                stack.append((x - 1, y))

            if self.check_validity(x, y + 1):
                self.passed_val.add((x, y + 1))
                stack.append((x, y + 1))

            if self.check_validity(x, y - 1):
                self.passed_val.add((x, y - 1))
                stack.append((x, y - 1))

        print("Done")
        self.canvas.bind("<B1-Motion>", self.paint)
        return

    def stack_fill_lines(self, event):
        self.canvas.unbind("<B1-Motion>")
        x_start, y_start = event.x, event.y
        stack = [(x_start, y_start)]
        pixel_count = 0

        while stack:
            pixel_count += 1
            x, y = stack.pop()

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

            for nx in range(x_left, x_right + 1):
                self.passed_val.add((nx, y))

            if pixel_count % self.frames_for_update == 0:
                self.root.update()

            for nx in range(x_left, x_right + 1):
                if self.check_validity(nx, y + 1):
                    stack.append((nx, y + 1))

                if self.check_validity(nx, y - 1):
                    stack.append((nx, y - 1))

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
        self.frames_for_update = 10000
        self.passed_val = set()
        self.color_for_fill = "#00CCCC"

        self.root.title("Task1c")

        self.image_tk = None
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
        self.image_tk = None
        self.image = None
        self.canvas.bind("<B1-Motion>", self.connected_area)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")]
        )
        if filename:
            self.image_path = filename
            self.image = Image.open(self.image_path)

            # Resize the image if necessary
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
            else:

                self.width, self.height = self.image.size
                self.canvas.config(width=self.width, height=self.height)
                self.canvas.delete("all")
                self.draw_image()

    def draw_image(self):
        frames_count = 0
        for x in range(self.width):
            for y in range(self.height):
                frames_count += 1
                if frames_count % self.frames_for_update == 0:
                    self.root.update()
                r, g, b = self.image.getpixel((x, y))
                hex_color = f"#{r:02x}{g:02x}{b:02x}"

                self.canvas.create_rectangle(
                    x, y, x + 1, y + 1, outline=hex_color, fill=hex_color
                )

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

    def connected_area(self, event):
        # check_function = self.check_validity_and_color
        check_function = self.check_validity_and_color_in_range
        self.canvas.unbind("<B1-Motion>")
        x_start, y_start = event.x, event.y

        r, g, b = self.image.getpixel((x_start, y_start))
        color = f"#{r:02x}{g:02x}{b:02x}"

        stack = [(x_start, y_start)]
        self.passed_val.add((x_start, y_start))

        frames_count = 0
        while stack:
            x, y = stack.pop()
            self.canvas.create_oval(
                x,
                y,
                x + 1,
                y + 1,
                fill=self.color_for_fill,
                outline=self.color_for_fill,
            )
            frames_count += 1
            if frames_count % (self.frames_for_update // 10) == 0:
                self.root.update()

            if check_function(x + 1, y, color):
                self.passed_val.add((x + 1, y))
                stack.append((x + 1, y))

            if check_function(x - 1, y, color):
                self.passed_val.add((x - 1, y))
                stack.append((x - 1, y))

            if check_function(x, y + 1, color):
                self.passed_val.add((x, y + 1))
                stack.append((x, y + 1))

            if check_function(x, y - 1, color):
                self.passed_val.add((x, y - 1))
                stack.append((x, y - 1))

        print("Done")
        self.canvas.bind("<B1-Motion>", self.connected_area)
        return