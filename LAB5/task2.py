import tkinter as tk
import random

class MountainVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Task2")

        # Настройка холста для рисования
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Параметры
        self.roughness = tk.DoubleVar(value=1.0)
        self.iterations = tk.IntVar(value=5)
        self.step_delay = tk.IntVar(value=500)  # Задержка между шагами в миллисекундах

        # Панель управления
        control_frame = tk.Frame(self.root)
        control_frame.pack()

        tk.Label(control_frame, text="Roughness:").pack(side=tk.LEFT)
        self.roughness_slider = tk.Scale(control_frame, from_=0.1, to=2.0, resolution=0.1,
                                         orient=tk.HORIZONTAL, variable=self.roughness)
        self.roughness_slider.pack(side=tk.LEFT)

        tk.Label(control_frame, text="Iterations:").pack(side=tk.LEFT)
        self.iterations_slider = tk.Scale(control_frame, from_=1, to=10,
                                          orient=tk.HORIZONTAL, variable=self.iterations)
        self.iterations_slider.pack(side=tk.LEFT)

        tk.Label(control_frame, text="Step Delay (ms):").pack(side=tk.LEFT)
        self.delay_slider = tk.Scale(control_frame, from_=100, to=2000, orient=tk.HORIZONTAL,
                                     variable=self.step_delay)
        self.delay_slider.pack(side=tk.LEFT)

        tk.Button(control_frame, text="Generate", command=self.start_generation).pack(side=tk.LEFT)

        self.canvas.bind("<Configure>", self.on_resize)

        self.points = []  # Хранение точек для рисования

    def on_resize(self, event):
        self.generate_mountain()

    def midpoint_displacement_step(self, points, roughness):
        """Функция, выполняющая один шаг алгоритма и возвращающая новые точки."""
        new_points = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2

            # Смещение точки по оси Y с учетом грубости
            displacement = random.uniform(-1, 1) * roughness
            mid_y += displacement

            new_points.append(p1)
            new_points.append((mid_x, mid_y))

        new_points.append(points[-1])
        return new_points

    def draw_mountain(self):
        """Рисует горный массив на холсте."""
        self.canvas.delete("all")
        if len(self.points) < 2:
            return

        for i in range(len(self.points) - 1):
            self.canvas.create_line(self.points[i][0], self.points[i][1],
                                    self.points[i + 1][0], self.points[i + 1][1],
                                    fill="black", width=2)

    def animate_step(self, depth, roughness):
        """Анимирует шаги алгоритма и перерисовывает линии."""
        if depth == 0:
            return

        # Выполняем шаг алгоритма Midpoint Displacement
        self.points = self.midpoint_displacement_step(self.points, roughness)
        self.draw_mountain()

        # Уменьшаем грубость для следующего шага и планируем следующий шаг через задержку
        self.root.after(self.step_delay.get(), self.animate_step, depth - 1, roughness / 2)

    def start_generation(self):
        """Инициализация точек и запуск анимации генерации."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # Начальные точки (по краям экрана)
        self.points = [(0, height // 2), (width, height // 2)]

        roughness = self.roughness.get() * (height // 2)
        iterations = self.iterations.get()

        # Старт анимации
        self.animate_step(iterations, roughness)

    def generate_mountain(self):
        """Очищает холст и перерисовывает на resize."""
        self.start_generation()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Lab 4")
    app = MountainVisualizer(root)
    root.mainloop()