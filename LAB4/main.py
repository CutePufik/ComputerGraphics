import tkinter as tk
import numpy as np
from math import cos, sin, radians


class PolygonEditor:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Панель управления
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_label = tk.Label(control_frame, text="Выберите действие", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Поля для ввода смещения
        self.dx_entry = self.create_labeled_entry(control_frame, "dx:")
        self.dy_entry = self.create_labeled_entry(control_frame, "dy:")
        self.translate_btn = tk.Button(control_frame, text="Смещение", command=self.translate)
        self.translate_btn.pack(fill=tk.X, padx=10, pady=5)

        # Поля для ввода угла поворота
        self.angle_entry = self.create_labeled_entry(control_frame, "Угол (градусы):")

        # Поля для ввода точки вращения и масштабирования
        self.status_label_point = tk.Label(control_frame, text="Точка вращения и масштабирования", font=("Arial", 12))
        self.status_label_point.pack(pady=10)

        self.x_entry = self.create_labeled_entry(control_frame, "X:")
        self.y_entry = self.create_labeled_entry(control_frame, "Y:")

        # Кнопка Поворот
        self.rotate_btn = tk.Button(control_frame, text="Поворот", command=self.rotate)
        self.rotate_btn.pack(fill=tk.X, padx=10, pady=5)

        # Масштабирование
        self.scale_entry = self.create_labeled_entry(control_frame, "Коэффициент масштаба:")
        self.scale_btn = tk.Button(control_frame, text="Масштабирование", command=self.scale)
        self.scale_btn.pack(fill=tk.X, padx=10, pady=5)

        self.clear_btn = tk.Button(control_frame, text="Очистить сцену", command=self.clear_scene)
        self.clear_btn.pack(fill=tk.X, padx=10, pady=5)

        self.intersections_button = tk.Button(control_frame, text="Пересечения", command=self.check_polygon_intersections)
        self.intersections_button.pack(fill=tk.X, padx=10, pady=5)

        self.message_window = tk.Text(control_frame, height=10, width=40)
        self.message_window.pack(padx=10, pady=10)

        self.polygons = []
        self.current_polygon = []
        self.selected_polygon = None

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.check_point)

    def create_labeled_entry(self, parent, label_text):
        frame = tk.Frame(parent)
        frame.pack(padx=10, pady=5, fill=tk.X)
        label = tk.Label(frame, text=label_text)
        label.pack(side=tk.LEFT)
        entry = tk.Entry(frame)
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return entry

    def get_input_point(self):
        x_text = self.x_entry.get()
        y_text = self.y_entry.get()
        if x_text and y_text:
            try:
                return float(x_text), float(y_text)
            except ValueError:
                self.message_window.insert(tk.END, "Неверный формат координат точки\n")
        return None

    def add_point(self, event):
        x, y = event.x, event.y
        self.current_polygon.append((x, y))
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        if len(self.current_polygon) > 1:
            self.canvas.create_line(self.current_polygon[-2], self.current_polygon[-1], fill="black")
        if len(self.current_polygon) == 2:
            self.selected_polygon = self.current_polygon
            self.polygons.append(self.current_polygon)

    def clear_scene(self):
        self.canvas.delete("all")
        self.polygons.clear()
        self.current_polygon = []
        self.status_label.config(text="Сцена очищена")
        self.message_window.delete(1.0, tk.END)

    def translate(self):
        if not self.selected_polygon:
            return
        dx = float(self.dx_entry.get() or 0)
        dy = float(self.dy_entry.get() or 0)
        translation_matrix = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
        for i, (x, y) in enumerate(self.selected_polygon):
            p = np.array([x, y, 1])
            new_p = translation_matrix @ p
            self.selected_polygon[i] = (new_p[0], new_p[1])
        self.redraw()

    def rotate(self):
        if not self.selected_polygon:
            return
        angle = radians(float(self.angle_entry.get() or 0))
        point = self.get_input_point() or self.get_polygon_center(self.selected_polygon)
        # матрица смещения к точке, поворот, обратное смещение
        T1 = np.array([[1,0,-point[0]],[0,1,-point[1]],[0,0,1]])
        R = np.array([[cos(angle),-sin(angle),0],[sin(angle),cos(angle),0],[0,0,1]])
        T2 = np.array([[1,0,point[0]],[0,1,point[1]],[0,0,1]])
        M = T2 @ R @ T1
        for i, (x,y) in enumerate(self.selected_polygon):
            p = np.array([x,y,1])
            new_p = M @ p
            self.selected_polygon[i] = (new_p[0], new_p[1])
        self.redraw()

    def scale(self):
        if not self.selected_polygon:
            return
        s = float(self.scale_entry.get() or 1)
        point = self.get_input_point() or self.get_polygon_center(self.selected_polygon)
        T1 = np.array([[1,0,-point[0]],[0,1,-point[1]],[0,0,1]])
        S = np.array([[s,0,0],[0,s,0],[0,0,1]])
        T2 = np.array([[1,0,point[0]],[0,1,point[1]],[0,0,1]])
        M = T2 @ S @ T1
        for i, (x,y) in enumerate(self.selected_polygon):
            p = np.array([x,y,1])
            new_p = M @ p
            self.selected_polygon[i] = (new_p[0], new_p[1])
        self.redraw()

    def get_polygon_center(self, polygon):
        xs = [x for x,_ in polygon]
        ys = [y for _,y in polygon]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    def redraw(self):
        self.canvas.delete("all")
        for polygon in self.polygons:
            if len(polygon)>1:
                self.canvas.create_polygon(polygon, outline="black", fill="", width=2)


    def check_point(self, event):
        """Проверка принадлежности точки полигону и классификация"""
        x, y = event.x, event.y
        self.message_window.delete(1.0, tk.END)  # Очистка предыдущих сообщений

        for polygon in self.polygons:
            if self.is_point_inside_polygon((x, y), polygon):
                self.message_window.insert(tk.END, "Точка внутри полигона\n")
            else:
                self.message_window.insert(tk.END, "Точка снаружи полигона\n")

            if len(polygon) >= 2:
                self.classify_point_position((x, y), polygon)

    def is_point_inside_polygon(self, point, polygon):
        """Проверка принадлежности точки полигону"""
        x, y = point
        n = len(polygon)
        inside = False
        px, py = polygon[0]
        for i in range(n + 1):
            cx, cy = polygon[i % n]
            if y > min(py, cy):
                if y <= max(py, cy):
                    if x <= max(px, cx):
                        if py != cy:
                            xinters = (y - py) * (cx - px) / (cy - py) + px
                        if px == cx or x <= xinters:
                            inside = not inside
            px, py = cx, cy
        return inside

    def classify_point_position(self, point, polygon):
        """Классификация положения точки относительно прямых полигона"""
        px, py = point
        for i in range(len(polygon) - 1):
            x1, y1 = polygon[i]
            x2, y2 = polygon[i + 1]
            position = self.get_point_position_relative_to_line(px, py, x1, y1, x2, y2)
            line_description = f"Точка ({px}, {py}) относительно прямой ({x1}, {y1}) - ({x2}, {y2}): {position}\n"
            self.message_window.insert(tk.END, line_description)

    def get_point_position_relative_to_line(self, px, py, x1, y1, x2, y2):
        """Определение положения точки относительно прямой"""
        determinant = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if determinant < 0:
            return "Слева"
        elif determinant > 0:
            return "Справа"
        else:
            return "На линии"

    def check_polygon_intersections(self):
        """Проверка пересечений рёбер полигона"""
        self.redraw()
        self.message_window.delete(1.0, tk.END)  # Очистка предыдущих сообщений

        polygon = self.selected_polygon
        n = len(polygon)
        if n < 3:
            self.message_window.insert(
                tk.END, "Полигон не может иметь менее 3 вершин\n"
            )
            return

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]  # Замыкаем ребро с первой точкой

            # Проверяем на пересечение только с не смежными рёбрами
            for j in range(
                i + 2, n - 1
            ):  # Проверка на пересечение со всеми не смежными рёбрами
                if j % n == i or j % n == (i + 1) % n:  # Избегаем смежных рёбер
                    continue

                x3, y3 = polygon[j % n]
                x4, y4 = polygon[(j + 1) % n]

                intersection_point = self.get_intersection_point(
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4)
                )

                if intersection_point:
                    ix, iy = intersection_point
                    intersection = f"Ребро ({x1}, {y1})-({x2}, {y2}) пересекается с ребром ({x3}, {y3})-({x4}, {y4}) в точке ({ix}, {iy})\n"
                    self.canvas.create_oval(
                        ix - 5, iy - 5, ix + 5, iy + 5, fill="green"
                    )
                    self.message_window.insert(tk.END, intersection)

    def get_intersection_point(self, p1, p2, p3, p4):
        """Нахождение точки пересечения двух отрезков p1p2 и p3p4, если она существует"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Отрезки параллельны

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        # Проверяем, лежат ли параметры t и u в диапазоне от 0 до 1 (отрезки пересекаются в пределах их длины)
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Вычисляем точку пересечения
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return ix, iy

        return None


root = tk.Tk()
app = PolygonEditor(root)
root.mainloop()