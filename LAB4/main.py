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

        self.status_label = tk.Label(
            control_frame, text="Режим: Построение полигонов", font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        # Кнопки режимов работы
        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(pady=10)
        
        self.build_btn = tk.Button(
            mode_frame, text="Построение", command=lambda: self.set_mode("build")
        )
        self.build_btn.pack(side=tk.LEFT, padx=5)
        
        self.intersect_btn = tk.Button(
            mode_frame, text="Пересечения", command=lambda: self.set_mode("intersect")
        )
        self.intersect_btn.pack(side=tk.LEFT, padx=5)
        
        self.point_check_btn = tk.Button(
            mode_frame, text="Проверка точки", command=lambda: self.set_mode("point_check")
        )
        self.point_check_btn.pack(side=tk.LEFT, padx=5)

        # Поля для ввода смещения
        self.dx_entry = self.create_labeled_entry(control_frame, "dx:")
        self.dy_entry = self.create_labeled_entry(control_frame, "dy:")
        self.translate_btn = tk.Button(
            control_frame, text="Смещение", command=self.translate
        )
        self.translate_btn.pack(fill=tk.X, padx=10, pady=5)

        # Поля для ввода угла поворота
        self.angle_entry = self.create_labeled_entry(control_frame, "Угол (градусы):")

        # Поля для ввода точки вращения и масштабирования
        self.status_label = tk.Label(
            control_frame, text="Точка вращения и масштабирования", font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        # Поля для ввода пользовательской точки (x, y)
        self.x_entry = self.create_labeled_entry(control_frame, "X:")
        self.y_entry = self.create_labeled_entry(control_frame, "Y:")

        # Кнопка Поворот
        self.rotate_btn = tk.Button(control_frame, text="Поворот", command=self.rotate)
        self.rotate_btn.pack(fill=tk.X, padx=10, pady=5)

        # Поле для ввода коэффициента масштабирования
        self.scale_entry = self.create_labeled_entry(
            control_frame, "Коэффициент масштаба:"
        )
        self.scale_btn = tk.Button(
            control_frame, text="Масштабирование", command=self.scale
        )
        self.scale_btn.pack(fill=tk.X, padx=10, pady=5)

        self.clear_btn = tk.Button(
            control_frame, text="Очистить сцену", command=self.clear_scene
        )
        self.clear_btn.pack(fill=tk.X, padx=10, pady=5)

        # Окно для сообщений
        self.message_window = tk.Text(control_frame, height=10, width=40)
        self.message_window.pack(padx=10, pady=10)

        # === ПУНКТ 1: Инициализация данных для создания полигонов ===
        self.polygons = []  # список всех полигонов
        self.current_polygon = []  # текущий строящийся полигон
        self.selected_polygon_idx = None  # индекс выбранного полигона для трансформаций
        self.mode = "build"  # режим работы: build, intersect, point_check
        
        # === ПУНКТ 4: Данные для проверки пересечения ребер ===
        self.intersection_edges = []  # список ребер для проверки пересечений
        
        # Привязка событий мыши
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<Button-3>", self.select_polygon)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)

    def create_labeled_entry(self, parent, label_text):
        """Функция для создания поля ввода с меткой"""
        frame = tk.Frame(parent)
        frame.pack(padx=10, pady=5, fill=tk.X)
        label = tk.Label(frame, text=label_text)
        label.pack(side=tk.LEFT)
        entry = tk.Entry(frame)
        entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return entry

    def set_mode(self, mode):
        """Установка режима работы"""
        self.mode = mode
        self.message_window.delete(1.0, tk.END)
        
        if mode == "build":
            self.status_label.config(text="Режим: Построение полигонов")
            self.message_window.insert(tk.END, "Режим построения: клик - добавить точку, двойной клик - завершить полигон\n")
        elif mode == "intersect":
            self.status_label.config(text="Режим: Проверка пересечений")
            self.intersection_edges = []  # сброс ребер для пересечений
            self.message_window.insert(tk.END, "Режим пересечений: кликните 4 точки для двух ребер\n")
        elif mode == "point_check":
            self.status_label.config(text="Режим: Проверка точки")
            self.message_window.insert(tk.END, "Режим проверки точки: кликните для проверки точки\n")

    def canvas_click(self, event):
        """Обработка кликов мыши в зависимости от режима"""
        x, y = event.x, event.y
        
        if self.mode == "build":
            self.add_point(x, y)
        elif self.mode == "intersect":
            self.add_intersection_edge_point(x, y)
        elif self.mode == "point_check":
            self.check_point(x, y)

    # === ПУНКТ 1: Создание полигонов через клики мышью ===
    def add_point(self, x, y):
        """Добавление вершины полигона"""
        self.current_polygon.append((x, y))
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black", tags="point")

        # Если точек больше одной, рисуем ребро
        if len(self.current_polygon) > 1:
            self.canvas.create_line(
                self.current_polygon[-2], self.current_polygon[-1], 
                fill="blue", width=2, tags="edge"
            )

    def finish_polygon(self, event):
        """Завершение построения полигона"""
        if len(self.current_polygon) > 0:
            # === ПУНКТ 1: Поддержка полигонов с 1 и 2 вершинами ===
            if len(self.current_polygon) == 1:
                # Точка - полигон с одной вершиной
                self.polygons.append(self.current_polygon.copy())
                self.message_window.insert(tk.END, f"Создана точка (полигон с 1 вершиной)\n")
            elif len(self.current_polygon) == 2:
                # Ребро - полигон с двумя вершинами
                self.polygons.append(self.current_polygon.copy())
                self.message_window.insert(tk.END, f"Создано ребро (полигон с 2 вершинами)\n")
            else:
                # Многоугольник - замыкаем
                self.polygons.append(self.current_polygon.copy())
                # Замыкаем полигон
                self.canvas.create_line(
                    self.current_polygon[-1], self.current_polygon[0], 
                    fill="blue", width=2, tags="polygon"
                )
                self.message_window.insert(tk.END, f"Создан полигон с {len(self.current_polygon)} вершинами\n")
            
            self.current_polygon = []
            self.redraw_polygons()
    def select_polygon(self, event):
        """Улучшенный выбор полигона"""
        x, y = event.x, event.y
        self.selected_polygon_idx = None
        
        # Проверяем полигоны в обратном порядке (последние нарисованные сверху)
        for i in range(len(self.polygons)-1, -1, -1):
            polygon = self.polygons[i]
            
            # Для точки
            if len(polygon) == 1:
                px, py = polygon[0]
                if abs(px - x) <= 5 and abs(py - y) <= 5:
                    self.selected_polygon_idx = i
                    break
            # Для ребра
            elif len(polygon) == 2:
                if self.is_point_near_line((x, y), polygon[0], polygon[1], threshold=5):
                    self.selected_polygon_idx = i
                    break
            # Для многоугольника
            else:
                if self.is_point_inside_polygon((x, y), polygon):
                    self.selected_polygon_idx = i
                    break
        
        if self.selected_polygon_idx is not None:
            self.redraw_polygons()

    # === ПУНКТ 2: Очистка сцены ===
    def clear_scene(self):
        """Очистка сцены"""
        self.canvas.delete("all")
        self.polygons.clear()
        self.current_polygon = []
        self.selected_polygon_idx = None
        self.intersection_edges = []
        self.message_window.delete(1.0, tk.END)
        self.message_window.insert(tk.END, "Сцена очищена\n")

    def get_input_point(self):
        """Получить координаты точки из полей ввода X и Y, если они не пустые"""
        x_text = self.x_entry.get()
        y_text = self.y_entry.get()

        if x_text and y_text:
            try:
                return float(x_text), float(y_text)
            except ValueError:
                self.message_window.insert(tk.END, "Неверный формат координат точки\n")
                return None
        return None

    def get_selected_polygon(self):
        """Получить выбранный полигон"""
        if self.selected_polygon_idx is not None and self.selected_polygon_idx < len(self.polygons):
            return self.polygons[self.selected_polygon_idx]
        return None

    # === ПУНКТ 3: Аффинные преобразования с матрицами ===
    def translate(self):
        """3.1 Смещение полигона на dx, dy"""
        polygon = self.get_selected_polygon()
        if polygon:
            dx = float(self.dx_entry.get() or 0)
            dy = float(self.dy_entry.get() or 0)
            
            # Матрица смещения
            translation_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

            for i, (x, y) in enumerate(polygon):
                point = np.array([x, y, 1])
                new_point = translation_matrix @ point
                polygon[i] = (new_point[0], new_point[1])

            self.redraw_polygons()
            self.message_window.insert(tk.END, f"Полигон смещен на ({dx}, {dy})\n")

    def rotate(self):
        """3.2 Поворот вокруг заданной точки или 3.3 вокруг центра"""
        polygon = self.get_selected_polygon()
        if polygon:
            angle = radians(float(self.angle_entry.get() or 0))
            
            # Определяем точку вращения
            rotation_point = self.get_input_point()
            if rotation_point is None:
                # 3.3 Поворот вокруг своего центра
                rotation_point = self.get_polygon_center(polygon)
                self.message_window.insert(tk.END, "Поворот вокруг центра полигона\n")
            else:
                # 3.2 Поворот вокруг заданной точки
                self.message_window.insert(tk.END, f"Поворот вокруг точки {rotation_point}\n")

            # Матрица поворота
            rotation_matrix = np.array([
                [cos(angle), -sin(angle), 0],
                [sin(angle), cos(angle), 0],
                [0, 0, 1]
            ])

            for i, (x, y) in enumerate(polygon):
                # Перенос в начало координат
                translated_point = np.array([x - rotation_point[0], y - rotation_point[1], 1])
                # Поворот
                rotated_point = rotation_matrix @ translated_point
                # Обратный перенос
                polygon[i] = (
                    rotated_point[0] + rotation_point[0],
                    rotated_point[1] + rotation_point[1]
                )

            self.redraw_polygons()

    def scale(self):
        """3.4 Масштабирование относительно заданной точки или 3.5 относительно центра"""
        polygon = self.get_selected_polygon()
        if polygon:
            scale_factor = float(self.scale_entry.get() or 1)
            
            # Определяем точку масштабирования
            scaling_point = self.get_input_point()
            if scaling_point is None:
                # 3.5 Масштабирование относительно своего центра
                scaling_point = self.get_polygon_center(polygon)
                self.message_window.insert(tk.END, "Масштабирование относительно центра полигона\n")
            else:
                # 3.4 Масштабирование относительно заданной точки
                self.message_window.insert(tk.END, f"Масштабирование относительно точки {scaling_point}\n")

            # Матрица масштабирования
            scaling_matrix = np.array([
                [scale_factor, 0, 0],
                [0, scale_factor, 0],
                [0, 0, 1]
            ])

            for i, (x, y) in enumerate(polygon):
                # Перенос в начало координат
                translated_point = np.array([x - scaling_point[0], y - scaling_point[1], 1])
                # Масштабирование
                scaled_point = scaling_matrix @ translated_point
                # Обратный перенос
                polygon[i] = (
                    scaled_point[0] + scaling_point[0],
                    scaled_point[1] + scaling_point[1]
                )

            self.redraw_polygons()

    def get_polygon_center(self, polygon):
        """Нахождение центра полигона"""
        x_coords = [x for x, y in polygon]
        y_coords = [y for x, y in polygon]
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)

    def redraw_polygons(self):
        """Перерисовка всех полигонов с выделением выбранного"""
        self.canvas.delete("all")
        
        for i, polygon in enumerate(self.polygons):
            color = "red" if i == self.selected_polygon_idx else "blue"
            
            # Рисуем вершины
            for x, y in polygon:
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")
            
            # Рисуем ребра/полигон
            if len(polygon) == 1:
                # Точка
                x, y = polygon[0]
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline=color, width=2)
            elif len(polygon) == 2:
                # Ребро
                self.canvas.create_line(polygon[0], polygon[1], fill=color, width=2)
            else:
                # Многоугольник
                self.canvas.create_polygon(polygon, outline=color, fill="", width=2)

    # === ПУНКТ 4: Поиск точки пересечения двух ребер ===
    def add_intersection_edge_point(self, x, y):
        """Добавление точек для проверки пересечения двух ребер"""
        self.intersection_edges.append((x, y))
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="green", tags="intersection_point")
        
        # Рисуем ребро, если есть две точки
        if len(self.intersection_edges) % 2 == 0:
            idx = len(self.intersection_edges) - 2
            self.canvas.create_line(
                self.intersection_edges[idx], self.intersection_edges[idx+1], 
                fill="green", width=2, tags="intersection_edge"
            )
        
        # Когда есть 4 точки (2 ребра), проверяем пересечение
        if len(self.intersection_edges) == 4:
            self.check_edge_intersection()
            # Сбрасываем для следующей проверки
            self.intersection_edges = []

    def check_edge_intersection(self):
        """Проверка пересечения двух ребер"""
        if len(self.intersection_edges) == 4:
            edge1 = (self.intersection_edges[0], self.intersection_edges[1])
            edge2 = (self.intersection_edges[2], self.intersection_edges[3])
            
            intersection = self.get_intersection_point(edge1[0], edge1[1], edge2[0], edge2[1])
            
            self.message_window.delete(1.0, tk.END)
            if intersection:
                ix, iy = intersection
                self.message_window.insert(
                    tk.END, 
                    f"Ребра пересекаются в точке ({ix:.2f}, {iy:.2f})\n"
                )
                self.canvas.create_oval(ix-5, iy-5, ix+5, iy+5, fill="red", tags="intersection")
            else:
                self.message_window.insert(tk.END, "Ребра не пересекаются\n")

    def get_intersection_point(self, p1, p2, p3, p4):
        """Нахождение точки пересечения двух отрезков"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-10:
            return None  # Отрезки параллельны

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return ix, iy

        return None

    # === ПУНКТ 5: Проверка принадлежности точки полигонам ===
    def check_point(self, x, y):
        """Проверка принадлежности точки полигонам"""
        self.message_window.delete(1.0, tk.END)
        self.message_window.insert(tk.END, f"Проверка точки ({x}, {y}):\n")
        
        # Рисуем проверяемую точку
        self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="purple", tags="check_point")
        
        for i, polygon in enumerate(self.polygons):
            is_inside = self.is_point_inside_polygon((x, y), polygon)
            is_convex = self.is_polygon_convex(polygon)
            
            polygon_type = "выпуклый" if is_convex else "невыпуклый"
            result = "внутри" if is_inside else "снаружи"
            
            self.message_window.insert(
                tk.END, 
                f"Полигон {i+1} ({polygon_type}): точка {result}\n"
            )
            
            # === ПУНКТ 6: Классификация положения точки относительно ребер ===
            if len(polygon) >= 2:
                self.message_window.insert(tk.END, "Положение относительно ребер:\n")
                for j in range(len(polygon)):
                    p1 = polygon[j]
                    p2 = polygon[(j + 1) % len(polygon)]
                    position = self.classify_point_relative_to_edge((x, y), p1, p2)
                    self.message_window.insert(tk.END, f"  Ребро {j+1}: {position}\n")

    def is_point_inside_polygon(self, point, polygon):
        """Проверка принадлежности точки полигону (алгоритм трассировки луча)"""
        if len(polygon) < 3:
            return False
            
        x, y = point
        inside = False
        n = len(polygon)
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    def is_polygon_convex(self, polygon):
        """Корректная проверка выпуклости полигона"""
        if len(polygon) < 3:
            return True
            
        n = len(polygon)
        if n == 3:
            return True
            
        sign = None
        for i in range(n):
            # Векторы текущего ребра и следующего
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            x3, y3 = polygon[(i + 2) % n]
            
            # Векторное произведение
            cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            
            if abs(cross_product) > 1e-10:
                current_sign = 1 if cross_product > 0 else -1
                if sign is None:
                    sign = current_sign
                elif sign != current_sign:
                    return False
        return True

    # === ПУНКТ 6: Классификация положения точки относительно ребра ===
    def classify_point_relative_to_edge(self, point, edge_p1, edge_p2):
        """Классификация положения точки относительно ребра"""
        px, py = point
        x1, y1 = edge_p1
        x2, y2 = edge_p2
        
        # Векторное произведение
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        if abs(cross_product) < 1e-10:
            return "на линии"
        elif cross_product > 0:
            return "слева"
        else:
            return "справа"


root = tk.Tk()
root.title("Редактор полигонов")
app = PolygonEditor(root)
root.mainloop()