import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox


class Task3Window:
    def __init__(self, root: tk.Tk, parent):
        self.root = root
        self.parent = parent
        self.root.configure(bg=parent.back_ground)
        self.root.geometry("400x500+500+20")
        self.root.title("Triangle Gradient Settings")

        # Основной фрейм для элементов управления
        control_frame = tk.Frame(root, bg=parent.back_ground)
        control_frame.pack(pady=10, padx=10, fill=tk.X)

        # Поля для ввода координат вершин
        tk.Label(control_frame, text="Координаты вершин треугольника:", 
                bg=parent.back_ground, fg="white").pack(anchor="w", pady=(0, 5))
        
        self.entry_fields = []
        for i in range(3):
            vertex_frame = tk.Frame(control_frame, bg=parent.back_ground)
            vertex_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(vertex_frame, text=f"Вершина {i+1}:", 
                    bg=parent.back_ground, fg="white", width=10).pack(side=tk.LEFT)
            entry_x = tk.Entry(vertex_frame, width=8)
            entry_x.pack(side=tk.LEFT, padx=2)
            tk.Label(vertex_frame, text="X", 
                    bg=parent.back_ground, fg="white").pack(side=tk.LEFT)
            
            entry_y = tk.Entry(vertex_frame, width=8)
            entry_y.pack(side=tk.LEFT, padx=2)
            tk.Label(vertex_frame, text="Y", 
                    bg=parent.back_ground, fg="white").pack(side=tk.LEFT)
            
            self.entry_fields.append((entry_x, entry_y))

        # Поля для ввода цветов
        tk.Label(control_frame, text="Цвета вершин (RGB 0-1 или 0-255):", 
                bg=parent.back_ground, fg="white").pack(anchor="w", pady=(10, 5))
        
        self.color_fields = []
        for i in range(3):
            color_frame = tk.Frame(control_frame, bg=parent.back_ground)
            color_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(color_frame, text=f"Цвет {i+1}:", 
                    bg=parent.back_ground, fg="white", width=10).pack(side=tk.LEFT)
            entry_r = tk.Entry(color_frame, width=6)
            entry_r.pack(side=tk.LEFT, padx=1)
            entry_g = tk.Entry(color_frame, width=6)
            entry_g.pack(side=tk.LEFT, padx=1)
            entry_b = tk.Entry(color_frame, width=6)
            entry_b.pack(side=tk.LEFT, padx=1)
            
            self.color_fields.append((entry_r, entry_g, entry_b))

        # Кнопка отрисовки
        button_frame = tk.Frame(control_frame, bg=parent.back_ground)
        button_frame.pack(pady=15)
        
        self.gradient_button = tk.Button(
            button_frame,
            text="Показать градиентный треугольник",
            command=self.draw_gradient_triangle,
            bg="#555",
            fg="white",
            width=30,
            height=3
        )
        self.gradient_button.pack()

        # Примеры значений по умолчанию
        self.set_default_values()

    def set_default_values(self):
        """Установка значений по умолчанию для демонстрации"""
        # Координаты треугольника
        default_coords = [("100", "100"), ("300", "100"), ("200", "300")]
        for (entry_x, entry_y), (x, y) in zip(self.entry_fields, default_coords):
            entry_x.insert(0, x)
            entry_y.insert(0, y)
        
        # Цвета вершин
        default_colors = [
            ("1.0", "0.0", "0.0"),    
            ("0.0", "1.0", "0.0"),    
            ("0.0", "0.0", "1.0")     
        ]
        for (entry_r, entry_g, entry_b), (r, g, b) in zip(self.color_fields, default_colors):
            entry_r.insert(0, r)
            entry_g.insert(0, g)
            entry_b.insert(0, b)

    def validate_float_input(self, value):
        """Проверка и преобразование ввода в float"""
        if value.strip() == '':
            return None
        
        value = value.replace(',', '.')
        
        try:
            return float(value)
        except ValueError:
            return None

    def draw_gradient_triangle(self):
        try:
            # Получаем и проверяем координаты
            triangle = []
            for i, (entry_x, entry_y) in enumerate(self.entry_fields):
                x = self.validate_float_input(entry_x.get())
                y = self.validate_float_input(entry_y.get())
                
                if x is None or y is None:
                    messagebox.showerror("Ошибка", f"Пожалуйста, заполните все поля координат для вершины {i+1}")
                    return
                triangle.append([x, y])
            
            triangle = np.array(triangle)

            # Получаем и проверяем цвета
            vertex_colors = []
            for i, (entry_r, entry_g, entry_b) in enumerate(self.color_fields):
                r = self.validate_float_input(entry_r.get())
                g = self.validate_float_input(entry_g.get())
                b = self.validate_float_input(entry_b.get())
                
                if r is None or g is None or b is None:
                    messagebox.showerror("Ошибка", f"Пожалуйста, заполните все поля цветов (R, G, B) для цвета {i+1}")
                    return
                
                # Проверяем диапазон цветов (0-1 или 0-255)
                if r > 1 or g > 1 or b > 1:
                    # Если значения больше 1, нормализуем к диапазону 0-1
                    r, g, b = r/255.0, g/255.0, b/255.0
                
                vertex_colors.append([r, g, b])
            
            vertex_colors = np.array(vertex_colors)

            # СОЗДАЕМ ОТДЕЛЬНОЕ ОКНО ДЛЯ ТРЕУГОЛЬНИКА
            triangle_window = tk.Toplevel(self.root)
            triangle_window.title("Градиентный треугольник")
            triangle_window.geometry("500x550+600+100")
            
            # Создаем фигуру matplotlib
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            
            # Функция для вычисления барицентрических координат
            def barycentric_coords(p, a, b, c):
                v0 = b - a
                v1 = c - a
                v2 = p - a
                d00 = np.dot(v0, v0)
                d01 = np.dot(v0, v1)
                d11 = np.dot(v1, v1)
                d20 = np.dot(v2, v0)
                d21 = np.dot(v2, v1)
                denom = d00 * d11 - d01 * d01
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                return u, v, w

            # Определяем границы треугольника
            min_x, min_y = np.min(triangle, axis=0).astype(int)
            max_x, max_y = np.max(triangle, axis=0).astype(int)

            # Создаем изображение для более быстрой отрисовки
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            image = np.ones((height, width, 3))  # Белый фон

            # Проходим по всем точкам внутри ограничивающего прямоугольника
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    point = np.array([x, y])
                    u, v, w = barycentric_coords(
                        point, triangle[0], triangle[1], triangle[2]
                    )

                    # Если точка внутри треугольника (все барицентрические координаты >= 0)
                    if u >= 0 and v >= 0 and w >= 0:
                        # Интерполируем цвет на основе барицентрических координат
                        color = (
                            u * vertex_colors[0] + 
                            v * vertex_colors[1] + 
                            w * vertex_colors[2]
                        )
                        # Закрашиваем точку в изображении
                        img_x = x - min_x
                        img_y = y - min_y
                        image[img_y, img_x] = color

            # Отображаем изображение
            ax.imshow(image, extent=[min_x, max_x, max_y, min_y], origin='upper')
            ax.set_xlim(min_x - 10, max_x + 10)
            ax.set_ylim(min_y - 10, max_y + 10)
            ax.set_aspect('equal')
            ax.set_title("Градиентный треугольник")
            ax.grid(True, alpha=0.3)
            
            # Добавляем подписи вершин
            for i, (x, y) in enumerate(triangle):
                ax.plot(x, y, 'ro', markersize=8)
                ax.text(x + 5, y + 5, f'V{i+1}', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

            # Встраиваем график в окно Tkinter
            canvas = FigureCanvasTkAgg(fig, master=triangle_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Кнопка закрытия
            close_button = tk.Button(
                triangle_window,
                text="Закрыть",
                command=triangle_window.destroy,
                bg="#555",
                fg="white",
                width=15
            )
            close_button.pack(pady=5)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла непредвиденная ошибка: {str(e)}")