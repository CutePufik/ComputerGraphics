import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk
from tkinter import simpledialog, messagebox

class Point:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z, 1])

    def transform(self, matrix):
        self.coords = np.dot(matrix, self.coords)

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]

    def to_tuple(self):
        return (self.x, self.y, self.z)

class Polygon:
    def __init__(self, points):
        self.points = points

class Polyhedron:
    def __init__(self, faces):
        self.faces = faces

    @staticmethod
    def create_tetrahedron():
        #points = []
        # Пример координат вершин тетраэдра
        points = [Point(1, 1, 1), Point(-1, -1, 1), Point(-1, 1, -1), Point(1, -1, -1)]
        #print("Введите координаты 4 вершин тетраэдра (x, y, z):")
        #for i in range(4):
            #x, y, z = map(float, input(f"Введите координаты вершины {i + 1} (через пробел): ").split())
            #points.append(Point(x, y, z))

        faces = [
            Polygon([points[0], points[1], points[2]]),
            Polygon([points[0], points[1], points[3]]),
            Polygon([points[0], points[2], points[3]]),
            Polygon([points[1], points[2], points[3]])
        ]
        return Polyhedron(faces)

    @staticmethod
    def create_hexahedron():
        #points = []
        # Пример координат вершин гексаэдра со стороной 2, центрированного в начале координат
        points = [Point(-1, -1, -1), Point(1, -1, -1), Point(1, 1, -1), Point(-1, 1, -1), Point(-1, -1, 1), Point(1, -1, 1), Point(1, 1, 1), Point(-1, 1, 1)]

        #print("Введите координаты 8 вершин гексаэдра (x, y, z):")
        #for i in range(8):
            #x, y, z = map(float, input(f"Введите координаты вершины {i + 1} (через пробел): ").split())
            #points.append(Point(x, y, z))

        faces = [
            Polygon([points[0], points[1], points[2], points[3]]),
            Polygon([points[4], points[5], points[6], points[7]]),
            Polygon([points[0], points[1], points[5], points[4]]),
            Polygon([points[2], points[3], points[7], points[6]]),
            Polygon([points[1], points[2], points[6], points[5]]),
            Polygon([points[0], points[3], points[7], points[4]])
        ]
        return Polyhedron(faces)

    @staticmethod
    def create_octahedron():
        # Октаэдр с вершинами на координатах ±1 вдоль осей
        points = [
            Point(1, 0, 0), Point(-1, 0, 0), Point(0, 1, 0), Point(0, -1, 0),
            Point(0, 0, 1), Point(0, 0, -1)
        ]
        #points = []
        #print("Введите координаты 6 вершин октаэдра (x, y, z):")
        #for i in range(6):
            #x, y, z = map(float, input(f"Введите координаты вершины {i + 1} (через пробел): ").split())
            #points.append(Point(x, y, z))

        faces = [
            Polygon([points[0], points[2], points[4]]),
            Polygon([points[0], points[2], points[5]]),
            Polygon([points[0], points[3], points[4]]),
            Polygon([points[0], points[3], points[5]]),
            Polygon([points[1], points[2], points[4]]),
            Polygon([points[1], points[2], points[5]]),
            Polygon([points[1], points[3], points[4]]),
            Polygon([points[1], points[3], points[5]])
        ]
        return Polyhedron(faces)

    def apply_transformation(self, matrix):
        for face in self.faces:
            for point in face.points:
                point.transform(matrix)

    def apply_projection(self, projection_type):
        if projection_type == "perspective":
            d = float(input("Введите фокусное расстояние для перспективной проекции: "))
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -1/d],
                [0, 0, 0, 1]
            ])
        elif projection_type == "axonometric":
            matrix = np.array([
                [np.sqrt(3)/2, 0, -np.sqrt(3)/2, 0],
                [0.5, 1, 0.5, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1]
            ])
        else:
            print("Неверный тип проекции.")
            return
        self.apply_transformation(matrix)

    def translate(self, dx, dy, dz):
        matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ])
        self.apply_transformation(matrix)

    def scale(self, sx, sy, sz):
        matrix = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])
        self.apply_transformation(matrix)

    def scale_about_center(self, sx, sy, sz):
        # Вычисляем центр многогранника
        center_x = np.mean([point.x for face in self.faces for point in face.points])
        center_y = np.mean([point.y for face in self.faces for point in face.points])
        center_z = np.mean([point.z for face in self.faces for point in face.points])

        # Перемещаем многогранник к началу координат
        self.translate(-center_x, -center_y, -center_z)

        # Масштабируем многогранник
        self.scale(sx, sy, sz)

        # Перемещаем многогранник обратно к исходному центру
        self.translate(center_x, center_y, center_z)

    def rotate_about_center_axis(self, axis, angle):
        # Вычисляем центр многогранника
        center_x = np.mean([point.x for face in self.faces for point in face.points])
        center_y = np.mean([point.y for face in self.faces for point in face.points])
        center_z = np.mean([point.z for face in self.faces for point in face.points])

        # Перемещаем многогранник так, чтобы центр был в начале координат
        self.translate(-center_x, -center_y, -center_z)

        # Вращаем многогранник вокруг выбранной оси
        if axis == "x":
            self.rotate_x(angle)
        elif axis == "y":
            self.rotate_y(angle)
        elif axis == "z":
            self.rotate_z(angle)
        else:
            print("Неверная ось для вращения. Доступные оси: 'x', 'y', 'z'")
            return

        # Перемещаем многогранник обратно в исходное положение
        self.translate(center_x, center_y, center_z)

    def rotate_x(self, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        matrix = np.array([
            [1, 0, 0, 0],
            [0, cos_a, -sin_a, 0],
            [0, sin_a, cos_a, 0],
            [0, 0, 0, 1]
        ])
        self.apply_transformation(matrix)

    def rotate_y(self, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        matrix = np.array([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ])
        self.apply_transformation(matrix)

    def rotate_z(self, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        matrix = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.apply_transformation(matrix)

    def reflect(self, plane):
        if plane == "xy":
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
        elif plane == "yz":
            matrix = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif plane == "zx":
            matrix = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            print("Неверная плоскость для отражения. Доступные плоскости: 'xy', 'yz', 'zx'")
            return
        self.apply_transformation(matrix)

    def rotate_about_axis(self, p1, p2, angle):
        # Вычисляем вектор направления оси вращения
        direction = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        direction = direction / np.linalg.norm(direction)
        ux, uy, uz = direction

        # Перемещение так, чтобы точка p1 была в начале координат
        self.translate(-p1.x, -p1.y, -p1.z)

        # Матрица вращения вокруг произвольной оси
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a + ux**2 * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a, 0],
            [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy**2 * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a, 0],
            [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz**2 * (1 - cos_a), 0],
            [0, 0, 0, 1]
        ])

        # Применение матрицы вращения
        self.apply_transformation(R)

        # Возврат многогранника в исходное положение
        self.translate(p1.x, p1.y, p1.z)

def plot_polyhedron(polyhedron):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for face in polyhedron.faces:
        vertices = [point.to_tuple() for point in face.points]
        poly = Poly3DCollection([vertices], alpha=.25, edgecolor='k')
        ax.add_collection3d(poly)

    # Установка пределов для осей
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Polyhedron Transformation")
        self.geometry("400x600")

        self.polyhedron = None
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Выберите многогранник").pack(pady=10)
        tk.Button(self, text="Тетраэдр", command=self.create_tetrahedron).pack()
        tk.Button(self, text="Гексаэдр", command=self.create_hexahedron).pack()
        tk.Button(self, text="Октаэдр", command=self.create_octahedron).pack()

        tk.Label(self, text="Применить проекцию").pack(pady=10)
        tk.Button(self, text="Перспективная", command=lambda: self.apply_projection("perspective")).pack()
        tk.Button(self, text="Аксонометрическая", command=lambda: self.apply_projection("axonometric")).pack()

        tk.Label(self, text="Применить трансформации").pack(pady=10)
        tk.Button(self, text="Смещение", command=self.translate_polyhedron).pack()
        tk.Button(self, text="Масштабирование", command=self.scale_polyhedron).pack()
        tk.Button(self, text="Масштабирование относительно центра", command=self.scale_about_center).pack()
        tk.Button(self, text="Вращение вокруг оси центра", command=self.rotate_about_center_axis).pack()
        tk.Button(self, text="Поворот по осям", command=self.rotate_axes).pack()
        tk.Button(self, text="Поворот вокруг произвольной оси", command=self.rotate_about_axis).pack()
        tk.Button(self, text="Отражение", command=self.reflect_polyhedron).pack()
        tk.Button(self, text="Показать", command=self.show_polyhedron).pack(pady=10)

    def create_tetrahedron(self):
        self.polyhedron = Polyhedron.create_tetrahedron()
        messagebox.showinfo("Успех", "Тетраэдр создан")

    def create_hexahedron(self):
        self.polyhedron = Polyhedron.create_hexahedron()
        messagebox.showinfo("Успех", "Гексаэдр создан")

    def create_octahedron(self):
        self.polyhedron = Polyhedron.create_octahedron()
        messagebox.showinfo("Успех", "Октаэдр создан")

    def apply_projection(self, projection_type):
        if self.polyhedron:
            self.polyhedron.apply_projection(projection_type)
        else:
            messagebox.showerror("Ошибка", "Сначала создайте многогранник")

    def translate_polyhedron(self):
        if self.polyhedron:
            dx = simpledialog.askfloat("Смещение по X", "Введите смещение по X:")
            dy = simpledialog.askfloat("Смещение по Y", "Введите смещение по Y:")
            dz = simpledialog.askfloat("Смещение по Z", "Введите смещение по Z:")
            self.polyhedron.translate(dx, dy, dz)

    def scale_polyhedron(self):
        if self.polyhedron:
            sx = simpledialog.askfloat("Масштаб по X", "Введите масштаб по X:")
            sy = simpledialog.askfloat("Масштаб по Y", "Введите масштаб по Y:")
            sz = simpledialog.askfloat("Масштаб по Z", "Введите масштаб по Z:")
            self.polyhedron.scale(sx, sy, sz)

    def scale_about_center(self):
        if self.polyhedron:
            sx = simpledialog.askfloat("Масштаб по X", "Введите масштаб по X:")
            sy = simpledialog.askfloat("Масштаб по Y", "Введите масштаб по Y:")
            sz = simpledialog.askfloat("Масштаб по Z", "Введите масштаб по Z:")
            self.polyhedron.scale_about_center(sx, sy, sz)

    def rotate_about_center_axis(self):
        if self.polyhedron:
            axis = simpledialog.askstring("Ось", "Введите ось (x, y, z):").lower()
            angle = simpledialog.askfloat("Угол", "Введите угол поворота в радианах:")
            self.polyhedron.rotate_about_center_axis(axis, angle)

    def rotate_axes(self):
        if self.polyhedron:
            angle_x = simpledialog.askfloat("Угол вокруг X", "Введите угол вокруг X в радианах:")
            angle_y = simpledialog.askfloat("Угол вокруг Y", "Введите угол вокруг Y в радианах:")
            angle_z = simpledialog.askfloat("Угол вокруг Z", "Введите угол вокруг Z в радианах:")
            self.polyhedron.rotate_x(angle_x)
            self.polyhedron.rotate_y(angle_y)
            self.polyhedron.rotate_z(angle_z)

    def rotate_about_axis(self):
        if self.polyhedron:
            # Запрос координат первой точки
            x1 = simpledialog.askfloat("Точка 1", "Введите X координату первой точки:")
            y1 = simpledialog.askfloat("Точка 1", "Введите Y координату первой точки:")
            z1 = simpledialog.askfloat("Точка 1", "Введите Z координату первой точки:")

            # Запрос координат второй точки
            x2 = simpledialog.askfloat("Точка 2", "Введите X координату второй точки:")
            y2 = simpledialog.askfloat("Точка 2", "Введите Y координату второй точки:")
            z2 = simpledialog.askfloat("Точка 2", "Введите Z координату второй точки:")

            # Запрос угла поворота
            angle = simpledialog.askfloat("Угол поворота", "Введите угол поворота в радианах:")

            # Создание точек для оси вращения и применение поворота
            p1, p2 = Point(x1, y1, z1), Point(x2, y2, z2)
            self.polyhedron.rotate_about_axis(p1, p2, angle)

    def reflect_polyhedron(self):
        if self.polyhedron:
            plane = simpledialog.askstring("Плоскость", "Введите плоскость (xy, yz, zx):").lower()
            self.polyhedron.reflect(plane)

    def show_polyhedron(self):
        if self.polyhedron:
            plot_polyhedron(self.polyhedron)

app = Application()
app.mainloop()