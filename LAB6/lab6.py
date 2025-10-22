import tkinter as tk
import numpy as np


class Sphere:
    def __init__(self, radius, segments, rings):
        self.radius = radius
        self.segments = segments
        self.rings = rings
        self.vertices = []
        self.edges = []
        self.generate_sphere()

    def generate_sphere(self):
        """Генерация вершин и рёбер для представления поверхности сферы."""
        for i in range(self.rings + 1):  # Шаг по широте
            theta = np.pi * i / self.rings  # Угол от 0 до pi (от полюса до полюса)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(self.segments):  # Шаг по долготе
                phi = 2 * np.pi * j / self.segments  # Угол от 0 до 2*pi
                x = self.radius * sin_theta * np.cos(phi)
                y = self.radius * sin_theta * np.sin(phi)
                z = self.radius * cos_theta
                self.vertices.append((x, y, z))

                # Добавление рёбер
                # Соединяем точки вдоль долгот (вертикальные рёбра)
                if j > 0:
                    self.edges.append(
                        (i * self.segments + j, i * self.segments + (j - 1))
                    )
                # Соединяем точки вдоль широт (горизонтальные рёбра)
                if i > 0:
                    self.edges.append(
                        ((i - 1) * self.segments + j, i * self.segments + j)
                    )

            # Замыкание последней точки с первой вдоль долготы
            self.edges.append(
                (i * self.segments, i * self.segments + self.segments - 1)
            )

        # Замыкаем последние кольца
        for j in range(self.segments):
            self.edges.append(((self.rings - 1) * self.segments + j, j))


class Camera:
    """Класс для камеры, которая определяет её положение и ориентацию."""

    def __init__(self, position, look_at, up_vector):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)

    def get_view_matrix(self):
        """Возвращает матрицу вида камеры для преобразования в пространство камеры."""
        z_axis = self.position - self.look_at
        z_axis = z_axis / np.linalg.norm(z_axis)  # Нормализуем вектор

        x_axis = np.cross(self.up_vector, z_axis)  # Перпендикулярно z
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis])

        # Позиция камеры в пространстве
        translation_vector = -np.dot(rotation_matrix, self.position)

        # Формируем итоговую матрицу вида
        return np.vstack(
            [np.column_stack([rotation_matrix, translation_vector]), [0, 0, 0, 1]]
        )


class Polyhedron3D:
    """Класс для многогранника."""

    def __init__(self, vertices, edges):
        self.vertices = vertices  # Координаты вершин
        self.edges = edges  # Пары индексов для рёбер

    def project(self, point, matrix):
        """Проецируем точку через заданную матрицу."""
        x, y, z = point
        # Добавляем однородную координату w = 1 для 4D-вектора
        point_4d = np.array([x, y, z, 1])
        projected = np.dot(matrix, point_4d)
        # Возвращаем 2D-координаты (делим на w, чтобы привести к нормализованным координатам)
        return projected[0] / projected[3], projected[1] / projected[3]

    def translate(self, tx, ty, tz):
        """Смещение многогранника на (tx, ty, tz)."""
        translation_matrix = np.array(
            [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]
        )
        self.vertices = np.dot(
            np.c_[self.vertices, np.ones(len(self.vertices))], translation_matrix.T
        )[:, :3]

    def scale(self, sx, sy, sz):
        """Масштабирование многогранника относительно его центра."""
        centroid = np.mean(self.vertices, axis=0)
        scaling_matrix = np.array(
            [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
        )

        self.translate(-centroid[0], -centroid[1], -centroid[2])
        self.vertices = np.dot(
            np.c_[self.vertices, np.ones(len(self.vertices))], scaling_matrix.T
        )[:, :3]
        self.translate(centroid[0], centroid[1], centroid[2])

    def rotate(self, angle, axis):
        """Поворот многогранника вокруг оси (x, y, z)."""
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = axis

        # Матрица поворота (по оси с заданными углами)
        rotation_matrix = np.array(
            [
                [
                    cos_angle + ux**2 * (1 - cos_angle),
                    ux * uy * (1 - cos_angle) - uz * sin_angle,
                    ux * uz * (1 - cos_angle) + uy * sin_angle,
                ],
                [
                    uy * ux * (1 - cos_angle) + uz * sin_angle,
                    cos_angle + uy**2 * (1 - cos_angle),
                    uy * uz * (1 - cos_angle) - ux * sin_angle,
                ],
                [
                    uz * ux * (1 - cos_angle) - uy * sin_angle,
                    uz * uy * (1 - cos_angle) + ux * sin_angle,
                    cos_angle + uz**2 * (1 - cos_angle),
                ],
            ]
        )
        self.vertices = np.dot(self.vertices, rotation_matrix.T)

    def reflect(self, plane_normal, point_on_plane):
        """Отражение относительно выбранной плоскости."""
        # Создаем матрицу для отражения
        normal = plane_normal / np.linalg.norm(plane_normal)  # Нормализуем нормаль
        d = -np.dot(normal, point_on_plane)

        reflection_matrix = np.array(
            [
                [
                    1 - 2 * normal[0] ** 2,
                    -2 * normal[0] * normal[1],
                    -2 * normal[0] * normal[2],
                    -2 * normal[0] * d,
                ],
                [
                    -2 * normal[1] * normal[0],
                    1 - 2 * normal[1] ** 2,
                    -2 * normal[1] * normal[2],
                    -2 * normal[1] * d,
                ],
                [
                    -2 * normal[2] * normal[0],
                    -2 * normal[2] * normal[1],
                    1 - 2 * normal[2] ** 2,
                    -2 * normal[2] * d,
                ],
                [0, 0, 0, 1],
            ]
        )
        self.vertices = np.dot(
            np.c_[self.vertices, np.ones(len(self.vertices))], reflection_matrix.T
        )[:, :3]

    def rotate_around_line(self, point1, point2, angle):
        """Поворот многогранника вокруг прямой, заданной двумя точками."""
        # Вектор направления оси вращения
        direction = point2 - point1
        axis = direction / np.linalg.norm(direction)

        # Переносим точку point1 в начало
        self.translate(-point1[0], -point1[1], -point1[2])

        # Поворот вокруг оси
        self.rotate(angle, axis)

        # Возвращаем многогранник на прежнее место
        self.translate(point1[0], point1[1], point1[2])


def get_polyhedron(name):
    """Возвращает многогранник по имени."""
    if name == "tetrahedron":
        # Тетраэдр
        vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    elif name == "cube":
        # Куб
        vertices = np.array(
            [
                [1, 1, 1],
                [-1, 1, 1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, -1],
                [1, -1, -1],
            ]
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Верхняя грань
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Нижняя грань
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Вертикальные рёбра
        ]
    elif name == "octahedron":
        # Октаэдр
        vertices = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )
        edges = [
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
        ]
    elif name == "icosahedron":
        phi = (1 + np.sqrt(5)) / 2  # Золотое сечение
        vertices = np.array(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ]
        )
        edges = [
            (0, 1),
            (0, 5),
            (0, 7),
            (0, 11),
            (0, 10),
            (1, 5),
            (1, 7),
            (1, 9),
            (1, 8),
            (2, 4),
            (2, 6),
            (2, 11),
            (2, 10),
            (3, 4),
            (3, 6),
            (3, 8),
            (3, 9),
            (4, 5),
            (4, 9),
            (5, 11),
            (6, 7),
            (6, 10),
            (7, 8),
            (8, 9),
            (10, 11),
        ]
    elif name == "dodecahedron":
        # Додекаэдр
        phi = (1 + np.sqrt(5)) / 2
        a, b = 1 / phi, phi
        vertices = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
                [0, -a, -b],
                [0, -a, b],
                [0, a, -b],
                [0, a, b],
                [-a, -b, 0],
                [-a, b, 0],
                [a, -b, 0],
                [a, b, 0],
                [-b, 0, -a],
                [b, 0, -a],
                [-b, 0, a],
                [b, 0, a],
            ]
        )
        edges = [
            (0, 8),
            (0, 10),
            (0, 16),
            (1, 9),
            (1, 11),
            (1, 18),
            (2, 12),
            (2, 13),
            (2, 16),
            (3, 13),
            (3, 15),
            (3, 18),
            (4, 8),
            (4, 14),
            (4, 17),
            (5, 9),
            (5, 14),
            (5, 19),
            (6, 10),
            (6, 15),
            (6, 17),
            (7, 11),
            (7, 15),
            (7, 19),
            (8, 12),
            (9, 13),
            (10, 16),
            (11, 18),
            (12, 17),
            (13, 18),
            (14, 17),
            (15, 19),
            (16, 18),
            (17, 19),
        ]
    elif name == "sphere":
        sphere = Sphere(radius=3.0, segments=20, rings=20)

        vertices = np.array(sphere.vertices)

        edges = sphere.edges
        print(edges)

    return Polyhedron3D(vertices, edges)


def get_perspective_projection_matrix(fov=90, aspect_ratio=1, near=0.1, far=1000):
    """Перспективная проекционная матрица."""
    # Конвертируем угол обзора в радианы
    fov_rad = np.radians(fov)
    tan_half_fov = np.tan(fov_rad / 2)

    # Строим матрицу перспективы
    projection_matrix = np.array(
        [
            [1 / (aspect_ratio * tan_half_fov), 0, 0, 0],
            [0, 1 / tan_half_fov, 0, 0],
            [0, 0, -(far + near) / (far - near), -1],
            [0, 0, -(2 * far * near) / (far - near), 0],
        ]
    )

    return projection_matrix


def get_orthographic_projection_matrix(
    left=-1, right=1, bottom=-1, top=1, near=0.1, far=1000
):
    projection_matrix = np.array(
        [
            [2 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1],
        ]
    )

    return projection_matrix


def get_progection_matrix(projection_type):
    if projection_type == "perspective":
        return get_perspective_projection_matrix()
    else:
        return get_orthographic_projection_matrix()


class PolyhedronDrawer:
    """Класс для рисования многогранника на холсте."""

    def __init__(self, canvas, polyhedron, projection_matrix, view_matrix):
        self.canvas = canvas
        self.polyhedron = polyhedron
        self.projection_matrix = projection_matrix
        self.view_matrix = view_matrix
        self.center_x = 200  # Центр холста по X
        self.center_y = 200  # Центр холста по Y

    def draw(self):
        """Отрисовывает многогранник на холсте с применением проекции и матрицы вида."""
        self.canvas.delete("all")  # Очистить холст
        for start, end in self.polyhedron.edges:
            # Применяем матрицу вида, затем проекцию для каждой вершины
            x1, y1 = self.polyhedron.project(
                self.polyhedron.vertices[start],
                self.projection_matrix @ self.view_matrix,
            )
            x2, y2 = self.polyhedron.project(
                self.polyhedron.vertices[end], self.projection_matrix @ self.view_matrix
            )
            self.canvas.create_line(
                self.center_x + x1 * 100,
                self.center_y - y1 * 100,
                self.center_x + x2 * 100,
                self.center_y - y2 * 100,
            )


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lab6")
        self.canvas = None
        self.camera = None
        self.drawer = None

        self.view_matrix = None
        self.polyhedron = None
        self.projection_matrix = None

        self.camera_position = [5, 5, 6]
        self.camera_look_at = [0, 0, 0]
        self.camera_up_vector = [0, 0, 1]

        self.polyhedron_name = "cube"
        self.polyhedrons_names = [
            "cube",
            "tetrahedron",
            "octahedron",
            "icosahedron",
            "dodecahedron",
            "sphere",
        ]

        self.projection_type = "perspective"
        self.projection_types = ["perspective", "orthographic"]

        ### Camera position ###
        row_count = 0

        self.change_camera_pos_label = tk.Label(self.root, text="camera position:")

        self.change_camera_pos_label.grid(row=row_count, column=0)

        self.x_plus_button = tk.Button(
            self.root, text="   x+   ", command=lambda: self.change_camera_pos("x+")
        )
        self.x_plus_button.grid(row=row_count, column=1)

        self.x_minus_button = tk.Button(
            self.root, text="   x-   ", command=lambda: self.change_camera_pos("x-")
        )
        self.x_minus_button.grid(row=row_count, column=2)

        self.y_plus_button = tk.Button(
            self.root, text="   y+   ", command=lambda: self.change_camera_pos("y+")
        )
        self.y_plus_button.grid(row=row_count, column=3)

        self.y_minus_button = tk.Button(
            self.root, text="   y-   ", command=lambda: self.change_camera_pos("y-")
        )
        self.y_minus_button.grid(row=row_count, column=4)

        self.z_plus_button = tk.Button(
            self.root, text="   z+   ", command=lambda: self.change_camera_pos("z+")
        )
        self.z_plus_button.grid(row=row_count, column=5)

        self.z_minus_button = tk.Button(
            self.root, text="   z-   ", command=lambda: self.change_camera_pos("z-")
        )
        self.z_minus_button.grid(row=row_count, column=6)

        ### Camera Look ###

        row_count += 1

        self.change_look_at_label = tk.Label(self.root, text="camera look_at:")

        self.change_look_at_label.grid(row=row_count, column=0)

        self.look_at_x_plus_button = tk.Button(
            self.root, text="   x+   ", command=lambda: self.change_look_at("x+")
        )
        self.look_at_x_plus_button.grid(row=row_count, column=1)

        self.look_at_x_minus_button = tk.Button(
            self.root, text="   x-   ", command=lambda: self.change_look_at("x-")
        )
        self.look_at_x_minus_button.grid(row=row_count, column=2)

        self.look_at_y_plus_button = tk.Button(
            self.root, text="   y+   ", command=lambda: self.change_look_at("y+")
        )
        self.look_at_y_plus_button.grid(row=row_count, column=3)

        self.look_at_y_minus_button = tk.Button(
            self.root, text="   y-   ", command=lambda: self.change_look_at("y-")
        )
        self.look_at_y_minus_button.grid(row=row_count, column=4)

        self.look_at_z_plus_button = tk.Button(
            self.root, text="   z+   ", command=lambda: self.change_look_at("z+")
        )
        self.look_at_z_plus_button.grid(row=row_count, column=5)

        self.look_at_z_minus_button = tk.Button(
            self.root, text="   z-   ", command=lambda: self.change_look_at("z-")
        )
        self.look_at_z_minus_button.grid(row=row_count, column=6)

        ### Translation ###
        row_count += 1

        self.translation_label = tk.Label(self.root, text="translation:")

        self.translation_label.grid(row=row_count, column=0)

        self.translation_x_plus_button = tk.Button(
            self.root,
            text="   x+   ",
            command=lambda: self.translate_polyhedron(1, 0, 0),
        )
        self.translation_x_plus_button.grid(row=row_count, column=1)

        self.translation_x_minus_button = tk.Button(
            self.root,
            text="   x-   ",
            command=lambda: self.translate_polyhedron(-1, 0, 0),
        )
        self.translation_x_minus_button.grid(row=row_count, column=2)

        self.translation_y_plus_button = tk.Button(
            self.root,
            text="   y+   ",
            command=lambda: self.translate_polyhedron(0, 1, 0),
        )
        self.translation_y_plus_button.grid(row=row_count, column=3)

        self.translation_y_minus_button = tk.Button(
            self.root,
            text="   y-   ",
            command=lambda: self.translate_polyhedron(0, -1, 0),
        )
        self.translation_y_minus_button.grid(row=row_count, column=4)

        self.translation_z_plus_button = tk.Button(
            self.root,
            text="   z+   ",
            command=lambda: self.translate_polyhedron(0, 0, 1),
        )
        self.translation_z_plus_button.grid(row=row_count, column=5)

        self.translation_z_minus_button = tk.Button(
            self.root,
            text="   z-   ",
            command=lambda: self.translate_polyhedron(0, 0, -1),
        )
        self.translation_z_minus_button.grid(row=row_count, column=6)

        ### Rotation ###
        row_count += 1

        entry_width = 5

        self.rotation_label = tk.Label(self.root, text="Rotation:")
        self.rotation_label.grid(row=row_count, column=0)

        self.rotation_angle_label = tk.Label(self.root, text="a, (x,y,z):")
        self.rotation_angle_label.grid(row=row_count, column=1)

        self.angle_entry = tk.Entry(self.root, width=entry_width)
        self.angle_entry.grid(row=row_count, column=2)
        self.angle_entry.insert(0, "2")

        self.rotation_x_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_x_entry.grid(row=row_count, column=3)
        self.rotation_x_entry.insert(0, "1")

        self.rotation_y_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_y_entry.grid(row=row_count, column=4)
        self.rotation_y_entry.insert(0, "0")

        self.rotation_z_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_z_entry.grid(row=row_count, column=5)
        self.rotation_z_entry.insert(0, "0")

        self.rotate_button = tk.Button(
            self.root,
            text="rotate",
            command=self.rotate_polyhedron,
        )
        self.rotate_button.grid(row=row_count, column=6)

        ### Scale ###
        row_count += 1

        self.scale_label = tk.Label(self.root, text="scale:")

        self.scale_label.grid(row=row_count, column=0)

        self.scale_x_plus_button = tk.Button(
            self.root,
            text="   x+   ",
            command=lambda: self.scale_polyhedron(1, 0, 0),
        )
        self.scale_x_plus_button.grid(row=row_count, column=1)

        self.scale_x_minus_button = tk.Button(
            self.root,
            text="   x-   ",
            command=lambda: self.scale_polyhedron(-1, 0, 0),
        )
        self.scale_x_minus_button.grid(row=row_count, column=2)

        self.scale_y_plus_button = tk.Button(
            self.root,
            text="   y+   ",
            command=lambda: self.scale_polyhedron(0, 1, 0),
        )
        self.scale_y_plus_button.grid(row=row_count, column=3)

        self.scale_y_minus_button = tk.Button(
            self.root,
            text="   y-   ",
            command=lambda: self.scale_polyhedron(0, -1, 0),
        )
        self.scale_y_minus_button.grid(row=row_count, column=4)

        self.scale_z_plus_button = tk.Button(
            self.root,
            text="   z+   ",
            command=lambda: self.scale_polyhedron(0, 0, 1),
        )
        self.scale_z_plus_button.grid(row=row_count, column=5)

        self.scale_z_minus_button = tk.Button(
            self.root,
            text="   z-   ",
            command=lambda: self.scale_polyhedron(0, 0, -1),
        )
        self.scale_z_minus_button.grid(row=row_count, column=6)

        ### Reflection ###
        row_count += 1

        self.reflection_label = tk.Label(self.root, text="Reflection:")
        self.reflection_label.grid(row=row_count, column=0, rowspan=2)

        self.reflection_normal_label = tk.Label(self.root, text="Normal:")  # (x,y,z):
        self.reflection_normal_label.grid(row=row_count, column=1)

        # Ввод нормали плоскости для отражения
        self.reflection_normal_x_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_normal_x_entry.grid(row=row_count, column=2)
        self.reflection_normal_x_entry.insert(0, "1")

        self.reflection_normal_y_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_normal_y_entry.grid(row=row_count, column=3)
        self.reflection_normal_y_entry.insert(0, "0")

        self.reflection_normal_z_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_normal_z_entry.grid(row=row_count, column=4)
        self.reflection_normal_z_entry.insert(0, "1")

        self.reflect_button = tk.Button(
            self.root,
            text="Reflect",
            command=self.reflect_polyhedron,
        )
        self.reflect_button.grid(row=row_count, column=5, columnspan=2, rowspan=2)

        row_count += 1

        # Ввод точки на плоскости
        self.reflection_point_label = tk.Label(
            self.root, text="Point:"  #  on plane (x,y,z):
        )
        self.reflection_point_label.grid(row=row_count, column=1)

        self.reflection_point_x_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_point_x_entry.grid(row=row_count, column=2)
        self.reflection_point_x_entry.insert(0, "1")

        self.reflection_point_y_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_point_y_entry.grid(row=row_count, column=3)
        self.reflection_point_y_entry.insert(0, "0")

        self.reflection_point_z_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_point_z_entry.grid(row=row_count, column=4)
        self.reflection_point_z_entry.insert(0, "0")

        ### Rotation around a Line ###
        row_count += 1

        self.rotation_around_line_label = tk.Label(self.root, text="Around line:")
        self.rotation_around_line_label.grid(row=row_count, column=0, rowspan=2)

        self.rotation_line_point1_label = tk.Label(self.root, text="Point 1:")
        self.rotation_line_point1_label.grid(row=row_count, column=1)

        # Ввод первой точки на линии
        self.rotation_line_point1_x_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point1_x_entry.grid(row=row_count, column=2)
        self.rotation_line_point1_x_entry.insert(0, "0")

        self.rotation_line_point1_y_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point1_y_entry.grid(row=row_count, column=3)
        self.rotation_line_point1_y_entry.insert(0, "0")

        self.rotation_line_point1_z_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point1_z_entry.grid(row=row_count, column=4)
        self.rotation_line_point1_z_entry.insert(0, "0")

        self.rotation_angle_label = tk.Label(self.root, text="Angle:")
        self.rotation_angle_label.grid(row=row_count, column=5)

        self.rotation_angle_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_angle_entry.grid(row=row_count, column=6)
        self.rotation_angle_entry.insert(0, "30")

        row_count += 1

        self.rotation_line_point2_label = tk.Label(self.root, text="Point 2:")
        self.rotation_line_point2_label.grid(row=row_count, column=1)

        # Ввод второй точки на линии
        self.rotation_line_point2_x_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point2_x_entry.grid(row=row_count, column=2)
        self.rotation_line_point2_x_entry.insert(0, "1")

        self.rotation_line_point2_y_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point2_y_entry.grid(row=row_count, column=3)
        self.rotation_line_point2_y_entry.insert(0, "0")

        self.rotation_line_point2_z_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point2_z_entry.grid(row=row_count, column=4)
        self.rotation_line_point2_z_entry.insert(0, "0")

        self.rotate_around_line_button = tk.Button(
            self.root,
            text="Rotate",
            command=self.rotate_around_line_polyhedron,
        )
        self.rotate_around_line_button.grid(row=row_count, column=5, columnspan=2)

        ### Chage polyhedron and projection ###
        row_count += 1

        self.selected_polyhedron_name = tk.StringVar()
        self.selected_polyhedron_name.set(self.polyhedrons_names[0])

        self.polyhedrons_dropdown = tk.OptionMenu(
            self.root,
            self.selected_polyhedron_name,
            *self.polyhedrons_names,
            command=self.select_polyhedron,
        )

        self.polyhedrons_dropdown.grid(row=row_count, column=1, columnspan=3)

        self.selected_projection = tk.StringVar()
        self.selected_projection.set(self.projection_types[0])

        self.projection_dropdown = tk.OptionMenu(
            self.root,
            self.selected_projection,
            *self.projection_types,
            command=self.select_projection,
        )
        self.projection_dropdown.grid(row=row_count, column=4, columnspan=3)

        row_count += 1

        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.canvas.grid(row=row_count, column=0, columnspan=7)

        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

    def select_polyhedron(self, selected_value):
        print(f"Вы выбрали: {selected_value}")
        self.polyhedron_name = selected_value
        self.polyhedron = get_polyhedron(self.polyhedron_name)
        self.redraw()

    def select_projection(self, selected_value):
        print(f"Вы выбрали: {selected_value}")
        self.projection_type = selected_value
        self.redraw()

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            print("Прокрутили вверх")
            self.camera_position[0] += 1
            self.camera_position[1] += 1
            self.camera_position[2] += 1
            self.redraw()
        else:
            print("Прокрутили вниз")
            self.camera_position[0] -= 1
            self.camera_position[1] -= 1
            self.camera_position[2] -= 1
            self.redraw()
        return

    def change_camera_pos(self, com):
        if com == "x+":
            self.camera_position[0] += 1
        elif com == "x-":
            self.camera_position[0] -= 1
        elif com == "y+":
            self.camera_position[1] += 1
        elif com == "y-":
            self.camera_position[1] -= 1
        elif com == "z+":
            self.camera_position[2] += 1
        elif com == "z-":
            self.camera_position[2] -= 1
        self.redraw()

    def change_look_at(self, com):
        if com == "x+":
            self.camera_look_at[0] += 1
        elif com == "x-":
            self.camera_look_at[0] -= 1
        elif com == "y+":
            self.camera_look_at[1] += 1
        elif com == "y-":
            self.camera_look_at[1] -= 1
        elif com == "z+":
            self.camera_look_at[2] += 1
        elif com == "z-":
            self.camera_look_at[2] -= 1
        self.redraw()

    def translate_polyhedron(self, x, y, z):
        self.polyhedron.translate(x, y, z)
        self.redraw()
        return

    def rotate_polyhedron(self):

        angle = float(self.angle_entry.get())
        axis_x = float(self.rotation_x_entry.get())
        axis_y = float(self.rotation_y_entry.get())
        axis_z = float(self.rotation_z_entry.get())
        self.polyhedron.rotate(angle, np.array([axis_x, axis_y, axis_z]))
        self.redraw()
        return

    def scale_polyhedron(self, x, y, z):
        self.polyhedron.scale(1 + x * 0.3, 1 + y * 0.3, 1 + z * 0.3)
        self.redraw()
        return

    def reflect_polyhedron(self):
        normal_x = float(self.reflection_normal_x_entry.get())
        normal_y = float(self.reflection_normal_y_entry.get())
        normal_z = float(self.reflection_normal_z_entry.get())

        point_x = float(self.reflection_point_x_entry.get())
        point_y = float(self.reflection_point_y_entry.get())
        point_z = float(self.reflection_point_z_entry.get())

        # Преобразуем нормаль в numpy массив
        normal = np.array([normal_x, normal_y, normal_z])
        point_on_plane = np.array([point_x, point_y, point_z])

        self.polyhedron.reflect(normal, point_on_plane)
        self.redraw()
        return

    def rotate_around_line_polyhedron(self):
        point1_x = float(self.rotation_line_point1_x_entry.get())
        point1_y = float(self.rotation_line_point1_y_entry.get())
        point1_z = float(self.rotation_line_point1_z_entry.get())

        point2_x = float(self.rotation_line_point2_x_entry.get())
        point2_y = float(self.rotation_line_point2_y_entry.get())
        point2_z = float(self.rotation_line_point2_z_entry.get())

        angle = float(self.rotation_angle_entry.get())

        point1 = np.array([point1_x, point1_y, point1_z])
        point2 = np.array([point2_x, point2_y, point2_z])

        self.polyhedron.rotate_around_line(point1, point2, angle)
        self.redraw()
        return

    def start(self):
        self.polyhedron = get_polyhedron(self.polyhedron_name)
        self.redraw()
        self.root.mainloop()

    def redraw(self):
        print(
            f"camera_position: {self.camera_position}, look_at: {self.camera_look_at}"
        )
        self.canvas.delete("all")

        self.camera = Camera(
            position=self.camera_position,
            look_at=self.camera_look_at,
            up_vector=self.camera_up_vector,
        )
        self.view_matrix = self.camera.get_view_matrix()
        self.projection_matrix = get_progection_matrix(self.projection_type)
        self.drawer = PolyhedronDrawer(
            self.canvas, self.polyhedron, self.projection_matrix, self.view_matrix
        )
        self.drawer.draw()
        return


if __name__ == "__main__":
    root = tk.Tk()
    main_window = MainWindow(root=root)
    main_window.start()