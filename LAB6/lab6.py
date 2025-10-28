import tkinter as tk
from tkinter import messagebox
import numpy as np

# ----------------------------
# Утилиты: матрицы 4x4
# ----------------------------
def translation_matrix(tx, ty, tz):
    T = np.eye(4, dtype=float)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def scale_matrix(sx, sy, sz):
    S = np.eye(4, dtype=float)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz
    return S


def rotation_matrix_4x4(axis, angle_degrees):
    """Матрица поворота вокруг произвольной оси (в градусах)."""
    angle = np.radians(angle_degrees)
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=float)
    ux, uy, uz = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    R3 = np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ],
        dtype=float,
    )
    R = np.eye(4, dtype=float)
    R[:3, :3] = R3
    return R


def rotation_x_4(angle_degrees):
    a = np.radians(angle_degrees)
    R = np.eye(4, dtype=float)
    R[1, 1] = np.cos(a)
    R[1, 2] = -np.sin(a)
    R[2, 1] = np.sin(a)
    R[2, 2] = np.cos(a)
    return R


def rotation_y_4(angle_degrees):
    a = np.radians(angle_degrees)
    R = np.eye(4, dtype=float)
    R[0, 0] = np.cos(a)
    R[0, 2] = np.sin(a)
    R[2, 0] = -np.sin(a)
    R[2, 2] = np.cos(a)
    return R


def rotation_z_4(angle_degrees):
    a = np.radians(angle_degrees)
    R = np.eye(4, dtype=float)
    R[0, 0] = np.cos(a)
    R[0, 1] = -np.sin(a)
    R[1, 0] = np.sin(a)
    R[1, 1] = np.cos(a)
    return R


# ----------------------------
# Классы Point и Face
# ----------------------------
class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.coord = np.array([float(x), float(y), float(z)], dtype=float)

    def to_array(self):
        return self.coord.copy()


class Face:
    def __init__(self, vertex_indices: list):
        self.indices = list(vertex_indices)
        self.normal = None

    def compute_normal(self, vertices_array):
        if len(self.indices) < 3:
            self.normal = np.array([0.0, 0.0, 1.0], dtype=float)
            return self.normal
        a = vertices_array[self.indices[0]]
        b = vertices_array[self.indices[1]]
        c = vertices_array[self.indices[2]]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            self.normal = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            self.normal = n / norm
        return self.normal


# ----------------------------
# Полигон/Многогранник
# ----------------------------
class Polyhedron3D:
    def __init__(self, vertices: np.ndarray, edges: list, faces: list = None):
        self.vertices = np.array(vertices, dtype=float)
        self.edges = list(edges)
        self.faces = [Face(f) for f in faces] if faces else []

    def apply_matrix(self, M4x4):
        """Применяем 4x4 матрицу к всем вершинам."""
        hom = np.c_[self.vertices, np.ones(len(self.vertices), dtype=float)]
        transformed = (hom @ M4x4.T)[:, :3]
        self.vertices = transformed

    def project(self, point, matrix):
        """Проецируем точку 3D через 4x4-матрицу (projection @ view @ model). Возвращаем (x,y)."""
        x, y, z = point
        p4 = np.array([x, y, z, 1.0], dtype=float)
        projected = matrix @ p4
        w = projected[3]
        if abs(w) < 1e-9:
            return float(projected[0]), float(projected[1])
        return float(projected[0] / w), float(projected[1] / w)

    def translate(self, tx, ty, tz):
        M = translation_matrix(tx, ty, tz)
        self.apply_matrix(M)

    def scale(self, sx, sy, sz):
        """Масштаб относительно центра многогранника."""
        centroid = np.mean(self.vertices, axis=0)
        T1 = translation_matrix(-centroid[0], -centroid[1], -centroid[2])
        S = scale_matrix(sx, sy, sz)
        T2 = translation_matrix(centroid[0], centroid[1], centroid[2])
        M = T2 @ S @ T1
        self.apply_matrix(M)

    def rotate(self, angle_degrees, axis, about_center=True):
        """Вращение вокруг оси axis (вектора) на angle_degrees (input в градусах)."""
        R = rotation_matrix_4x4(axis, angle_degrees)
        if about_center:
            centroid = np.mean(self.vertices, axis=0)
            M = translation_matrix(centroid[0], centroid[1], centroid[2]) @ R @ translation_matrix(-centroid[0], -centroid[1], -centroid[2])
        else:
            M = R
        self.apply_matrix(M)

    # ИСПРАВЛЕНИЕ 1: Корректная матрица отражения
    def reflect(self, plane_normal, point_on_plane):
        """Отражение относительно плоскости, заданной нормалью и точкой на плоскости."""
        n = np.array(plane_normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            return
        n = n / norm
        
        # Каноническая матрица отражения
        R = np.eye(4, dtype=float)
        R[:3, :3] = np.eye(3) - 2 * np.outer(n, n)
        
        # Если плоскость не проходит через начало координат
        d = -np.dot(n, point_on_plane)
        if abs(d) > 1e-12:
            T1 = translation_matrix(-point_on_plane[0], -point_on_plane[1], -point_on_plane[2])
            T2 = translation_matrix(point_on_plane[0], point_on_plane[1], point_on_plane[2])
            R = T2 @ R @ T1
        
        self.apply_matrix(R)

    # НОВАЯ ФУНКЦИЯ 1: Отражение относительно координатных плоскостей
    def reflect_xy_plane(self):
        """Отражение относительно плоскости XY"""
        self.reflect([0, 0, 1], [0, 0, 0])

    def reflect_xz_plane(self):
        """Отражение относительно плоскости XZ"""
        self.reflect([0, 1, 0], [0, 0, 0])

    def reflect_yz_plane(self):
        """Отражение относительно плоскости YZ"""
        self.reflect([1, 0, 0], [0, 0, 0])

    # НОВАЯ ФУНКЦИЯ 2: Вращение вокруг оси через центр, параллельной координатной
    def rotate_around_axis_through_center(self, axis_name, angle_degrees):
        """Вращение вокруг оси, параллельной координатной и проходящей через центр"""
        centroid = np.mean(self.vertices, axis=0)
        
        if axis_name.lower() == 'x':
            axis = [1, 0, 0]
        elif axis_name.lower() == 'y':
            axis = [0, 1, 0]
        elif axis_name.lower() == 'z':
            axis = [0, 0, 1]
        else:
            raise ValueError("Axis must be 'x', 'y' or 'z'")
        
        self.rotate(angle_degrees, axis, about_center=True)

    # ИСПРАВЛЕНИЕ 2: Корректное вращение вокруг произвольной прямой по алгоритму из лекции
    def rotate_around_line(self, point1, point2, angle_degrees):
        """Вращение вокруг прямой через point1->point2 по алгоритму из лекции."""
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)
        v = p2 - p1
        
        if np.linalg.norm(v) < 1e-12:
            return
            
        # Нормализуем вектор направления
        l, m, n = v / np.linalg.norm(v)
        angle_rad = np.radians(angle_degrees)
        cos_phi = np.cos(angle_rad)
        sin_phi = np.sin(angle_rad)
        
        # Матрица вращения вокруг произвольной оси (формула Родрига)
        R = np.eye(4, dtype=float)
        R[0, 0] = l*l + cos_phi*(1 - l*l)
        R[0, 1] = l*(1 - cos_phi)*m - n*sin_phi
        R[0, 2] = l*(1 - cos_phi)*n + m*sin_phi
        
        R[1, 0] = l*(1 - cos_phi)*m + n*sin_phi
        R[1, 1] = m*m + cos_phi*(1 - m*m)
        R[1, 2] = m*(1 - cos_phi)*n - l*sin_phi
        
        R[2, 0] = l*(1 - cos_phi)*n - m*sin_phi
        R[2, 1] = m*(1 - cos_phi)*n + l*sin_phi
        R[2, 2] = n*n + cos_phi*(1 - n*n)
        
        # Комбинируем с переносами
        T1 = translation_matrix(-p1[0], -p1[1], -p1[2])
        T2 = translation_matrix(p1[0], p1[1], p1[2])
        M = T2 @ R @ T1
        
        self.apply_matrix(M)


# ----------------------------
# Sphere generator
# ----------------------------
class Sphere:
    def __init__(self, radius, segments, rings):
        self.radius = float(radius)
        self.segments = int(segments)
        self.rings = int(rings)
        self.vertices = []
        self.edges = []
        self.generate_sphere()

    def generate_sphere(self):
        for i in range(self.rings + 1):
            theta = np.pi * i / self.rings
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for j in range(self.segments):
                phi = 2 * np.pi * j / self.segments
                x = self.radius * sin_theta * np.cos(phi)
                y = self.radius * sin_theta * np.sin(phi)
                z = self.radius * cos_theta
                self.vertices.append((x, y, z))
                if j > 0:
                    self.edges.append((i * self.segments + j, i * self.segments + (j - 1)))
                if i > 0:
                    self.edges.append(((i - 1) * self.segments + j, i * self.segments + j))
            self.edges.append((i * self.segments, i * self.segments + self.segments - 1))
        for j in range(self.segments):
            self.edges.append(((self.rings - 1) * self.segments + j, j))


# ----------------------------
# Камера
# ----------------------------
class Camera:
    def __init__(self, position, look_at, up_vector):
        self.position = np.array(position, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.up_vector = np.array(up_vector, dtype=float)

    def get_view_matrix(self):
        z_axis = self.position - self.look_at
        if np.linalg.norm(z_axis) < 1e-12:
            z_axis = np.array([0.0, 0.0, 1.0])
        else:
            z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(self.up_vector, z_axis)
        if np.linalg.norm(x_axis) < 1e-12:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.eye(4, dtype=float)
        rotation[0, :3] = x_axis
        rotation[1, :3] = y_axis
        rotation[2, :3] = z_axis
        translation = np.eye(4, dtype=float)
        translation[:3, 3] = -self.position
        view = rotation @ translation
        return view


# ----------------------------
# Проекционные матрицы
# ----------------------------
def get_perspective_projection_matrix(fov=90, aspect_ratio=1.0, near=0.1, far=1000.0):
    fov_rad = np.radians(fov)
    tan_half_fov = np.tan(fov_rad / 2)
    if abs(tan_half_fov) < 1e-9:
        tan_half_fov = 1e-9
    proj = np.zeros((4, 4), dtype=float)
    proj[0, 0] = 1.0 / (aspect_ratio * tan_half_fov)
    proj[1, 1] = 1.0 / (tan_half_fov)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj


def get_orthographic_projection_matrix(left=-1, right=1, bottom=-1, top=1, near=0.1, far=1000.0):
    proj = np.eye(4, dtype=float)
    proj[0, 0] = 2.0 / (right - left)
    proj[1, 1] = 2.0 / (top - bottom)
    proj[2, 2] = -2.0 / (far - near)
    proj[0, 3] = -(right + left) / (right - left)
    proj[1, 3] = -(top + bottom) / (top - bottom)
    proj[2, 3] = -(far + near) / (far - near)
    return proj


def get_projection_matrix(projection_type):
    if projection_type == "perspective":
        return get_perspective_projection_matrix()
    else:
        return get_orthographic_projection_matrix()


# ----------------------------
# Генерация многогранников
# ----------------------------
def get_polyhedron(name):
    name = name.lower()
    if name == "tetrahedron":
        vertices = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=float)
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    elif name == "cube":
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
            ],
            dtype=float,
        )
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ]
    elif name == "octahedron":
        vertices = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float)
        edges = [
            (0,2),(0,3),(0,4),(0,5),
            (1,2),(1,3),(1,4),(1,5),
            (2,4),(2,5),(3,4),(3,5)
        ]
        faces = [
            [0,2,4],[0,3,4],[1,2,4],[1,3,4],
            [0,2,5],[0,3,5],[1,2,5],[1,3,5]
        ]
    elif name == "icosahedron":
        phi = (1 + np.sqrt(5)) / 2
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
            ],
            dtype=float,
        )
        edges = [
            (0, 1), (0, 5), (0, 7), (0, 10), (0, 11), (1, 5), (1, 7), (1, 8), (1, 9), (2, 3),
            (2, 4), (2, 6), (2, 10), (2, 11), (3, 4), (3, 6), (3, 8), (3, 9), (4, 5), (4, 9),
            (4, 11), (5, 9), (5, 11), (6, 7), (6, 8), (6, 10), (7, 8), (7, 10), (8, 9), (10, 11)
        ]
        faces = []  
    elif name == "dodecahedron":
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
            ],
            dtype=float,
        )
        edges = [
            (0, 8), (0, 12), (0, 16), (1, 9), (1, 12), (1, 18), (2, 10), (2, 13), (2, 16), (3, 11),
            (3, 13), (3, 18), (4, 8), (4, 14), (4, 17), (5, 9), (5, 14), (5, 19), (6, 10), (6, 15),
            (6, 17), (7, 11), (7, 15), (7, 19), (8, 10), (9, 11), (12, 14), (13, 15), (16, 18), (17, 19)
        ]
        faces = []
    elif name == "sphere":
        sphere = Sphere(radius=3.0, segments=20, rings=20)
        vertices = np.array(sphere.vertices, dtype=float)
        edges = sphere.edges
        faces = []
    else:
        return get_polyhedron("cube")
    return Polyhedron3D(vertices, edges, faces)


# ----------------------------
# Отрисовщик
# ----------------------------
class PolyhedronDrawer:
    def __init__(self, canvas, polyhedron: Polyhedron3D, projection_matrix, view_matrix, pre_rotation=None):
        self.canvas = canvas
        self.polyhedron = polyhedron
        self.projection_matrix = projection_matrix
        self.view_matrix = view_matrix
        self.pre_rotation = pre_rotation if pre_rotation is not None else np.eye(4, dtype=float)
        self.center_x = int(canvas.winfo_reqwidth() / 2)
        self.center_y = int(canvas.winfo_reqheight() / 2)

    def draw(self):
        self.canvas.delete("all")
        combined = self.projection_matrix @ self.view_matrix @ self.pre_rotation

        verts_cam = []
        for v in self.polyhedron.vertices:
            p4 = np.array([v[0], v[1], v[2], 1.0], dtype=float)
            cam_p = self.view_matrix @ self.pre_rotation @ p4
            verts_cam.append(cam_p[:3])
        verts_cam = np.array(verts_cam)

        edges_sorted = []
        for idx, (s, e) in enumerate(self.polyhedron.edges):
            z1 = verts_cam[s, 2]
            z2 = verts_cam[e, 2]
            avg_z = (z1 + z2) / 2.0
            edges_sorted.append((avg_z, s, e))
        edges_sorted.sort(key=lambda x: x[0], reverse=False)

        scale = 100.0
        for _, s, e in edges_sorted:
            v1 = self.polyhedron.vertices[s]
            v2 = self.polyhedron.vertices[e]
            x1, y1 = self.polyhedron.project(v1, combined)
            x2, y2 = self.polyhedron.project(v2, combined)
            self.canvas.create_line(
                self.center_x + x1 * scale,
                self.center_y - y1 * scale,
                self.center_x + x2 * scale,
                self.center_y - y2 * scale,
            )


# ----------------------------
# Главное окно и GUI
# ----------------------------
class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lab6 - Аффинные преобразования и проецирование (ИСПРАВЛЕННЫЙ)")
        self.canvas = None
        self.camera = None
        self.drawer = None

        self.view_matrix = None
        self.polyhedron = None
        self.projection_matrix = None

        self.camera_position = [5.0, 5.0, 6.0]
        self.camera_look_at = [0.0, 0.0, 0.0]
        self.camera_up_vector = [0.0, 0.0, 1.0]

        self.polyhedron_name = "cube"
        self.polyhedrons_names = ["cube", "tetrahedron", "octahedron", "icosahedron", "dodecahedron", "sphere"]

        self.projection_type = "perspective"
        self.projection_types = ["perspective", "orthographic", "isometric"]

        row_count = 0
        entry_width = 6

        # Camera position controls
        tk.Label(self.root, text="camera position:").grid(row=row_count, column=0)
        tk.Button(self.root, text="x+", command=lambda: self.change_camera_pos("x+")).grid(row=row_count, column=1)
        tk.Button(self.root, text="x-", command=lambda: self.change_camera_pos("x-")).grid(row=row_count, column=2)
        tk.Button(self.root, text="y+", command=lambda: self.change_camera_pos("y+")).grid(row=row_count, column=3)
        tk.Button(self.root, text="y-", command=lambda: self.change_camera_pos("y-")).grid(row=row_count, column=4)
        tk.Button(self.root, text="z+", command=lambda: self.change_camera_pos("z+")).grid(row=row_count, column=5)
        tk.Button(self.root, text="z-", command=lambda: self.change_camera_pos("z-")).grid(row=row_count, column=6)

        # Camera look_at
        row_count += 1
        tk.Label(self.root, text="camera look_at:").grid(row=row_count, column=0)
        tk.Button(self.root, text="x+", command=lambda: self.change_look_at("x+")).grid(row=row_count, column=1)
        tk.Button(self.root, text="x-", command=lambda: self.change_look_at("x-")).grid(row=row_count, column=2)
        tk.Button(self.root, text="y+", command=lambda: self.change_look_at("y+")).grid(row=row_count, column=3)
        tk.Button(self.root, text="y-", command=lambda: self.change_look_at("y-")).grid(row=row_count, column=4)
        tk.Button(self.root, text="z+", command=lambda: self.change_look_at("z+")).grid(row=row_count, column=5)
        tk.Button(self.root, text="z-", command=lambda: self.change_look_at("z-")).grid(row=row_count, column=6)

        # Translation
        row_count += 1
        tk.Label(self.root, text="translation:").grid(row=row_count, column=0)
        tk.Button(self.root, text="x+", command=lambda: self.translate_polyhedron(1, 0, 0)).grid(row=row_count, column=1)
        tk.Button(self.root, text="x-", command=lambda: self.translate_polyhedron(-1, 0, 0)).grid(row=row_count, column=2)
        tk.Button(self.root, text="y+", command=lambda: self.translate_polyhedron(0, 1, 0)).grid(row=row_count, column=3)
        tk.Button(self.root, text="y-", command=lambda: self.translate_polyhedron(0, -1, 0)).grid(row=row_count, column=4)
        tk.Button(self.root, text="z+", command=lambda: self.translate_polyhedron(0, 0, 1)).grid(row=row_count, column=5)
        tk.Button(self.root, text="z-", command=lambda: self.translate_polyhedron(0, 0, -1)).grid(row=row_count, column=6)

        # Rotation (around axis)
        row_count += 1
        tk.Label(self.root, text="Rotation (deg), axis:").grid(row=row_count, column=0)
        self.angle_entry = tk.Entry(self.root, width=entry_width)
        self.angle_entry.grid(row=row_count, column=1)
        self.angle_entry.insert(0, "15")
        self.rotation_x_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_x_entry.grid(row=row_count, column=2)
        self.rotation_x_entry.insert(0, "1")
        self.rotation_y_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_y_entry.grid(row=row_count, column=3)
        self.rotation_y_entry.insert(0, "0")
        self.rotation_z_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_z_entry.grid(row=row_count, column=4)
        self.rotation_z_entry.insert(0, "0")
        tk.Button(self.root, text="Rotate", command=self.rotate_polyhedron).grid(row=row_count, column=5, columnspan=2)

        # НОВЫЙ ЭЛЕМЕНТ 1: Вращение вокруг осей через центр
        row_count += 1
        tk.Label(self.root, text="Rotate around axis through center:").grid(row=row_count, column=0, columnspan=2)
        tk.Button(self.root, text="X-axis", command=lambda: self.rotate_around_axis("x")).grid(row=row_count, column=2)
        tk.Button(self.root, text="Y-axis", command=lambda: self.rotate_around_axis("y")).grid(row=row_count, column=3)
        tk.Button(self.root, text="Z-axis", command=lambda: self.rotate_around_axis("z")).grid(row=row_count, column=4)

        # Scale
        row_count += 1
        tk.Label(self.root, text="scale (multiplier step 0.3):").grid(row=row_count, column=0)
        tk.Button(self.root, text="x+", command=lambda: self.scale_polyhedron(1, 0, 0)).grid(row=row_count, column=1)
        tk.Button(self.root, text="x-", command=lambda: self.scale_polyhedron(-1, 0, 0)).grid(row=row_count, column=2)
        tk.Button(self.root, text="y+", command=lambda: self.scale_polyhedron(0, 1, 0)).grid(row=row_count, column=3)
        tk.Button(self.root, text="y-", command=lambda: self.scale_polyhedron(0, -1, 0)).grid(row=row_count, column=4)
        tk.Button(self.root, text="z+", command=lambda: self.scale_polyhedron(0, 0, 1)).grid(row=row_count, column=5)
        tk.Button(self.root, text="z-", command=lambda: self.scale_polyhedron(0, 0, -1)).grid(row=row_count, column=6)

        # НОВЫЙ ЭЛЕМЕНТ 2: Отражение относительно координатных плоскостей
        row_count += 1
        tk.Label(self.root, text="Reflection (coordinate planes):").grid(row=row_count, column=0, columnspan=2)
        tk.Button(self.root, text="XY Plane", command=lambda: self.reflect_coordinate_plane("xy")).grid(row=row_count, column=2)
        tk.Button(self.root, text="XZ Plane", command=lambda: self.reflect_coordinate_plane("xz")).grid(row=row_count, column=3)
        tk.Button(self.root, text="YZ Plane", command=lambda: self.reflect_coordinate_plane("yz")).grid(row=row_count, column=4)

        # Reflection (general)
        row_count += 1
        tk.Label(self.root, text="Reflection plane: normal (x,y,z) and point on plane").grid(row=row_count, column=0, columnspan=4)
        tk.Label(self.root, text="Normal:").grid(row=row_count+1, column=0)
        self.reflection_normal_x_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_normal_x_entry.grid(row=row_count+1, column=1)
        self.reflection_normal_x_entry.insert(0, "1")
        self.reflection_normal_y_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_normal_y_entry.grid(row=row_count+1, column=2)
        self.reflection_normal_y_entry.insert(0, "0")
        self.reflection_normal_z_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_normal_z_entry.grid(row=row_count+1, column=3)
        self.reflection_normal_z_entry.insert(0, "0")
        tk.Label(self.root, text="Point:").grid(row=row_count+1, column=4)
        self.reflection_point_x_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_point_x_entry.grid(row=row_count+1, column=5)
        self.reflection_point_x_entry.insert(0, "0")
        self.reflection_point_y_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_point_y_entry.grid(row=row_count+1, column=6)
        self.reflection_point_y_entry.insert(0, "0")
        self.reflection_point_z_entry = tk.Entry(self.root, width=entry_width)
        self.reflection_point_z_entry.grid(row=row_count+1, column=7)
        self.reflection_point_z_entry.insert(0, "0")
        tk.Button(self.root, text="Reflect", command=self.reflect_polyhedron).grid(row=row_count+1, column=8)

        row_count += 2

        # Rotation around line
        tk.Label(self.root, text="Rotate around line (Point1, Point2) angle (deg):").grid(row=row_count, column=0, columnspan=3)
        tk.Label(self.root, text="P1:").grid(row=row_count+1, column=0)
        self.rotation_line_point1_x_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point1_x_entry.grid(row=row_count+1, column=1)
        self.rotation_line_point1_x_entry.insert(0, "0")
        self.rotation_line_point1_y_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point1_y_entry.grid(row=row_count+1, column=2)
        self.rotation_line_point1_y_entry.insert(0, "0")
        self.rotation_line_point1_z_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point1_z_entry.grid(row=row_count+1, column=3)
        self.rotation_line_point1_z_entry.insert(0, "0")
        tk.Label(self.root, text="P2:").grid(row=row_count+1, column=4)
        self.rotation_line_point2_x_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point2_x_entry.grid(row=row_count+1, column=5)
        self.rotation_line_point2_x_entry.insert(0, "1")
        self.rotation_line_point2_y_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point2_y_entry.grid(row=row_count+1, column=6)
        self.rotation_line_point2_y_entry.insert(0, "0")
        self.rotation_line_point2_z_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_line_point2_z_entry.grid(row=row_count+1, column=7)
        self.rotation_line_point2_z_entry.insert(0, "0")
        tk.Label(self.root, text="Angle:").grid(row=row_count+1, column=8)
        self.rotation_angle_entry = tk.Entry(self.root, width=entry_width)
        self.rotation_angle_entry.grid(row=row_count+1, column=9)
        self.rotation_angle_entry.insert(0, "30")
        tk.Button(self.root, text="Rotate Line", command=self.rotate_around_line_polyhedron).grid(row=row_count+1, column=10)

        row_count += 2

        # Polyhedron and projection selection
        self.selected_polyhedron_name = tk.StringVar()
        self.selected_polyhedron_name.set(self.polyhedrons_names[0])
        self.polyhedrons_dropdown = tk.OptionMenu(self.root, self.selected_polyhedron_name, *self.polyhedrons_names, command=self.select_polyhedron)
        self.polyhedrons_dropdown.grid(row=row_count, column=0, columnspan=3)

        self.selected_projection = tk.StringVar()
        self.selected_projection.set(self.projection_types[0])
        self.projection_dropdown = tk.OptionMenu(self.root, self.selected_projection, *self.projection_types, command=self.select_projection)
        self.projection_dropdown.grid(row=row_count, column=4, columnspan=3)

        row_count += 1

        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.grid(row=row_count, column=0, columnspan=11)

        self.root.bind("<MouseWheel>", self.on_mouse_wheel)

    # НОВЫЙ МЕТОД 1: Вращение вокруг оси через центр
    def rotate_around_axis(self, axis_name):
        """Вращение вокруг оси через центр, параллельной координатной"""
        try:
            angle = float(self.angle_entry.get())
        except ValueError:
            messagebox.showerror("Input error", "Неверный ввод угла")
            return
        self.polyhedron.rotate_around_axis_through_center(axis_name, angle)
        self.redraw()

    # НОВЫЙ МЕТОД 2: Отражение относительно координатных плоскостей
    def reflect_coordinate_plane(self, plane_name):
        """Отражение относительно координатной плоскости"""
        if plane_name.lower() == "xy":
            self.polyhedron.reflect_xy_plane()
        elif plane_name.lower() == "xz":
            self.polyhedron.reflect_xz_plane()
        elif plane_name.lower() == "yz":
            self.polyhedron.reflect_yz_plane()
        self.redraw()

    def select_polyhedron(self, selected_value):
        self.polyhedron_name = selected_value
        self.polyhedron = get_polyhedron(self.polyhedron_name)
        self.redraw()

    def select_projection(self, selected_value):
        self.projection_type = selected_value
        self.redraw()

    def on_mouse_wheel(self, event):
        delta = event.delta / 120.0
        self.camera_position[2] -= delta
        if self.camera_position[2] < 0.5:
            self.camera_position[2] = 0.5
        self.redraw()

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

    def rotate_polyhedron(self):
        try:
            angle = float(self.angle_entry.get())
            axis_x = float(self.rotation_x_entry.get())
            axis_y = float(self.rotation_y_entry.get())
            axis_z = float(self.rotation_z_entry.get())
        except ValueError:
            messagebox.showerror("Input error", "Неверный ввод угла или координат оси")
            return
        self.polyhedron.rotate(angle, np.array([axis_x, axis_y, axis_z]), about_center=True)
        self.redraw()

    def scale_polyhedron(self, x, y, z):
        sx = 1.0 + x * 0.3
        sy = 1.0 + y * 0.3
        sz = 1.0 + z * 0.3
        self.polyhedron.scale(sx, sy, sz)
        self.redraw()

    def reflect_polyhedron(self):
        try:
            normal_x = float(self.reflection_normal_x_entry.get())
            normal_y = float(self.reflection_normal_y_entry.get())
            normal_z = float(self.reflection_normal_z_entry.get())
            point_x = float(self.reflection_point_x_entry.get())
            point_y = float(self.reflection_point_y_entry.get())
            point_z = float(self.reflection_point_z_entry.get())
        except ValueError:
            messagebox.showerror("Input error", "Неверный ввод для отражения")
            return
        normal = np.array([normal_x, normal_y, normal_z], dtype=float)
        point_on_plane = np.array([point_x, point_y, point_z], dtype=float)
        self.polyhedron.reflect(normal, point_on_plane)
        self.redraw()

    def rotate_around_line_polyhedron(self):
        try:
            p1x = float(self.rotation_line_point1_x_entry.get())
            p1y = float(self.rotation_line_point1_y_entry.get())
            p1z = float(self.rotation_line_point1_z_entry.get())
            p2x = float(self.rotation_line_point2_x_entry.get())
            p2y = float(self.rotation_line_point2_y_entry.get())
            p2z = float(self.rotation_line_point2_z_entry.get())
            angle = float(self.rotation_angle_entry.get())
        except ValueError:
            messagebox.showerror("Input error", "Неверный ввод для вращения вокруг линии")
            return
        p1 = np.array([p1x, p1y, p1z], dtype=float)
        p2 = np.array([p2x, p2y, p2z], dtype=float)
        self.polyhedron.rotate_around_line(p1, p2, angle)
        self.redraw()

    def start(self):
        self.polyhedron = get_polyhedron(self.polyhedron_name)
        self.redraw()
        self.root.mainloop()

    def redraw(self):
        self.canvas.delete("all")
        self.camera = Camera(position=self.camera_position, look_at=self.camera_look_at, up_vector=self.camera_up_vector)
        base_view = self.camera.get_view_matrix()
        proj = get_projection_matrix(self.projection_type)

        pre_rot = np.eye(4, dtype=float)
        if self.projection_type == "isometric":
            pre_rot = rotation_x_4(-35.264389682754654) @ rotation_z_4(45.0)

        self.view_matrix = base_view
        self.projection_matrix = proj
        self.drawer = PolyhedronDrawer(self.canvas, self.polyhedron, self.projection_matrix, self.view_matrix, pre_rotation=pre_rot)
        self.drawer.draw()


if __name__ == "__main__":
    root = tk.Tk()
    main_window = MainWindow(root=root)
    main_window.start()