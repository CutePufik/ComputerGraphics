import numpy as np
import matplotlib.pyplot as plt


def cross(o, a, b):
    """Вычисляет векторное произведение (o -> a) x (o -> b)"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Строит выпуклую оболочку методом Эндрю с явным разделением"""
    # Сортируем точки по x, затем по y (сортировка по координатам)
    points = sorted(points, key=lambda p: (p[0], p[1]))

    if len(points) < 3:
        return points

    # Находим самую левую и самую правую точки
    left = points[0]
    right = points[-1]

    # Делим множество на две части: под (lower) и над (upper) прямой
    # cross <= 0: под или на прямой; >= 0: над или на прямой
    lower_points = [p for p in points if cross(left, right, p) <= 0]
    upper_points = [p for p in points if cross(left, right, p) >= 0]

    # Строим нижнюю оболочку (Грэм-скан по координатам: слева направо, храним левые повороты)
    lower = []
    for p in lower_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Строим верхнюю оболочку (Грэм-скан по координатам: слева направо, храним правые повороты)
    upper = []
    for p in upper_points:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)

    # Сливаем оболочки (реверсируем верхнюю для counterclockwise порядка, убираем дубликаты)
    rev_upper = list(reversed(upper))[:-1]  # right ... before left
    return lower[:-1] + rev_upper


np.random.seed(0)
points = np.random.rand(100, 2)


hull = convex_hull(points)

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Точки")
plt.scatter(
    np.array(hull)[:, 0], np.array(hull)[:, 1], color="red", label="Выпуклая оболочка"
)

# Соединяем точки выпуклой оболочки
hull_closed = np.array(hull + [hull[0]])  # Замыкаем оболочку
plt.plot(hull_closed[:, 0], hull_closed[:, 1], color="red")

plt.title("Выпуклая оболочка методом Эндрю")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.show()