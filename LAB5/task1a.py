import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random


default_iterations = 0
default_start_direction = 90
l_systems_index = 3
iterations = [i for i in range(5, 6)]

# Определение L-систем
l_systems = [
    # Снежинка Коха
    {
        "atom": "F++F++F",
        "rules": {"F": "F-F++F-F"},
        "angle": 60,
        "start_direction": 0,
    },
    # Кривая Коха
    {
        "atom": "F",
        "rules": {"F": "F+F-F-F+F"},
        "angle": 90,
        "start_direction": 0,
    },
    # Треугольник Серпинского
    {
        "atom": "FXF--FF--FF",
        "rules": {"F": "FF", "X": "--FXF++FXF++FXF--"},
        "angle": 60,
        "start_direction": 0,
    },
    # Фрактальное растение
    {
        "atom": "X",
        "rules": {"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"},
        "angle": 25,
        "start_direction": 0,
    },
    # Кривая дракона
    {
        "atom": "FX",
        "rules": {"X": "X+YF+", "Y": "-FX-Y"},
        "angle": 90,
        "start_direction": 0,
    },
]


def generate_l_system(axiom, rules, iterations):
    for _ in range(iterations):
        result = "".join(rules.get(char, char) for char in axiom)
        axiom = result
    return axiom


def points_l_system(l_system_index):
    system = l_systems[l_system_index]
    axiom = system["atom"]
    angle = system["angle"]
    direction = system["start_direction"]
    if default_iterations:
        iterations = default_iterations
    else:
        iterations = system["iterations"]

    rules = system["rules"]

    instructions = generate_l_system(axiom, rules, iterations)
    print(instructions)

    x, y = 0, 0
    stack = []
    points = [(x, y)]
    current_angle = np.radians(direction)

    for char in instructions:
        if char == "F":
            # Двигаемся вперед и сохраняем новую точку
            x_new = x + np.sin(current_angle)
            y_new = y + np.cos(current_angle)
            points.append((x_new, y_new))
            x, y = x_new, y_new  # Обновляем текущую позицию
        elif char == "+":
            # Поворачиваем на угол (вправо)
            current_angle -= np.radians(angle)
        elif char == "-":
            # Поворачиваем на угол (влево)
            current_angle += np.radians(angle)
        elif char == "[":
            # Сохраняем текущую позицию и угол
            stack.append((x, y, current_angle))
        elif char == "]":
            # Восстанавливаем сохраненную позицию и угол
            x, y, current_angle = stack.pop()
            points.append((x, y))

    return points


def draw_by_points(points):
    points = np.array(points)
    points -= points.min(axis=0)
    points /= points.max(axis=0)

    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], color="green")
    plt.axis("equal")
    plt.axis("off")

    plt.show()


def draw_iter(l_system_index):
    for ell in iterations:
        global default_iterations
        default_iterations = ell
        print(f"iter = {ell}")
        points = points_l_system(l_system_index=l_system_index)
        draw_by_points(points=points)


draw_iter(l_systems_index)