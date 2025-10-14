import matplotlib.pyplot as plt
import numpy as np
import random


# Постепенный переход 139, 69, 1 -> 0, 128, 0
def get_color(depth, max_depth):
    r = (121 - 121 * (max_depth - depth) / max_depth) / 255
    g = (52 + (128 - 69) * (max_depth - depth) / max_depth) / 255
    b = (1 - 1 * (max_depth - depth) / max_depth) / 255
    return (r, g, b)


def get_thickness(depth, max_depth):
    return max(1, (depth + 1) // 2)


def draw_tree(ax, x, y, branch_length, angle, depth, max_depth):
    if depth == 0:
        return

    x_end = x + branch_length * np.cos(np.radians(angle))
    y_end = y + branch_length * np.sin(np.radians(angle))

    color = get_color(depth, max_depth)
    thickness = get_thickness(depth, max_depth)

    ax.plot([x, x_end], [y, y_end], color=color, linewidth=thickness)

    new_branch_length = branch_length * 0.7
    draw_tree(
        ax,
        x_end,
        y_end,
        new_branch_length,
        angle + 25 + random.uniform(-15, 15),
        depth - 1,
        max_depth,
    )
    draw_tree(
        ax,
        x_end,
        y_end,
        new_branch_length,
        angle - 25 + random.uniform(-15, 15),
        depth - 1,
        max_depth,
    )


def draw_fractal_tree(branch_length=100, angle=90, depth=8):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    draw_tree(ax, 0, 0, branch_length, angle, depth, depth)

    plt.show()


draw_fractal_tree(branch_length=100, angle=90, depth=10)