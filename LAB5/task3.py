import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Circle
import numpy as np

class SplineEditor:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.points = []
        self.circles = []
        self.curve = None
        self.selected_circle = None

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def add_point(self, x, y):
        point = np.array([x, y])
        self.points.append(point)

        circle = Circle((x, y), 0.2, color='red', picker=True)
        self.circles.append(circle)
        self.ax.add_patch(circle)

        self.update_curve()

    def remove_point(self, circle):
        index = self.circles.index(circle)
        self.points.pop(index)
        circle.remove()
        self.circles.pop(index)
        self.update_curve()

    def calculate_cubic_spline(self):
        points = np.array(self.points)
        x, y = points[:, 0], points[:, 1]
        n = len(x) - 1  # количество сегментов

        # Коэффициенты для каждой кривой
        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n):
            alpha[i] = (3 / h[i] * (y[i + 1] - y[i]) - 3 / h[i - 1] * (y[i] - y[i - 1]))

        # Решаем систему линейных уравнений для нахождения c
        l = np.ones(n + 1)
        mu = np.zeros(n)
        z = np.zeros(n + 1)
        for i in range(1, n):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        # Вычисление коэффициентов c, b, и d для каждого сегмента
        b = np.zeros(n)
        c = np.zeros(n + 1)
        d = np.zeros(n)
        a = y[:-1]

        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

        # Создаем кривую
        curve_points = []
        for i in range(n):
            xs = np.linspace(x[i], x[i + 1], 20)
            for xi in xs:
                yi = (a[i] + b[i] * (xi - x[i]) + c[i] * (xi - x[i]) ** 2 + d[i] * (xi - x[i]) ** 3)
                curve_points.append([xi, yi])

        return np.array(curve_points)

    def update_curve(self):
        if self.curve:
            self.curve.remove()

        if len(self.points) >= 2:
            curve_points = self.calculate_cubic_spline()
            self.curve, = self.ax.plot(curve_points[:, 0], curve_points[:, 1], 'blue')

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == MouseButton.LEFT:
            for circle in self.circles:
                contains, _ = circle.contains(event)
                if contains:
                    self.selected_circle = circle
                    return
            self.add_point(event.xdata, event.ydata)
        elif event.button == MouseButton.RIGHT:
            for circle in self.circles:
                contains, _ = circle.contains(event)
                if contains:
                    self.remove_point(circle)
                    return

    def on_release(self, event):
        self.selected_circle = None

    def on_motion(self, event):
        if not self.selected_circle or event.inaxes != self.ax:
            return
        index = self.circles.index(self.selected_circle)
        self.points[index] = np.array([event.xdata, event.ydata])
        self.selected_circle.center = (event.xdata, event.ydata)
        self.update_curve()

    def show(self):
        plt.show()

if __name__ == "__main__":
    editor = SplineEditor()
    editor.show()