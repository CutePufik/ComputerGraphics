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

    def calculate_bezier_curve(self):
        points = np.array(self.points)
        n = len(points)
        if n < 2:
            return np.array([])

        curve_points = []
        num_per_segment = 20

        for i in range(n - 1):
            Pi = points[i]
            Pi1 = points[i + 1]
            if i == 0:
                Pm1 = points[0]
            else:
                Pm1 = points[i - 1]
            if i + 2 < n:
                Pi2 = points[i + 2]
            else:
                Pi2 = points[i + 1]

            C1 = Pi + (Pi1 - Pm1) / 6.0
            C2 = Pi1 - (Pi2 - Pi) / 6.0

            t = np.linspace(0, 1, num_per_segment)
            t = t[:, np.newaxis]
            seg = (1 - t) ** 3 * Pi + 3 * (1 - t) ** 2 * t * C1 + 3 * (1 - t) * t ** 2 * C2 + t ** 3 * Pi1
            curve_points.append(seg)

        return np.vstack(curve_points)

    def update_curve(self):
        if self.curve:
            self.curve.remove()

        if len(self.points) >= 2:
            curve_points = self.calculate_bezier_curve()
            if curve_points.size > 0:
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