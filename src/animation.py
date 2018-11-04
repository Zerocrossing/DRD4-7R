import matplotlib.pyplot as plt
from src.utils import *


# Need to figure out how to use animation (funcanimation or similar) to keep the same figure and update
class Animation:

    def __init__(self, data):
        self.fig, self.ax = plt.subplots()
        self.x, self.y = zip(*data)
        return

    def update(self, arr):
        start_timer("animation")
        plt.plot(self.x, self.y, 'ro')
        for i in range(0, len(arr) - 1):
            x1, x2 = self.x[arr[i]], self.x[arr[i + 1]]
            y1, y2 = self.y[arr[i]], self.y[arr[i + 1]]
            plt.plot([x1, x2], [y1, y2], 'k-')
        plt.show()
        add_timer("animation")
