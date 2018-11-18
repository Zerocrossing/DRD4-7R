import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ffmpeg
from src.utils import *

# Need to figure out how to use animation (funcanimation or similar) to keep the same figure and update
class Animation:

    def __init__(self, data, history):
        self.fig, self.ax = plt.subplots()
        self.x, self.y = zip(*data)
        self.history = history
        self.lines = []
        plt.plot(self.y, self.x, 'ro')
        plt.gca().invert_xaxis()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Animation of Best Individual Over Multiple Generations")
        for index in range(len(self.history[0])-1):
            lobj = self.ax.plot([], [])[0]
            self.lines.append(lobj)

        return

    def init(self):
        for line in self.lines:
            line.set_data([],[])
        return self.lines

    def animate(self, i):
        xlist, ylist = [], []
        for ind in range(len(self.history[i])-1):
            x1, x2 = self.x[self.history[i][ind]], self.x[self.history[i][ind+1]]
            y1, y2 = self.y[self.history[i][ind]], self.y[self.history[i][ind+1]]
            xlist.append([x1,x2])
            ylist.append([y1,y2])

        for lnum, line in enumerate(self.lines):
            line.set_data(ylist[lnum], xlist[lnum])

        return self.lines

    def start(self):
        start_timer("animation")
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames =len(self.history), interval=500, blit=True)
        anim.save('animation.mp4')
        self.plot_last()
        plt.show()
        add_timer("animation")

    def plot_last(self):
        arr = self.history[-1]
        for i in range(len(arr) - 1):
            x1, x2 = self.x[arr[i]], self.x[arr[i + 1]]
            y1, y2 = self.y[arr[i]], self.y[arr[i + 1]]
            plt.plot([y1, y2], [x1, x2], 'k-')
