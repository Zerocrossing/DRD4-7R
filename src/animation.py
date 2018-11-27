"""
Plot the best individual
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Animation:

    def __init__(self, data, history, fitness, generation):
        self.fig, self.ax = plt.subplots()
        self.x, self.y = zip(*data)
        self.history = history
        self.lines = []
        self.fitness = fitness
        self.generation = generation
        # transposing data to reflect real maps
        plt.plot(self.y, self.x, 'ro')
        plt.gca().invert_xaxis()
        # labels
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Generation=" + str(self.generation[0]) + ", Fitness=" + str(round(self.fitness[0])))
        # initialize line objects
        for index in range(len(self.history[0])-1):
            lobj = self.ax.plot([], [])[0]
            self.lines.append(lobj)

    def init(self):
        for line in self.lines:
            line.set_data([],[])
        return self.lines

    def animate(self, i):
        x_list, y_list = [], []
        # draw line objects between each point
        for ind in range(len(self.history[i])-1):
            x1, x2 = self.x[self.history[i][ind]], self.x[self.history[i][ind+1]]
            y1, y2 = self.y[self.history[i][ind]], self.y[self.history[i][ind+1]]
            x_list.append([x1,x2])
            y_list.append([y1,y2])
        # return artist for animation
        for lnum, line in enumerate(self.lines):
            line.set_data(y_list[lnum], x_list[lnum])
        plt.title("Generation=" + str(self.generation[i]) + ", Fitness=" + str(round(abs(self.fitness[i]))))
        return self.lines

    def start(self):
        ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                       frames =len(self.history), interval=500, blit=True)
        ani.save('animation.mp4')
        self.plot_best_individual()
        plt.show()

    def plot_best_individual(self):
        arr = self.history[-1]
        for i in range(len(arr) - 1):
            x1, x2 = self.x[arr[i]], self.x[arr[i + 1]]
            y1, y2 = self.y[arr[i]], self.y[arr[i + 1]]
            plt.plot([y1, y2], [x1, x2], 'k-')

