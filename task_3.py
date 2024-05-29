import matplotlib.pylab as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 3. Реализовать с помощью Matplotlib анимацию врашения фигуры Лисажу
# при нулевом сдвиге фаз и изменении соотношения частот от 0 до 1

figure = plt.figure()
ratio = np.linspace((0, 1), (5, 5), num=100)
degree = np.linspace((0, 0), (2 * np.pi, 2 * np.pi), num=12)[::-1]
colors = np.linspace((0, 0, 0), (1, 1, 1), num=1200)

t = np.linspace(-4, 4, num=500)

x = np.array([np.sin(a * t + alpha) for alpha, beta in degree for a, b in ratio])
y = np.array([np.sin(b * t + beta) for alpha, beta in degree for a, b in ratio])


def animate(i):
    figure.clear()
    plt.plot(x[i], y[i], color=colors[i])


anim = FuncAnimation(figure, animate, frames=1200, interval=1)
plt.show()
