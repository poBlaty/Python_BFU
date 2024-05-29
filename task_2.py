import matplotlib.pyplot as plt
import numpy as np

# 2. Реализовать на Python и отрисовать с помощью Matplotlib ряд из фигур Лисажу (4 графика)
# с разным соотношение частот (3:2), (3:4), (5:4), (5:6).

ratios = ((3, 2), (3, 4), (5, 4), (5, 6))

t = np.linspace(0, 2 * np.pi, 1000) # множесто из 1000 точек от 0 до 2 * np.pi

fig, axs = plt.subplots(2, 2, figsize=(8, 8)) # окно 8 на 8 для четырех графиков

for ax, (a, b) in zip(axs.ravel(), ratios):
    x = np.sin(a * t)
    y = np.sin(b * t)

    ax.plot(x, y, label=f'Ratio {a}:{b}') # построение графика
    ax.legend() # отображение label из ax.plot()
    ax.grid(True)  # сетка


plt.show()
