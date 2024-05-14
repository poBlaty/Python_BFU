import numpy as np

# 1.Сохранить этот текст в файл. Прочитать матрицу из файла.
# Hайдите для этой матрицы сумму всех элементов, максимальный и минимальный
# элемент (число)

def read_matrix(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        lines = f.readlines()
        mat = []
        for i in lines:
            mat.append(list(map(int, i.replace("\n", "").split(','))))
        return np.array(mat)


def task_1():
    mat = read_matrix("input.txt")
    mat_sum = np.sum(mat)
    maxx = np.max(mat)
    minn = np.min(mat)
    return mat_sum, maxx, minn

print(task_1())

# 2. Реализовать кодирование длин серий (Run-length encoding). Дан вектор x. Необходимо
# вернуть кортеж из двух векторов одинаковой длины. Первый содержит числа, а второй -
# сколько раз их нужно повторить. Пример: x = np.array([2, 2, 2, 3, 3, 3, 5]). Ответ:
# (np.array([2, 3, 5]), np.array([3, 3, 1])).

# print(np.unique([2, 2, 2, 3, 3, 3, 5], return_counts=True))

# 3. Написать программу NumPy генерирующую массив случайных чисел нормального
# распределения размера 10х4. Найти минимально, максимальное, средние значения,
# стандартное отклонение. Сохранить первые 5 строк в отдельную переменную.

# a = np.random.normal(size=[10, 4])
# m = a[0:5]
# print(np.min(m), np.max(m), np.average(m), np.std(m))

# 4. Найти максимальный элемент в векторе x среди элементов, перед которыми стоит
# нулевой. Для x = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0]) ответ 5.

## s = np.array([6, 2, 0, 3, 0, 0, 5, 7, 0])
# s = np.random.randint(0, 10, size=10)
# print(s)
# index = np.where(s[:-1] == 0)[0]
# if len(index) > 0:
#     print(np.max(s[index + 1]))
# else:
#     print("There are no zeros before other elements.")

# 5. Реализовать функцию вычисления логарифма плотности многомерного нормального
# распределения Входные параметры: точки X, размер (N, D), мат. ожидание m, вектор
# длины D, матрица ковариаций C, размер (D, D). Разрешается использовать библиотечные
# функции для подсчета определителя матрицы, а также обратной матрицы, в том числе в
# невекторизованном варианте. Сравнить с scipy.stats.multivariate_normal(m, C).logpdf(X) как
# по скорости работы, так и по точности вычислений.

# from typing import NamedTuple
# from scipy.stats import multivariate_normal
# import time
#
#
# class Data(NamedTuple):
#     result: float
#     time: float
#
#
# def calculate_time(func):
#     def inner(*args, **kwargs):
#         start = time.time()
#         result = func()
#         end = time.time()
#         return Data(result=result, time=end - start)
#
#     return inner
#
#
# def log_multivariate(X, m, C):
#     D = m.shape[0]
#     det_C = np.linalg.det(C)
#     inv_C = np.linalg.inv(C)
#     constant = -0.5 * (D * np.log(2 * np.pi) + np.log(det_C))
#     diff = X - m
#     exponent = -0.5 * np.sum(diff @ inv_C * diff, axis=1)
#     return constant + exponent
#
#
# X = np.random.randn(1000, 4)
# m = np.random.randn(4)
# C = np.random.randn(4, 4)
# C = np.dot(C, C.T)
#
#
# @calculate_time
# def through_np():
#     return log_multivariate(X, m, C)
#
#
# @calculate_time
# def through_spipy():
#     return multivariate_normal(m, C).logpdf(X)
#
#
# first = through_np()
# second = through_spipy()
#
# print("через numpy", first.time)
# print("Время выполнения используя scipy", second.time)
#
# print("Разница в логарифме плотности", np.max(np.abs(first.result - second.result)))

# 6. Поменять местами две строки в двумерном массиве NumPy -  поменяйте  строки 1 и 3
# массиваа.  a = np.arange(16).reshape(4,4)

# a = np.arange(16).reshape(4, 4)
# a[[0, 2]] = a[[2, 0]]
# print(a)

# 7. Найти уникальные значения и их количество в столбцеspeciesтаблицыiris.
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris = np.genfromtxt(url, delimiter=',', dtype='object')
#
# species_column = iris[:, 4]
# unique_species, counts = np.unique(species_column, return_counts=True)
#
# for i in range(len(unique_species)):
#     print("Уникальное значение", unique_species[i].decode('utf-8'), " Количество", counts[i])


# 8. Найти индексы ненулевых элементов в [0,1,2,0,0,4,0,6,9]

# arr = np.array([0, 1, 2, 0, 0, 4, 0, 6, 9])
# nonzero = np.nonzero(arr)
# print(nonzero)