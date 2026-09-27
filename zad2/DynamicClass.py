import numpy as np
import matplotlib.pyplot as plt


class DynamicClass:
    def __init__(self, var: float = 1, x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), 
                 resolution: int = 1000, max_iter: int = 200, escape_radius: float = 2.0):
        """
        Инициализация динамической системы.
        :param var: номер варианта (используется в c = 0.028 * var)
        :param x_range: диапазон по вещественной оси (Re)
        :param y_range: диапазон по мнимой оси (Im)
        :param resolution: разрешение сетки (resolution x resolution точек)
        :param max_iter: максимальное число итераций
        :param escape_radius: радиус отсечения (порог выхода на бесконечность)
        """
        self.var = var
        self.c = complex(0.028 * var, 0.0)  # константа c = 0.028 * var
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.max_iter = max_iter
        self.escape_radius = escape_radius

        self.mask = None          # Булева маска ограниченных точек (True = внутри множества)
        self.iterations = None    # Количество итераций до вылета (для градиентной отрисовки)

    def compute(self):
        """
        Векторизованное вычисление орбит точек z_1.
        Определяет, какие точки остаются ограниченными.
        """
        # Создаем сетку начальных комплексных точек z_1
        x = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Массив для фиксации числа итераций
        self.iterations = np.full(Z.shape, self.max_iter, dtype=int)
        
        # Маска активных точек, которые ещё не вылетели
        active = np.ones(Z.shape, dtype=bool)

        for i in range(self.max_iter):
            # z_{i+1} = z_i^2 + c для всех активных точек
            Z[active] = Z[active] ** 2 + self.c
            
            # Точки, вышедшие за порог радиуса
            escaped = np.abs(Z) > self.escape_radius
            
            # Фиксируем шаг выхода для вновь вылетевших точек
            newly_escaped = escaped & active
            self.iterations[newly_escaped] = i
            
            # Исключаем их из дальнейших вычислений
            active[escaped] = False

        # Ограниченными считаются точки, не превысившие радиус за max_iter шагов
        self.mask = (self.iterations == self.max_iter)
        return self.mask

    def plot(self, show_colored: bool = True):
        """
        Отображение точек на комплексной плоскости.
        :param show_colored: если True, показывает красивый градиент вылета точек;
                             если False, чисто бинарное закрашивание.
        """
        if self.mask is None:
            self.compute()

        plt.figure(figsize=(9, 8))
        
        if show_colored:
            # Черный цвет (значение max_iter) соответствует точкам множества
            plt.imshow(self.iterations, extent=[self.x_range[0], self.x_range[1],
                                                self.y_range[0], self.y_range[1]],
                       cmap='magma', origin='lower')
            cbar = plt.colorbar(label='Итерации до выхода на бесконечность')
        else:
            plt.imshow(self.mask, extent=[self.x_range[0], self.x_range[1],
                                          self.y_range[0], self.y_range[1]],
                       cmap='binary', origin='lower')

        plt.title(f"Множество Жюлиа: $z_{{i+1}} = z_i^2 + {self.c.real:.4f}$ (var = {self.var})", fontsize=14)
        plt.xlabel("$\nathrm{Re}(z)$", fontsize=12)
        plt.ylabel("$\nathrm{Im}(z)$", fontsize=12)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def estimate_dimension(self, target: str = 'boundary') -> float:
        """
        Оценка фрактальной размерности методом Box-Counting (подсчет ячеек).
        :param target: 
            'boundary' — размерность границы области (фрактальная кривая);
            'filled'   — размерность всей закрашенной области (равна 2, если имеет площадь).
        :return: оценка размерности Минковского.
        """
        if self.mask is None:
            self.compute()

        if target == 'boundary':
            # Выделяем границу бинарной маски (дифференцирование соседей)
            padded = np.pad(self.mask, 1, mode='constant', constant_values=False)
            eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] &
                      padded[1:-1, :-2] & padded[1:-1, 2:])
            binary_img = self.mask ^ eroded
        else:
            binary_img = self.mask.copy()

        # Масштабируем до степени двойки для ровного деления ячеек
        p = int(np.floor(np.log2(min(binary_img.shape))))
        n = 2 ** p
        img = binary_img[:n, :n]

        scales = []
        counts = []

        # Перебираем размеры ячеек (степени двойки)
        for k in range(2, p - 1):
            box_size = 2 ** k
            # Разбиваем матрицу на блоки размера box_size x box_size
            reduced = img.reshape(n // box_size, box_size, n // box_size, box_size)
            # Ячейка занята, если внутри неё есть хотя бы один ненулевой пиксель
            occupied = reduced.any(axis=(1, 3)).sum()

            if occupied > 0:
                scales.append(1.0 / box_size)
                counts.append(occupied)

        # Вычисляем наклон прямой в логарифмических координатах: ln(N) = D * ln(1/eps) + C
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        dimension = coeffs[0]
        return dimension