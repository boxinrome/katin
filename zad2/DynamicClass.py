import numpy as np
import matplotlib.pyplot as plt


class DynamicClass:
    def __init__(self, var: float = 12, x_range=(-1.8, 1.8), y_range=(-1.5, 1.5), 
                 resolution: int = 1000, max_iter: int = 150):
        self.var = var
        self.c = complex(0.028 * var, 0.0)
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.max_iter = max_iter
        
        # Теоретический радиус отсечения: R = max(2, |c| + 1)
        self.escape_radius = max(2.0, abs(self.c) + 1.0)

        self.mask = None
        self.iterations = None

    def compute(self):
        """Векторизованный расчет траекторий z_{i+1} = z_i^2 + c."""
        x = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        self.iterations = np.full(Z.shape, self.max_iter, dtype=int)
        active = np.ones(Z.shape, dtype=bool)

        for i in range(self.max_iter):
            Z[active] = Z[active] ** 2 + self.c
            escaped = np.abs(Z) > self.escape_radius
            newly_escaped = escaped & active
            self.iterations[newly_escaped] = i
            active[escaped] = False

        # Ограниченные точки (те, что не вылетели)
        self.mask = (self.iterations == self.max_iter)
        
        # Если множество распалось в пыль (c > 0.25), пиксели могут быть очень редкими.
        # В таком случае берем точки с наибольшей живучестью (квази-инвариантное множество)
        if np.sum(self.mask) < 50:
            threshold = int(self.max_iter * 0.75)
            self.mask = self.iterations >= threshold

        return self.mask

    def plot(self):
        """Отрисовка динамической системы на плоскости."""
        if self.mask is None:
            self.compute()

        plt.figure(figsize=(9, 8))
        plt.imshow(self.iterations, extent=[self.x_range[0], self.x_range[1],
                                            self.y_range[0], self.y_range[1]],
                   cmap='inferno', origin='lower')
        plt.colorbar(label='Число итераций до вылета')
        plt.title(f"Множество Жюлиа: $z_{{i+1}} = z_i^2 + {self.c.real:.4f}$ (var = {self.var})", fontsize=13)
        plt.xlabel("$\mathrm{Re}(z)$", fontsize=12)
        plt.ylabel("$\mathrm{Im}(z)$", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def estimate_dimension(self) -> float:
        """
        Оценка размерности области методом Box-Counting.
        Корректно работает как для сплошных фигур, так и для фрактальной пыли.
        """
        if self.mask is None:
            self.compute()

        # Приводим к степени двойки (например, 512, 1024)
        p = int(np.floor(np.log2(min(self.mask.shape))))
        n = 2 ** p
        img = self.mask[:n, :n]

        scales = []
        counts = []

        # Перебираем размеры ячеек (от 4 до n/4 пикселей)
        for k in range(2, p - 1):
            box_size = 2 ** k
            reduced = img.reshape(n // box_size, box_size, n // box_size, box_size)
            occupied = reduced.any(axis=(1, 3)).sum()

            if occupied > 0:
                scales.append(1.0 / box_size)
                counts.append(occupied)

        if len(scales) < 2:
            print("Предупреждение: недостаточно масштабов для регрессии.")
            return 0.0

        # Наклон прямой ln(N) = D * ln(1/eps)
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        return float(coeffs[0])