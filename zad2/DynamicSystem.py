from DynamicClass import DynamicClass


def main():
    VAR = 9  
    print(f"=== Расчет динамической системы для var = {VAR} ===")
    
    # инициализация модели
    system = DynamicClass(
        var=VAR,
        x_range=(-1.8, 1.8),
        y_range=(-1.5, 1.5),
        resolution=1200,
        max_iter=200
    )

    print("Вычисление множества ограниченных точек...")
    system.compute()

    # Оценка размерности границы
    dim_boundary = system.estimate_dimension(target='boundary')
    # Оценка размерности всей закрашенной области
    dim_filled = system.estimate_dimension(target='filled')

    print("\n--- Результаты оценки размерности ---")
    print(f"Размерность границы области : D ≈ {dim_boundary:.4f}")
    print(f"Размерность самой закрашенной области:       D ≈ {dim_filled:.4f}")
    print("-----------------------------------------------------")

    # Отображение графика
    print("Построение визуализации...")
    system.plot(show_colored=True)


if __name__ == "__main__":
    main()