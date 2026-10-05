from DynamicClass import DynamicClass


def main():
    VAR = 12 

    print(f"=== Запуск для var = {VAR} ===")
    
    system = DynamicClass(
        var=VAR,
        x_range=(-1.6, 1.6),
        y_range=(-1.3, 1.3),
        resolution=1024,
        max_iter=100
    )

    print(f"Параметр c = {system.c.real:.4f}")
    print("Вычисление точек...")
    system.compute()

    dim = system.estimate_dimension()
    print("\n-------------------------------------------")
    print(f"Оценка размерности области: D ≈ {dim:.4f}")
    print("-------------------------------------------")

    print("Построение графика...")
    system.plot()


if __name__ == "__main__":
    main()