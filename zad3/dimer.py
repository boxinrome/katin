import numpy as np

# Параметры задачи:
VAR = 9
N_ATOMS = 6
DIM = 3

def calc_potential_and_forces(x_flat):
    """
    Вычисляем энергию U и вектор сил F (F = -grad U)
    для N атомов при потенциале LJ: U = VAR * sum(1/r^12 - 2/r^6)
    """
    pos = x_flat.reshape((N_ATOMS, DIM))
    forces = np.zeros_like(pos)
    U = 0.0

    for i in range(N_ATOMS):
        for j in range(i + 1, N_ATOMS):
            r_vec = pos[i] - pos[j] # вектор смещения между атомами i и j
            r2 = np.dot(r_vec, r_vec)  # distance squared
            r = np.sqrt(r2)
            
            
            inv_r6 = 1.0 / (r2 ** 3) # члены ряда -12 и минус 6 степени 
            inv_r12 = inv_r6 * inv_r6
            
          
            u_pair = VAR * (inv_r12 - 2.0 * inv_r6) #энергия пары  
            U += u_pair
            
            # dU/dr = VAR * (-12/r^13 + 12/r^7) = 12 * VAR * (inv_r6 - inv_r12) / r
            # F_i = - dU/dr * (r_vec / r) = 12 * VAR * (inv_r12 - inv_r6) / r^2 * r_vec
            f_mag_over_r = 12.0 * VAR * (inv_r12 - inv_r6) / r2
            f_pair = f_mag_over_r * r_vec
            
            forces[i] += f_pair
            forces[j] -= f_pair

    return U, forces.flatten()


def get_initial_octahedron():
    """
    Правильный октаэдр с центром в (0,0,0) и ребром a = 1.
    Расстояние от центра до вершин R = 1 / sqrt(2).
    """
    R = 1.0 / np.sqrt(2.0)
    pos = np.array([
        [ R,  0,  0],
        [-R,  0,  0],
        [ 0,  R,  0],
        [ 0, -R,  0],
        [ 0,  0,  R],
        [ 0,  0, -R]
    ], dtype=np.float64)
    return pos.flatten()


def dimer_saddle_search(x_init, delta_len=1e-3, step_rot=0.01, step_trans=0.005, max_iter=2000, tol=1e-4):
    """
    Поиск седловой точки методом димера .
    """
    np.random.seed(42)
    
    # 1. Задаем x1 и x2
    x1 = x_init.copy()
    random_dir = np.random.randn(*x1.shape)
    random_dir /= np.linalg.norm(random_dir)
    
    # Фиксированная длина димера L = 2 * delta_len
    L = 2.0 * delta_len
    x2 = x1 + L * random_dir

    print("Initializind algorithm... ")
    for it in range(max_iter):
        # Вектор ориентации димера
        n = (x2 - x1)
        dist = np.linalg.norm(n)
        n /= dist

    #    шаг а - вращаем димер
        # Поворачиваем x2 вокруг x1 (сохраняя длину L)
        for _ in range(20):
            _, f2 = calc_potential_and_forces(x2)
            f2_parallel = np.dot(f2, n) * n
            f2_perp = f2 - f2_parallel
            
            if np.linalg.norm(f2_perp) < 1e-6:
                break
            
            x2 = x2 + step_rot * f2_perp
            # Восстанавливаем точную длину димера
            n_new = (x2 - x1)
            x2 = x1 + L * (n_new / np.linalg.norm(n_new))
            n = (x2 - x1) / L

        # шаг б - трансляция димера     
        _, f1 = calc_potential_and_forces(x1)
        _, f2 = calc_potential_and_forces(x2)
        f_total = 0.5 * (f1 + f2)

        f_total_parallel = np.dot(f_total, n) * n
        f_total_perp = f_total - f_total_parallel

        # Эффективная сила трансляции: подъем по n, спуск поперек n
        f_eff = f_total_perp - f_total_parallel
        
        # Проверка сходимости
        norm_f = np.linalg.norm(f_eff)
        if norm_f < tol:
            print(f"Сходимость достигнута на итерации {it}, ||F_eff|| = {norm_f:.2e}")
            break

        shift = step_trans * f_eff
        x1 += shift
        x2 += shift

    # Центр димера соответствует седловой точке
    x_saddle = 0.5 * (x1 + x2)
    return x_saddle

    # Основной расчет
if __name__ == "__main__":
    x_oct = get_initial_octahedron()
    U_oct, _ = calc_potential_and_forces(x_oct)

    print("=" * 60)
    print("Staring solving")
    print("=" * 60)
    print(f"Энергия идеального октаэдра (ребро = 1): U_oct = {U_oct:.6f}\n")

    # Запуск поиска седловой точки (выхода из потенциальной ямы октаэдра)
    x_ts = dimer_saddle_search(x_oct)
    U_ts, forces_ts = calc_potential_and_forces(x_ts)

    barrier = U_ts - U_oct

    print("\nResults: ")
    print(f"Энергия в седловой точке (Transition State): U_TS = {U_ts:.6f}")
    print(f"Норма градиента в седловой точке:           ||grad U|| = {np.linalg.norm(forces_ts):.2e}")
    print(f"Минимальная энергия преодоления барьера:   E_act = U_TS - U_oct = {barrier:.6f}")
    # print(f"Полная энергия распада на свободные атомы: E_atomization = {-U_oct:.6f}")
    print("=" * 60)