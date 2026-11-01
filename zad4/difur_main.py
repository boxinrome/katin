import math

def solve_variant(a_func, b_func, n=5000):
    a = 0.0
    b = 2.0 * math.pi
    d = (b - a) / n
    x = [a + i * d for i in range(n + 1)]
    
    y1 = [0.0] * (n + 1)
    y1[0] = 0.0
    y1[1] = d
    for i in range(1, n):
        y1[i + 1] = 2.0 * y1[i] - y1[i - 1] - (d ** 2) * a_func(x[i]) * y1[i]
        
    y2 = [0.0] * (n + 1)
    y2[0] = d
    y2[1] = 0.0
    for i in range(1, n):
        y2[i + 1] = 2.0 * y2[i] - y2[i - 1] - (d ** 2) * a_func(x[i]) * y2[i]
        
    yn = [0.0] * (n + 1)
    yn[0] = 0.0
    yn[1] = d
    for i in range(1, n):
        yn[i + 1] = 2.0 * yn[i] - yn[i - 1] + (d ** 2) * (b_func(x[i]) - a_func(x[i]) * yn[i])
        
    m11 = y1[0] - y1[n]
    m12 = y2[0] - y2[n]
    rhs1 = yn[n] - yn[0]
    
    dy1_0 = (y1[1] - y1[0]) / d
    dy1_n = (y1[n] - y1[n - 1]) / d
    dy2_0 = (y2[1] - y2[0]) / d
    dy2_n = (y2[n] - y2[n - 1]) / d
    dyn_0 = (yn[1] - yn[0]) / d
    dyn_n = (yn[n] - yn[n - 1]) / d
    
    m21 = dy1_0 - dy1_n
    m22 = dy2_0 - dy2_n
    rhs2 = dyn_n - dyn_0
    
    det = m11 * m22 - m12 * m21
    c1 = (rhs1 * m22 - m12 * rhs2) / det
    c2 = (m11 * rhs2 - rhs1 * m21) / det
    
    y = [c1 * y1[i] + c2 * y2[i] + yn[i] for i in range(n + 1)]
    
    def interpolate(x_val):
        idx = int((x_val - a) // d)
        xi = x[idx]
        xi1 = x[idx + 1]
        return y[idx] * (xi1 - x_val) / d + y[idx + 1] * (x_val - xi) / d

    return interpolate(1.0), interpolate(2.0)

variants = [
    (lambda x: math.sin(x), lambda x: math.sin(x) + 3.0 * math.sin(3.0 * x)),
    (lambda x: math.cos(x), lambda x: math.cos(x) - 1.0),
    (lambda x: 4.0 * math.sin(x), lambda x: math.cos(x) ** 2),
    (lambda x: 2.0 * math.sin(x) + math.cos(x), lambda x: math.sin(x) + 2.0),
    (lambda x: math.cos(x) ** 3, lambda x: math.cos(x) ** 2),
    (lambda x: 3.0 * math.sin(x), lambda x: math.cos(x) ** 3),
    (lambda x: math.cos(x), lambda x: math.sin(x) + math.cos(2.0 * x)),
    (lambda x: math.sin(x) + math.cos(x), lambda x: 2.0 * (math.sin(x) ** 2) + 3.0 * math.sin(3.0 * x)),
    (lambda x: 2.0 * math.sin(x), lambda x: 1.0 + 3.0 * math.cos(x)),
    (lambda x: math.sin(x) ** 2, lambda x: math.cos(4.0 * x) + 2.0 * math.sin(x))
]

for idx, (a_func, b_func) in enumerate(variants, start=1):
    f1, f2 = solve_variant(a_func, b_func)
    print(f"{idx}: f(1) = {f1:.8f}, f(2) = {f2:.8f}")