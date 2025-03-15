import numpy as np
from scipy.optimize import minimize_scalar


def descenso_por_gradiente(function, grad_function, x0=[0.0, 0.0], max_error=1e-6, max_iter=100):
    k = 1

    xn = x0
    grad = grad_function(xn)

    while k < max_iter and np.linalg.norm(grad) > max_error:

        d = -grad

        # Búsqueda de línea para encontrar el mejor lambda
        objective = lambda l: function(xn + l * d)
        res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        lamb = res.x

        xn = xn + lamb * d
        k += 1 

    return function(xn), xn


def main():
    f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
    
    x0 = [0.0, 0.0]

    min, x_min = descenso_por_gradiente(f, grad_f, x0)

    print(f"min f() = f({x_min}) = {min}")

    return 0


if __name__ == "__main__":
    main()