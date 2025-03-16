import numpy as np

def search_lambda(function, grad, xn, d, lam=1.0, rho=0.5, c=0.1):

    while function(xn + lam * d) > function(xn) + c * lam * np.dot(grad, d):
        lam *= rho

    return lam

def davidon_fletcher_powell(function, grad_function, x0=[0.0, 0.0], max_error=1e-6, max_iter=10000):
    j = 1

    # Matriz identidad como aproximación inicial del Hessiano inverso
    D = np.eye(len(x0))  

    xn = x0
    grad = grad_function(xn)

    while j < max_iter and np.linalg.norm(grad) >= max_error:
        # Dirección
        d = -D @ grad 

        # Paso 1
        lam = search_lambda(function, grad, xn, d)

        # Paso 2
        x_new = xn + lam * d
        p = x_new - xn
        q = grad_function(x_new) - grad

        Dy = D @ q
        D = D - np.outer(Dy, Dy) / (q @ Dy) + np.outer(p, p) / (q @ (lam * d))

        xn = x_new
        grad = grad_function(xn)

        j += 1

    
    return function(xn), xn


def main():
    f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
    
    x0 = [0.0, 0.0]

    min, x_min = davidon_fletcher_powell(f, grad_f, x0)

    print(f"min f() = f({x_min}) = {min}")

    return 0


if __name__ == "__main__":
    main()