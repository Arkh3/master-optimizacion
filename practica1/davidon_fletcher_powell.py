import numpy as np

def davidon_fletcher_powell(function, grad_function, x0=[0.0, 0.0], max_error=1e-6, max_iter=100):

    j = 1

    # Matriz identidad como aproximación inicial del Hessiano inverso
    D = np.eye(len(x0))  

    xn = x0
    grad = grad_function(xn)

    while j < max_iter and np.linalg.norm(grad) > max_error:
    
        # Dirección
        d = -D @ grad 

        # Paso 1
        lam = 1 
        c = 1e-4
        rho = 0.9

        while function(xn + lam * d) > function(xn) + c * lam * np.dot(grad, d):
            lam *= rho

        # Paso 2
        x_new = xn + lam * d
        p = x_new - xn
        q = grad_function(x_new) - grad

        if np.dot(p, q) > 0:  # Evitar divisiones por cero o valores negativos
            rho = 1.0 / np.dot(p, q)
            Dy = D @ q
            D = D + rho * np.outer(p, p) - (np.outer(Dy, p) + np.outer(p, Dy)) / np.dot(q, p)

        xn = x_new
        grad = grad_function(xn)

        j += 1

    
    return function(xn), xn


def main():
    f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
    
    x0 = [0.0, 0.0]

    min, xn = davidon_fletcher_powell(f, grad_f, x0)

    print(f"min f() = f({xn}) = {min}")

    return 0


if __name__ == "__main__":
    main()