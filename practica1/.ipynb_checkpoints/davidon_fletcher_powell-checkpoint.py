import numpy as np
from typing import Callable, Tuple, Union, List

def search_lambda(
    function: Callable[[np.ndarray], float],
    grad: np.ndarray,
    xn: np.ndarray,
    d: np.ndarray,
    lam: float = 1.0,
    rho: float = 0.5,
    c: float = 0.1
) -> float:
    """
    Minimiza la función respecto a lambda.

    Parameters:
    function : Función a minimizar.
    grad : Gradiente de la función a minimizar.
    xn : Punto actual.
    d : Dirección de descenso.
    lam : Tamaño del paso en la dirección del gradiente.
    rho : Factor de reducción.
    c : La constante para la condición.

    Returns: Tamaño del paso en la dirección del gradiente.
    """
    while function(xn + lam * d) > function(xn) + c * lam * np.dot(grad, d):
        lam *= rho

    return lam


def davidon_fletcher_powell(
    function: Callable[[np.ndarray], float],
    grad_function: Callable[[np.ndarray], np.ndarray],
    x0: Union[np.ndarray, List[float]],
    max_error: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[float, np.ndarray]:
    """
    Estima el mínimo de una función según el algoritmo de Davidon-Fletcher-Powell.

    Parameters:
    function : Función a minimizar.
    grad_function : Gradiente de la función a minimizar.
    x0 : Valor inicial.
    max_error : Error máximo para parar el algoritmo.
    max_iter : Número de iteraciones máximo.

    Returns: Mínimo de la función y valor que toma en ese punto.
    """
    j = 1

    # Matriz identidad como aproximación inicial del Hessiano inverso
    D = np.eye(len(x0))  

    xn = x0
    grad = grad_function(xn)

    while j < max_iter and np.linalg.norm(grad) >= max_error:
        # Dirección
        d = -np.dot(D, grad)

        # Paso 1
        lam = search_lambda(function, grad, xn, d)

        # Paso 2
        x_new = xn + lam * d
        p = x_new - xn #y
        q = grad_function(x_new) - grad 

        Dy = D @ q
        D = D - np.outer(Dy, Dy) / (q @ Dy) + np.outer(p, p) / (p @ q)

        xn = x_new
        grad = grad_function(xn)
        j += 1

    return function(xn), xn


def main():
    f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
    
    x0 = [0.0, 0.0]

    x_min, y_min = davidon_fletcher_powell(f, grad_f, x0)

    print(f'El valor de x que minimiza la función es:')
    print(f'x={x_min}\nf(x)={y_min}')

    return 0


if __name__ == "__main__":
    main()