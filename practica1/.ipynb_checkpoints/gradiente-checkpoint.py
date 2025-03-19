import numpy as np
from typing import Callable, Union, Tuple

def descenso_por_gradiente(
    function: Callable[[Union[np.ndarray, float]], float],
    grad_function: Callable[[Union[np.ndarray, float]], Union[np.ndarray, float]],
    x0: Union[np.ndarray, float],
    learning_rate: float = 0.01,
    max_error: float = 1e-6,
    max_iter: int = 100
) -> Tuple[Union[np.ndarray, float], float]:
    """
    Estima el mínimo de una función mediante el descenso por gradiente.

    Parameters:
    function : Función a minimizar.
    grad_function : Gradiente de la función a minimizar.
    x0 : Valor inicial.
    learning_rate : Tasa de aprendizaje.
    max_error : Error máximo para parar el algoritmo.
    max_iter : Número de iteraciones máximo.

    Returns: Mínimo de la función y valor que toma en ese punto.
    """
    i = 0
    x = x0
    gradient = grad_function(x)

    while i < max_iter and np.linalg.norm(gradient) >= max_error:
        x -= learning_rate * gradient 
        gradient = grad_function(x)
        i += 1

    return x, function(x)


def main():
    # Función a minimizar
    f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
    
    # Gradiente de la función
    grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])

    # Punto inicial
    x0 = [0.0, 0.0]
    
    x_min, y_min = descenso_por_gradiente(f, grad_f, x0, max_iter=500)
    print(f'El valor de x que minimiza la función es:')
    print(f'x={x_min}\nf(x)={y_min}')
    
    return 0


if __name__ == "__main__":
    main()