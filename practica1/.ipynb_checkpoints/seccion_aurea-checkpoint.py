import math
from typing import Callable, Tuple


def seccion_aurea(
    function: Callable[[float], float],
    search_interval: Tuple[float, float],
    max_error: float = 0.001,
    max_iter: int = 100
) -> float:
    """
    Algoritmo de búsqueda de la Sección Áurea.

    Parameters:
    function : Función a minimizar.
    search_interval : Intervalo inicial.
    max_error : Error máximo para parar el algoritmo.
    max_iter : Número de iteraciones máximo.

    Returns: Mínimo de la función y valor que toma en ese punto.
    """
   
    razon_aurea = (math.sqrt(5) - 1) / 2

    a, b = search_interval

    i = 0
    while i < max_iter and (b - a) >= max_error:
        # Puntos posibles
        aux_1 = b - razon_aurea * (b - a)
        aux_2 = a + razon_aurea * (b - a)

        # Nuevo intervalo
        if function(aux_1) > function(aux_2):
            a = aux_1
        else:
            b = aux_2
        
        i += 1
    theta_min = (a + b) / 2
    # Resultado: Punto medio del intervalo
    return theta_min, function(theta_min)


def main():
    theta = lambda lamb: (5 - 2 * lamb + (4 + lamb)**3)**2 + 2 * (-3 - 3 * lamb)**4
    search_interval = (-5, 5)
    
    x_min, y_min = seccion_aurea(theta, search_interval)
    print(f'El valor de x que minimiza la función es:')
    print(f'x={x_min}\nf(x)={y_min}')

    return 0


if __name__ == "__main__":
    main()