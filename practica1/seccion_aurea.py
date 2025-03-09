def seccion_aurea(function, search_interval, max_error=0.001):
   
    razon_aurea = 0.618
    max_iter = 1000

    a, b = search_interval[0], search_interval[1]

    i = 0

    while i < max_iter and (b - a) >= max_error:
        # Puntos posibles
        aux_1 = b- razon_aurea * (b - a)
        aux_2 = a + razon_aurea * (b - a)

        theta_1 = function(aux_1)
        theta_2 = function(aux_2)

        # Nuevo intervalo
        if theta_1 > theta_2:
            a = aux_1
        else:
            b = aux_2
        
        i += 1

    # Resultado: Punto medio del intervalo inicial
    theta_min = (a + b) / 2

    return theta_min


def main():
    theta = lambda lamb: (5 - 2 * lamb + (4 + lamb)**3)**2 + 2 * (-3 - 3 * lamb)**4
    search_interval = (-5, 5)
    
    min = seccion_aurea(theta, search_interval)

    print(min)

    return 0


if __name__ == "__main__":
    main()