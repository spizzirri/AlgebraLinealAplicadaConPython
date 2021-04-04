import numpy as np
from modelos.colores import bcolors

"""
    | Matriz        |   | Matriz Diagonal |   | Inversa de    |
A = | Auto vectores | * | con             | * | Matriz        |
    |               |   | Auto valores    |   | Auto vectores |

1) Los Auto valores los llamamos lambda en la matriz diagonal
2) Una matriz tiene inversa si es cuadrada y ademas su determinante es distinto de 0
- Cuando una matriz tiene inversa se llama inversible o regular.
- Cuando una matriz no tiene inversa es llama singular.
- Cuando una matriz es simetrica su inversa deja a la matriz sin cambios.
- Cuando una matriz es simetrica podemos obtener su inversa calculando la traspuesta, 
lo mismo pasa para la matriz inversa de auto vectores.
- El determinante (Det) de una matriz es un numero obtenido al sumar todos los diferentes
productos de n elementos que se pueden formar con los elementos de dicha matriz, de modo
que en cada producto figuren un elemento de cada distinta fila y uno de cada distinta columna.
A cada producto se le asigna el signo (+) si la permutacion de los subindices de filas es del
mismo orden que la permutacion de los subindices de columnas, y signo (-) si son de distinto
orden

Ejemplo de descomposicion de matriz cuadrada no simetrica:

A = | 3  2 |  DetA = 3*1 + (-4*2) = 3 - 8 = -5
    | 4  1 |

Autovalores = 5 y -1
Autovectorres = [0.7071..., -0.447...] y [0.7071..., 0.8944...]

A = | 0.7071..., -0.4472... | * | 5  0 | * ( | 0.7071..., -0.4472... | )-1
    | 0.7071...,  0.8944... |   | 0 -1 |   ( | 0.7071...,  0.8944... | )

"""

print(f"{bcolors.VIOLETA}*************************************")
print(f"Descomposicion de matrices cuadradas")
print(f"*************************************{bcolors.FIN}")

print("Matriz a usar\n")
A = np.array([[3, 2], [4, 1]])

print(A, "\n")

print(f"{bcolors.AZUL}Obtengo los auto valores y auto vectores")
print(f"========================================================\n{bcolors.FIN}")
autovalores, autovectores = np.linalg.eig(A)

print("autovalores: \n", autovalores, "autovectores: \n", autovectores)

print(f"{bcolors.AZUL}Matriz auto vectores * Matriz diagonal auto valores")
print(f"====================================================================\n{bcolors.FIN}")
A_calc = autovectores.dot(np.diag(autovalores))

print("A_calc: \n", A_calc, "\n")

print(f"{bcolors.AZUL}Matriz auto vectores * Matriz diagonal auto valores * Matriz inversa auto vectores")
print(f"==================================================================================================\n{bcolors.FIN}")
A_calc = A_calc.dot(np.linalg.inv(autovectores))

print("A_calc: \n", A_calc, "\n\n")