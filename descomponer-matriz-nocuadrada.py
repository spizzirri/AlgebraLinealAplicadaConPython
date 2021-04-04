import numpy as np
from modelos.colores import bcolors

"""
Usaremos la tecnica de 'Descomposicion en valores singulares'

    | Matriz con          |
U = | vectores izquierdos |
    | singulares          |

    | Matriz con        |
V = | vectores derechos |
    | singulares        |

    | Matriz diagonal | 
D = | con valores     |
    | singulares      |

U y V son matrices ortogonales.
D es una matriz diagonal.

Una matriz ortogonal es aquella que tiene todos sus vectores ortogonales
Una matriz es ortogonal si: 
    A traspuesta * A = Identidad 
    es decir 
    A traspuesta = inversa de A.


"""

print(f"{bcolors.VIOLETA}***************************************")
print(f"Descomposicion de matrices NO cuadradas")
print(f"***************************************{bcolors.FIN}")

print("Matriz a usar\n")
A = np.array([[1,2,3], [3,4,5]])
print(A, "\n")

print(f"{bcolors.AZUL}Obtenemos las matrices U, V y D")
print(f"===============================\n{bcolors.FIN}")
U, D, V = np.linalg.svd(A)
print("U: \n", U, "\n\nD: \n", D, "\n\nV:\n", V, "\n\n")

print(f"{bcolors.VERDE} Calculamos la matriz diagonal con los valores de D")
print(f"=======================================================\n{bcolors.FIN}")
diagonalD = np.diag(D).dot(np.eye(len(U), len(V)))
print("Matriz diagonal: \n", diagonalD, "\n\n")


print(f"{bcolors.AZUL} Multiplicamos U * D * V")
print(f"==========================\n{bcolors.FIN}")
print("Matriz A: \n", U.dot(diagonalD).dot(V), "\n\n")