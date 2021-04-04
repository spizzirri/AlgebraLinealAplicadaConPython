import numpy as np
import matplotlib.pyplot as plt
from funciones.utilidades import graficarMatriz2D, graficarVectores 
from modelos.colores import bcolors

"""
En la descomposicion SVD se obtienen tres matrices, S, V y D.
La matriz D es una matriz diagonal compuesta por los valores singualares.

Veamos cual es el efecto que estos valores singulares tiene en la transformacion

Recordamos que A = V * D * S
"""

print(f"{bcolors.VIOLETA} Analisis de los valores singulares")
print(f"========================================={bcolors.FIN}")

A = np.array([[3, 7], [5, 2]])
print("Matriz a usar: \n", A, "\n")

print(f"{bcolors.AZUL}Obtenemos la descomposicion S, V, D")
print(f"===================================\n{bcolors.FIN}")
U, D, V = np.linalg.svd(A)

print(f"{bcolors.VERDE}Matriz diagonal - Valores singulares{bcolors.FIN}")

print("D[0]: ", D[0], "\nD[1]: ", D[1], "\n")

"""
 Dado matriz D de 2x2 y matriz U de 2x2
 Hacemos el producto entre matrices

 DxU = | D[0]   0  | * | U[0, 0] U[0, 1] |
       | 0    D[1] |   | U[1, 0] U[1, 1] |

 DxU = | D[0]*U[0, 0]  D[0]*U[0, 1] | 
       | D[1]*U[1, 0]  D[0]*U[1, 1] |

"""

diagD = np.diag(D)
DxU = diagD.dot(U)

print(f"{bcolors.VERDE}Producto DxU\n{bcolors.FIN}")
print(DxU)

print(f"{bcolors.VERDE}Matriz original A traspuesta\n{bcolors.FIN}")
print(A.T)

print(f"{bcolors.AZUL} Graficamos")
graficarMatriz2D(A)
graficarVectores(DxU, cols=['red', 'blue'])
plt.text(3, 5, r"$DxU_0$", size = 18)
plt.text(7, 2, r"$DxU_1$", size = 18)

plt.text(-5, -4, r"$D_0$", size = 18)
plt.text(-4, 1, r"$D_1$", size = 18)

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()

