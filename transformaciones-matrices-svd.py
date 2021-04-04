import numpy as np
import matplotlib.pyplot as plt
from funciones.utilidades import graficarMatriz2D
from modelos.colores import bcolors

"""
Analizando el efecto de las matrices de la descomposicion SVD sobre un circulo unitario

La matriz D es la que aumenta, amplifica o reduce nuestro espacio. (ESCALA)

"""

A = np.array([[3, 7], [5, 2]])
print("Matriz a usar\n", A, "\n")

print(f"{bcolors.AZUL} Obtenemos las matrices S, V y D")
print(f"============================={bcolors.FIN}\n")

U, D, V = np.linalg.svd(A)
print("U: \n", U, "\nV: \n", V, "\nD :\n", D, "\n\n")

print(f"{bcolors.AZUL}Circulo unitario")
print(f"============================={bcolors.FIN}\n")

I = np.eye(2)

print("Grafico usando la identidad: \n", I, "\n\n")
graficarMatriz2D(I)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

print(f"{bcolors.AZUL}Primer rotacion")
print(f"=============================\n{bcolors.FIN}")

print("Grafico usando V: \n", V, "\n\n")

graficarMatriz2D(V)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

print(f"{bcolors.AZUL}Escalamiento")
print(f"=============================\n{bcolors.FIN}")

diagDdotV = np.diag(D).dot(V)

print("Grafico usando DxV: \n", diagDdotV, "\n\n")
graficarMatriz2D(diagDdotV)

plt.xlim(-9, 9)
plt.ylim(-9, 9)
plt.show()

print(f"{bcolors.AZUL}Segunda rotacion")
print(f"============================={bcolors.FIN}\n")

rotacionFinal = U.dot(np.diag(D).dot(V))

print("Grafico usando SxDxV: \n",rotacionFinal ,"\n\n")
graficarMatriz2D(rotacionFinal)

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()
