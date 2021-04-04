import numpy as np
import matplotlib.pyplot as plt
from funciones.utilidades import graficarMatriz2D

"""
Analizando el efecto de las matrices sobre un circulo unitario

"""

I = np.eye(2)
print("Matriz identidad\n", I, "\n")

print("Grafico 1: ")
graficarMatriz2D(I)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()


A = np.array([[3, 7], [5, 2]])
print("Matriz a usar\n", A, "\n")

print("Grafico 2: ")
graficarMatriz2D(A)

plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()