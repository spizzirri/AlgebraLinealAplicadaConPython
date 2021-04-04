"""
Norma de un vector
==================
Dado un vector de un espacio vectorial euclídeo, 
la norma de un vector es definida como la distancia 
(en línea recta) entre dos puntos A y B que delimitan al vector

Determinante
============
Llamamos determinante de A,  det A,  al número obtenido al sumar 
todos los diferentes productos de n elementos que se pueden formar con los elementos de dicha matriz, 
de modo que en cada producto figuren un elemento de cada distinta fila y uno de cada distinta columna

"""

import numpy as np
import matplotlib.pyplot as plt
from funciones.utilidades import graficarVectores

A = np.array([[-1, 3], [2, -2]])
vector = np.array([[2], [1]])
vector_transformado = A.dot(vector)

print("A:\n", A)
print("A flatten:\n", A.flatten())
print("vector:\n", vector)
print("vector flatten:\n", vector.flatten())
print("vector transformado\n", vector_transformado)
print("vector transformado flatten\n", vector_transformado.flatten())

print("Determinante de A:\n", np.linalg.det(A))
print("Normal de A:\n", np.linalg.norm(vector))
print("Normal vector transformado:\n", np.linalg.norm(vector_transformado))

graficarVectores([vector.flatten(), vector_transformado.flatten()], cols=['blue', 'red'])

plt.xlim(-0.5, 3)
plt.ylim(-0.5, 2)

plt.show()