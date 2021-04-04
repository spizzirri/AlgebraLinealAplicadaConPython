"""
Las transformaciones lineales ejercen trasformaciones sobre nuestros vectores. 
Un auto vector es cuando a un vector le aplicamos la transformación y este no sufre 
cambios en su direccion.
"""

import matplotlib.pyplot as plt
import numpy as np
from funciones.utilidades import graficarVectores

orange_light = '#FF9A13'
blue_light = '#1190FF'

X = np.array([[3, 2], [4, 1]])
print("X:\n", X)

v = np.array([[1], [1]])
print("v:\n", v)
s = np.array([[-1], [2]])
print("s:\n", s)

u = X.dot(v)
print("u:\n", u)
t = X.dot(s)
print("t:\n", t)

graficarVectores([u.flatten(), v.flatten()], cols=[orange_light, blue_light])
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.show()

graficarVectores([t.flatten(), s.flatten()], cols=[orange_light, blue_light])
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
