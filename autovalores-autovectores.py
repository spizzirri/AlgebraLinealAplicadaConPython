import matplotlib.pyplot as plt
import numpy as np
from funciones.utilidades import graficarVectores

"""
Las transformaciones lineales ejercen trasformaciones sobre nuestros vectores. 
Un auto vector es cuando a un vector le aplicamos la transformación y este no sufre 
cambios en su direccion.
"""

X =  np.array([[3, 2], [4, 1]])
print(X)

# metodo que devuelvo los auto valores y autovectores
autovalores, autovectores = np.linalg.eig(X)
print("AUTO VALORES:\n", autovalores)
print("AUTO VECTORES:\n", autovectores)

v = np.array([[-1], [2]])
Xv = X.dot(v)
v_np = autovectores[:, 1]

graficarVectores([Xv.flatten(), v.flatten(), v_np], cols=["green", "orange", "blue"])

plt.xlim(-2, 2)
plt.ylim(-3, 3)

plt.show()
