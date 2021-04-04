import numpy as np
from modelos.colores import bcolors

"""
Resolver el sistema usando una pseudo inversa

y =  1 * x + 4 => -1*x + y = 4 
y =  2 * x + 5 => -2*x + y = 5
y = -3 * x + 6 =>  3*x + y = 6

Tomamos los coeficientes y generamos una matriz

"""
matriz = np.array([[-1, 1], [-2, 1], [3, 1]])
print("Matriz a usar\n", matriz, "\n")

matriz_pse = np.linalg.pinv(matriz)
print("Pseudo inversa obtenida con linalg.pinv\n", matriz_pse, "\n")

print(f"{bcolors.AZUL}Definimos nuestro vector solucion\n{bcolors.FIN}")
b = np.array([[4], [5], [6]])
print("Vector a usar\n", b, "\n")

print(f"{bcolors.AZUL}Aplicamos el producto entre la matriz pseudo y el vector\n{bcolors.FIN}")
resultado = matriz_pse.dot(b)
print("Resultado: \n", resultado, "\n")

# Calcular los auto valores y auto vectores
X =  np.array([[3, 4], [3, 2]])
autovalores, autovectores = np.linalg.eig(X)
print(autovectores)
print(autovalores)


# Calcular la matriz pseudoinversa
X = np.array([[1, 2],[3, 4],[5, 6]])
moore = np.linalg.pinv(X)
print(moore)