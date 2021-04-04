import numpy as np
import matplotlib.pyplot as plt
from modelos.colores import bcolors


"""
Un sistema de ecuaciones sea A*x = B, puede tener
- 0 soluciones 
- 1 solucion: --> 
    Tenemos inversa y ya sabemos calcularlo.
    Tenemos una matriz cuadrada y todos sus vectores son linealmente independientes.
- infinitas soluciones


Dado A*x = b
Queremos encontrar una Apseudo tal que la norma 2 de (A*x-b) sea minima
"""

# Definimos una rango de valores de X para evaluar el sistema
x = np.linspace(-5, 5, 1000)
"""
Tenemos dos variables, x e y, y tres ecuaciones.
Es un sistema sobre determinado
"""
print(f"{bcolors.AZUL} Sistema de ecuaciones\n{bcolors.FIN}")
print("y = -4*x + 3\ny =  2*x + 5\ny = -3*x + 1\n")

y_1 = -4*x +3
y_2 = 2*x + 5
y_3 = -3*x + 1

print(f"{bcolors.AZUL} Graficamos\n{bcolors.FIN}")
"""
Veamos el grafico del sistema
"""
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)

plt.xlim(-2, 2.5)
plt.ylim(-6, 6)

plt.plot()
plt.show()

print(f"{bcolors.AZUL} Despejamos para calcular la pseudoinversa\n{bcolors.FIN}")
"""
 En el grafico observamos que no hay un punto en el que las tres rectas se corten
 en un unico punto.
 Veamos sin con una pseudoinversa podemos encontrar un punto que minimize la norma 2

Despejamos:

y = -4*x + 3 =>  4*x + y = 3
y =  2*x + 5 => -2*x + y = 5
y = -3*x + 1 =>  3*x + y = 1

Tomamos los coeficientes y generamos una matriz

"""
matriz = np.array([[4, 1], [-2, 1], [3, 1]])
print("Matriz a usar\n", matriz, "\n")

matriz_pse = np.linalg.pinv(matriz)
print("Pseudo inversa obtenida con linalg.pinv\n", matriz_pse, "\n")

print(f"{bcolors.AZUL}Definimos nuestro vector solucion\n{bcolors.FIN}")
b = np.array([[3], [5], [1]])
print("Vector a usar\n", b, "\n")

print(f"{bcolors.AZUL}Aplicamos el producto entre la matriz pseudo y el vector\n{bcolors.FIN}")
resultado = matriz_pse.dot(b)
print("Resultado: \n", resultado, "\n")

print(f"{bcolors.AZUL} Graficamos otra vez\n{bcolors.FIN}")
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)

plt.xlim(-2, 2.5)
plt.ylim(-6, 6)

plt.scatter(resultado[0], resultado[1])

plt.show()