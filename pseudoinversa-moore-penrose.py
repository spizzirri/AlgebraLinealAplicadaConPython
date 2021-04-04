import numpy as np
from modelos.colores import bcolors

"""
La pseudoinversa de Moore Penrose es una aplicacion directa de SVD que nos permite
resolver sistemas de ecuaciones lineales

Sistema de ecuaciones lineales:

Ax = B, si existe solucion entonces => x = A^(-1)B

Pero que ocurre si no existe A^(-1)??

Podemos buscar una matriz que al ser multiplicada por A nos de un valor muy cercano
a la matriz identidad. Una matriz pseudoinversa.

Lo que necesitamos es crear una nueva matriz D pseudo
- Debe tener las mismas dimensiones que nuestra matriz A pero traspuesta

A(pseudoinv) = V traspuesta * D(pseudo) * U traspuesta

- La pseudo inversa, en caso de existir, es unica

"""
# Pedimos que los numeros muy cercanos a 0 se muestren como cero
np.set_printoptions(suppress=True)

print(f"{bcolors.VIOLETA}Calculando una pseudoinversa")
print(f"============================{bcolors.FIN}\n")

A = np.array([[2, 3], [5, 7], [11, 13]])
print("Matriz a usar: \n", A, "\n")

print(f"{bcolors.AZUL}Obtenemos la descomposicion SVD\n{bcolors.FIN}")
U, D, V = np.linalg.svd(A)
print("U\n", U, "\n\nD\n", D, "\n\nV\n", V)

print(f"{bcolors.AZUL}Creando la matriz D pseudo\n{bcolors.FIN}")
print(f"{bcolors.VERDE}Le asignamos las mismas dimensiones de A traspuesta{bcolors.FIN}")
D_pse = np.zeros(( A.shape[0], A.shape[1] )).T

print("D_pse :\n", D_pse, "\n")
print("Valores de D_pse que se van a reemplazar: \n", D_pse[:D.shape[0], :D.shape[0]], "\n")
print("Valores de D que pondremos en D_pse: \n", np.linalg.inv(np.diag(D)), "\n")

D_pse[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))
print("D_pse: \n", D_pse, "\n")

print(f"{bcolors.AZUL}Creando la matriz A pseudo inversa\n{bcolors.FIN}")
A_pse = V.T.dot(D_pse).dot(U.T)
print("A pseudo inversa: \n", A_pse, "\n")

print(f"{bcolors.AZUL}Creando la matriz A pseudo inversa usando el metodo pinv de linalg\n{bcolors.FIN}")
A_pse_con_pinv = np.linalg.pinv(A)
print("A pseudo inversa calculado con pinv: \n", A_pse_con_pinv, "\n")

print(f"{bcolors.AZUL}Verificamos que la matriz que calculamos nos devuelva una matriz cercana a la identidad\n{bcolors.FIN}")
I = A_pse.dot(A)
print("I :\n", I, "\n")

print(f"{bcolors.AZUL}Creando la matriz A pseudo inversa usando A traspuesta\n{bcolors.FIN}")
A_pse_con_A_traspuesta = np.linalg.inv(A.T.dot(A)).dot(A.T)
print("A_pse_con_A_traspuesta :\n", A_pse_con_A_traspuesta, "\n")
