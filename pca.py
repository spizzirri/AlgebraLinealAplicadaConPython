import numpy as np
import matplotlib.pyplot as plt
from modelos.colores import bcolors
from funciones.utilidades import graficarVectores

"""
Analisis de componentes principales (PCA) 
-----------------------------------------

Es una tecnica muy util para reducir la cantidad de dimensiones
con las que estamos trabajando.

Muchas veces tenemos un conjunto de dimensiones muy grandes y lo que
necesitamos es reducirlo para quedarnos con el 80% de la informacion 
que contiene nuestro conjunto de datos.

La pregunta es: Con que variables debemos quedarnos para lograr esto??
"""

print(f"{bcolors.VIOLETA}Analisis de componentes principales{bcolors.FIN}")

#Semilla para generar numeros aleatorios
np.random.seed(42)

#rand: Distribucion Uniforme
print(f"{bcolors.AZUL}Generamos una distribucion uniforma de valores (X){bcolors.FIN}")
x = 3*np.random.rand(200)

#randn: Distribucion Normal
print(f"{bcolors.AZUL}Generamos una distribucion normal de valores (Y){bcolors.FIN}")
y = 20*x + 2*np.random.randn(200)

# Reacomodamos los elementos X e Y en una arreglo de N dimensiones
print(f"{bcolors.AZUL}Los valores generados los transformamos a en un arreglo{bcolors.FIN}")
x = x.reshape(200, 1)
y = y.reshape(200, 1)

# Combinamos x e y en un solo arreglo
print(f"{bcolors.AZUL}Combinamos los arreglos X e Y en una misma matriz\n{bcolors.FIN}")
xy = np.hstack([x, y])
print("Dimensiones de la matriz", xy.shape, "\n")

print(f"{bcolors.AZUL}Graficamos la matriz{bcolors.FIN}")
todasLasFilasDeLaColumnaCero = xy[:, 0] # X
todasLasFilasDeLaColumnaUno = xy[:, 1] # Y
plt.plot(todasLasFilasDeLaColumnaCero, todasLasFilasDeLaColumnaUno, '.')
plt.show()

print(f"{bcolors.AZUL}Giramos nuestro sistema de referencia (los vectores generadores){bcolors.FIN}")
print(f"{bcolors.AZUL}De esta manera nos queda el grafico centrado en 0,0 y vemos mejor la varianza{bcolors.FIN}")
xy_centrado = xy - np.mean(xy, axis = 0)
todasLasFilasDeLaColumnaCeroCentrada = xy_centrado[:, 0] # X
todasLasFilasDeLaColumnaUnoCentrada = xy_centrado[:, 1] # Y
plt.plot(todasLasFilasDeLaColumnaCeroCentrada, todasLasFilasDeLaColumnaUnoCentrada, '.')
plt.show()

"""
Para poder encontrar las componentes principales debemos resolver un problema.
Estas componentes son las que buscan un D tal que nos da el maximo de:
 la traza de ( D traspuesto por X traspuesto por X*D )
Como es un problema de maximizacion debemos ademas pedir que el resultado tenga norma 1
Esto quiere decir que D traspuesto * D es igual a 1

"""

print(f"{bcolors.AZUL}Calculamos los autovalores y autovectores usando nuestra matriz con los datos centrados\n{bcolors.FIN}")
autovalores, autovectores = np.linalg.eig(xy_centrado.T.dot(xy_centrado))
print("Auto valores: \n", autovalores, "\nAuto vectores: \n", autovectores, "\n")

"""
Con estos auto vectores vemos que son los que maximizan nuestra funcion
Cada columna esta asociada con una auto valores.
El vector asociado con el auto valor mas grande nos dice la direccion de maxima varianza

Queremos ver ahora como es que estos autovectores estan relacionados con la direccion en la cual
se mueven nuestros valores.

"""

print(f"{bcolors.AZUL}Graficamos los auto vectores{bcolors.FIN}")
graficarVectores(autovectores.T, ['blue', 'red'])

"""
En el grafico hay un auto vector que nos quedo con una amplitud muy chica, nos conviene
amplificarlo ya que no importa su amplitud sino cual esta relacionado con el autovalor mas grande.
Es el autovalor el que define cual es la direccion que contiene mas informacion

Para amplificarlo, lo dividimos por 10
"""
#plt.plot(todasLasFilasDeLaColumnaCeroCentrada, todasLasFilasDeLaColumnaUnoCentrada, '.')
plt.plot(todasLasFilasDeLaColumnaCeroCentrada, todasLasFilasDeLaColumnaUnoCentrada/20, '.')
plt.show()

"""
Observamos que la mayor varianza se produce en el eje Y ya que es el valor
asociado al segundo vector, el rojo

Si rotamos el grafico verticalmente conservaremos muchos mas datos que si lo hacemos en el eje X
"""

print(f"{bcolors.AZUL}Graficamos con un nuevo sistema de referencias{bcolors.FIN}")

xy_nuevo = autovectores.T.dot(xy_centrado.T)
todasLasColumnasDeLaFilaCero = xy_nuevo[0, :]
todasLasColumnasDeLaFilaUno = xy_nuevo[1, :]

plt.plot(todasLasColumnasDeLaFilaCero, todasLasColumnasDeLaFilaUno, '.')
plt.show()
