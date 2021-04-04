"""
Analisis de componentes principales (PCA) 
-----------------------------------------

Es una tecnica muy util para reducir la cantidad de dimensiones
con las que estamos trabajando.

Muchas veces tenemos un conjunto de dimensiones muy grandes y lo que
necesitamos es reducirlo para quedarnos con el 80% de la informacion 
que contiene nuestro conjunto de datos.

La pregunta es: Con que variables debemos quedarnos para lograr esto??

Veamos un ejemplo aplicado a imagenes
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt
import pandas as pd
from modelos.colores import bcolors
from glob import iglob

im = imageio.imread(".\\imagenes\\olivetti_faces\\s3\\3.pgm")
im = im.astype(np.uint8)

print(f"{bcolors.AZUL}Mostramos la matriz de la imagen cargada{bcolors.FIN}")
print("Matriz :\n", im, "\n")
print("Maximo original ", np.max(im))
print("Minimo original ", np.min(im), "\n")

print(f"{bcolors.AZUL}Normalizamos los valores para que esten entre 0 y 1{bcolors.FIN}")
matriz_normalizada = im / 255
print("Matriz :\n", matriz_normalizada, "\n")
print("Maximo original ", np.max(matriz_normalizada))
print("Minimo original ", np.min(matriz_normalizada), "\n")

print(f"{bcolors.AZUL}Visualizemos las imagenes para comprobar que esten iguales{bcolors.FIN}")
fix, ax = plt.subplots(1, 2, figsize=(12, 12), subplot_kw= { 'xticks': [], 'yticks': [] })

ax[0].imshow(im, cmap='gray')
ax[1].imshow(matriz_normalizada, cmap='gray')
plt.show()

print(f"{bcolors.AZUL}Importamos todas las imagenes{bcolors.FIN}")
caras = pd.DataFrame([])
for path in iglob('.\\imagenes\\olivetti_faces\\*\\*.pgm'):
    im = imageio.imread(path)
    cara = pd.Series(im.flatten(), name=path)
    caras = caras.append(cara)

fix, axes = plt.subplots(5, 10, figsize=(15, 8), 
                         subplot_kw= { 'xticks': [], 'yticks': [] }, 
                         gridspec_kw=dict(hspace = 0.01, wspace=0.01))

for i, ax in enumerate(axes.flat):
    imagen = caras.iloc[i].values.reshape(112, 92) 
    ax.imshow(imagen, cmap="gray")

plt.show()