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
from sklearn.decomposition import PCA

print(f"{bcolors.AZUL}Importamos todas las imagenes{bcolors.FIN}")
caras = pd.DataFrame([])
for path in iglob('.\\imagenes\\olivetti_faces\\*\\*.pgm'):
    im = imageio.imread(path)
    cara = pd.Series(im.flatten(), name=path)
    caras = caras.append(cara)

# Probar el mismo codigo con 0.7, 0.8, 0.9 y 0.99
print(f"{bcolors.AZUL}Nos quedamos con los datos necesarios para tener el 50% de la informacion{bcolors.FIN}")
caras_pca = PCA(n_components = 0.5)

print(f"{bcolors.AZUL}Entrenamos{bcolors.FIN}")
caras_pca.fit(caras)


filas = 3
columnas = caras_pca.n_components_ // filas

fix, axes = plt.subplots(filas, columnas, figsize=(12, 6), 
                         subplot_kw= { 'xticks': [], 'yticks': [] }, 
                         gridspec_kw=dict(hspace = 0.01, wspace=0.01))

for i, ax in enumerate(axes.flat):
    imagen = caras_pca.components_[i].reshape(112, 92) 
    ax.imshow(imagen, cmap="gray")

plt.show()

print(f"{bcolors.VERDE}Cantidad de componentes{bcolors.FIN}")
print(caras_pca.components_, "\n")

print(f"{bcolors.AZUL}Calculamos las proyecciones{bcolors.FIN}")
componentes = caras_pca.transform(caras)
proyeccion = caras_pca.inverse_transform(componentes) 

fix, axes = plt.subplots(5, 10, figsize=(15, 8), 
                         subplot_kw= { 'xticks': [], 'yticks': [] }, 
                         gridspec_kw=dict(hspace = 0.01, wspace=0.01))

for i, ax in enumerate(axes.flat):
    imagen = proyeccion[i].reshape(112, 92) 
    ax.imshow(imagen, cmap="gray")

plt.show()