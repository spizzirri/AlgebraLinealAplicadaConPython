import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from modelos.colores import bcolors

"""
Veamos como aplicar la descomposicion SVD a una imagen y que efecto tendria sobre su reconstruccion

Los valores singulares, los de las matriz D, estan ordenados de mayor a menor en orden de cantidad 
de informacion que aportan para generar la imagen
"""

# Indica a matplotlib que quiere usar los valores por defectos que se tenian en la version 1.x
plt.style.use('classic')

image = Image.open(".\\imagenes\\imagen_ejemplo_frida_bredesen.jpg")
image.show()

#Convertimos la imagen a escapa de grises y transparencia
imagen_gris = image.convert('LA')

#De la imagen solo tomamos los datos de la banda 0 y le pedimos que guarde los datos como float
# RGB --> band = 0 es R, band = 1 es G y band = 2 es B 
imagen_matriz = np.array(list(imagen_gris.getdata(band=0)), float)

# Le asignamos las dimensiones para que sea una matriz en lugar de un vector
print(f"{bcolors.AZUL}Dimensiones de la imagen{bcolors.FIN}")
filas = imagen_gris.size[1]
columnas = imagen_gris.size[0]
imagen_matriz.shape = (filas, columnas)
print("Filas: ", filas, " Columnas: ", columnas)
print("Matriz shape: ", imagen_matriz.shape, "\n")

# Hacemos la descomposicion SVD
print(f"{bcolors.AZUL}Descomposicion SVD{bcolors.FIN}")
U, D, V = np.linalg.svd(imagen_matriz)

print("Dimensiones")
print("U: ", U.shape, "\nD: ", D.shape, "\nV: ", V.shape, "\n")


print(f"{bcolors.AZUL}Reconstruyendo la imagen{bcolors.FIN}")
for i in [1, 25, 50, 150]:
    imagen_reconstruida = np.matrix(U[:, :i]) * np.diag(D[:i]) * np.matrix(V[:i, :])
    plt.imshow(imagen_reconstruida, cmap='gray')
    plt.title("Valores singulares = %s" %i)
    plt.show()