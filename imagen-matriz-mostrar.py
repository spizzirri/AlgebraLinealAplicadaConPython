import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from modelos.colores import bcolors

# Indica a matplotlib que quiere usar los valores por defectos que se tenian en la version 1.x
plt.style.use('classic')

print(f"{bcolors.VIOLETA} Abriendo una imagen con la libreria PIL")
print(f"===================================================\n{bcolors.FIN}")

image = Image.open(".\\imagenes\\imagen_ejemplo_frida_bredesen.jpg")

#Convertimos la imagen a escapa de grises y transparencia
imagen_gris = image.convert('LA')

# Abrimos la imagen
imagen_gris.show()

print(f"{bcolors.AZUL} Imprimimos el objeto con la imagen en escala de grises{bcolors.FIN}")
print(imagen_gris, "\n")

print(f"{bcolors.AZUL} Obtenemos la imagen como vector, pero solo con informacion del color rojo{bcolors.FIN}")

#De la imagen solo tomamos los datos de la banda 0 y le pedimos que guarde los datos como float
# RGB --> band = 0 es R, band = 1 es G y band = 2 es B 
imagen_matriz = np.array(list(imagen_gris.getdata(band=0)), float)
print(imagen_matriz, "\n")

print(f"{bcolors.AZUL}Transformamos el vector en una matriz{bcolors.FIN}")

filas = imagen_gris.size[1]
columnas = imagen_gris.size[0]
print("Filas: ", filas, " Columnas: ", columnas, "\n")

imagen_matriz.shape = (filas, columnas)
print("Matriz: \n", imagen_matriz, "\n")
print("Dimensiones: ", imagen_matriz.shape, "\n")
plt.imshow(imagen_matriz)
plt.show()

print(f"{bcolors.AZUL}Dividimos los valores de la matriz por 10{bcolors.FIN}")
imagen_matriz_div_10 = imagen_matriz / 10
print("Matriz: \n", imagen_matriz_div_10, "\n")
plt.imshow(imagen_matriz_div_10)
plt.show()

print(f"{bcolors.AZUL}Mostramos la imagen usando la matriz dividida en diez para mostrar la imagen en escala de grises{bcolors.FIN}")
print(f"Podemos ver que la imagen se ve perfectamente porque la relacion entre los valores no se perdio")
print(f"Como sabemos que los valores van de 0 a 255 podemos dividir la matriz original por 255 para obtener valores entre 0 y 1\n")
plt.imshow(imagen_matriz_div_10, cmap='gray')
plt.show()