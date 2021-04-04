import matplotlib.pyplot as plt
import numpy as np

def graficarVectores(vecs, cols, alpha=1):
    plt.axvline(x=0, color='grey', zorder=0)
    plt.axhline(y=0, color='grey', zorder=0)

    for i in range(len(vecs)):
        x = np.concatenate([[0, 0], vecs[i]])
        plt.quiver( [x[0]],
                    [x[1]],
                    [x[2]],
                    [x[3]],
                    angles = 'xy', 
                    scale_units='xy',
                    scale=1,
                    color=cols[i],
                    alpha=alpha )

def graficarMatriz2D(matriz, vectorCol=['red', 'blue']):

    #circulo unitario
    x = np.linspace(-1, 1, 100000)
    """
    Circulo: (x-a)^2 + (y-b)^2 = r^2

    a y b son 0 en este caso porque el circulo esta centrado en el 0,0
    """
    y = np.sqrt(1-(x**2))

    #Circulo unitario transformado
    """
     A = | a1 a2 | * | x1 x2 x3 .... xN |
         | a3 a4 |   | y1 y2 y3 .... yN |

     A = | (a1 * x1 + a2 * y1) (a1 * x2 + a2 * y2) .... (a1 * xN + a2 * yN) |
         | (a3 * x1 + a4 * y1) (a3 * x2 + a4 * y2) .... (a3 * xN + a4 * yN) |
    """
    #Basicamente lo que se esta haciendo es una multiplicacion entre matrices
    # Valores de X e Y parte positiva del circulo
    x1 = matriz[0, 0]*x + matriz[0, 1]*y
    y1 = matriz[1, 0]*x + matriz[1, 1]*y

    #Basicamente lo que se esta haciendo es una multiplicacion entre matrices
    # Valores de X e Y parte negativa del circulo
    x1_neg = matriz[0, 0]*x - matriz[0, 1]*y
    y1_neg = matriz[1, 0]*x - matriz[1, 1]*y

    #Vectores a graficar
    u1 = [matriz[0, 0], matriz[1, 0]]
    v1 = [matriz[0, 1], matriz[1, 1]]

    graficarVectores([u1, v1], cols=[vectorCol[0], vectorCol[1]])

    #Dibujamos el circulo tambien
    plt.plot(x1, y1, 'green', alpha=0.7)
    plt.plot(x1_neg, y1_neg, 'green', alpha=0.7)