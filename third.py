#Acá la neurona aprende a prueba multiplicar... poca precision
import numpy as np
import math as math

# X = np.zeros((4, 2))

Y = np.zeros((4, 1))

X = np.array([[1, 1, 1], [1, 2, 1], [1, 2, 2], [1, 3, 2],[1, 2, 7]])
T = np.array([[1, 5, 2], [1, 9, 2], [1, 8, 2], [1, 15, 2]])
Y = np.array([1, 2, 4, 6, 14])

W = np.random.rand(3)
#W = [0, 0, 0]
synapsis = np.zeros((4, 1))
# axion = []
er = 0
print('X: ')
print(X)

# print(Y)
print('W: ')
print(W)


#Multiplicax*w; es la entrada asumiendo un peso; puede aplicarse normalización
# de ser necesario
def dendrita(x):
    n = np.multiply(x, W)
    return n


#Suma los resultados features x; este resultado nos dice el coeficiente intelectual de la neurona
#, gracias al factor de error; nos permite calibrar su inteligencia.
# Tratando de minimizar la sumatoria real contra la de la sumatoria
def neuron(n):
    return np.sum(n)

#Es la funcion de activación de la neurona; dependiendo de la actividad puede ser modificada
#incluso puede aplicarse una regularización en esta sálida.
def axon(s):
    return s

#Pensar es decir usar la neurona
def think(x):
    d = dendrita(x)
#    print('dendrita: ', d)
    n = neuron(d)
#    print('neuron: ', n)
    a = axon(n)
    return a



# Ejecuta todas las muestras (lecciones) en una interacción; devuelve el error
# cuadrático medio; dependiendo de la cantidad de lecciones que tome la neurona; ajusta su calibraje.
# mejorando su coeficiente intelectual
def lesson(x,y):
    #print(x)
    return y - think(x)
        

# Este es el profesor de la neurona; regula las lecciones y le da una calificación por su trabajo.
# A medida que se entrena, la calificación es mayor.
# de acuerdo a su learning rate; es el tiempo que el trainer le da a la neurona para asimilar
# conocimientos.
def trainer(lessons, rate):
    for l in range(0, lessons):
        er = 0
        for i, training_set in enumerate(X):
            #print(training_set, Y[i])
            
             #print(t)
             grade = lesson(training_set,Y[i])
             #print(grade)
             if grade != 0:
                 er = er + 1
                 for j, t in enumerate(training_set):
                    W[j] = W[j] + ( (rate * grade) * t )
            
        if er==0:
            break



trainer(100000, 0.01)

print('W: ', W)
print('train 1: ', think(X[0]))
print('train 2: ', think(X[1]))
print('train 3: ', think(X[2]))
print('train 4: ', think(X[3]))
print('train 5: ', think(X[4]))

print('dev 1: ', think(T[0]))
print('dev 2: ', think(T[1]))
print('dev 3: ', think(T[2]))
print('dev 4: ', think(T[3]))