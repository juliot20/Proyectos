import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from statistics import mean, stdev
import numpy as np
import warnings
from scipy.optimize import differential_evolution
xdatos = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28])
ydatos = np.array([1, 8, 14, 27, 36, 43, 55, 69, 86, 109, 137, 200, 245, 313, 345, 443, 558, 674, 786, 901, 989, 1075,
                   1181, 1317, 1475, 1673, 1801, 1988, 2100])
def sigmoide(x, a, b, c,d):
    y = a*np.tanh(b*(x+c))+d
    return y
#Función para minimizar el error que se usara en el algoritmo genetico
def sumaErroresCuadrado(parameterTuple):
    warnings.filterwarnings("ignore")
    val = sigmoide(xdatos, *parameterTuple)
    return np.sum((ydatos - val) ** 2.0)
def parametrosIniciales():
    maxX = max(xdatos)
    minX = min(xdatos)
    maxY = max(ydatos)
    minY = min(ydatos)
    parametros =[]
    parametros.append([minY, maxY]) #Buscar cota para la amplitud (a)
    parametros.append([minX, maxX]) # Buscar cota para b
    parametros.append([minX, maxX]) #Busca cota para c
    parametros.append([-1, 1])

    resultado = differential_evolution(sumaErroresCuadrado, parametros, seed=3)
    return resultado.x

plt.plot(xdatos, ydatos, 'b-', label='datos')
p = parametrosIniciales()
#Ajuste de los parametros
popt, pcov = curve_fit(sigmoide, xdatos, ydatos, p)
print(popt)
prediccion = sigmoide(xdatos, *popt)

MSE = mean_squared_error(prediccion, ydatos) #Error cuadratico medio
RMSE = np.sqrt(MSE) #Raiz cuadrada del error cuadratico medio
Rcuadrado = 1.0 - (np.var(prediccion-ydatos) / np.var(ydatos)) #Coeficiente de determinacion
print('MSE', MSE)
print('RMSE', RMSE)
print('Coeficiente de determinacion', Rcuadrado)
plt.plot(xdatos, prediccion, 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
plt.xlabel('Dias')
plt.ylabel('Numero de infectados')
plt.legend()
plt.show()
plt.plot(np.linspace(0, 300, 400), sigmoide(np.linspace(0, 300, 400), *popt))
plt.xlim((0, 100))
plt.ylim((0, 4000))
plt.xlabel('Dias')
plt.ylabel('Numero de infectados')
plt.show()
def casos(m):
    y = np.empty([len(m)])
    y[0] = ydatos[0]
    for i in range(1, len(m)):
        y[i] = m[i] - m[i-1]
    return y
y = casos(ydatos)
suma = 0
media = mean(y)
desviacion = stdev(y)
for t in y:
    suma = suma + (t-media)**4

Q = suma/((len(y))*(desviacion**4))
print('Curtosis', Q)
if (Q < 3):
    print('Es Platicútica')
elif(Q == 3):
    print('Es mesocútica')
else:
    print('Es leptocúrtica')