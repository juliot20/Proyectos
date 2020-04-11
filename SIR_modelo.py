import matplotlib.pyplot as plt
import numpy as np
N = 4000000
valores_iniciales = [N-8, 8, 0]
parametros = [1.75, 0.5]
t_maximo = 200
dt = 0.1
t = np.linspace(0, t_maximo, int(1+t_maximo/dt))
#Modelo clásico SIR
def modeloSIR(valores_iniciales, parametros, t, N):
    s0, i0, r0 = valores_iniciales
    s, i, r = [s0], [i0], [r0]
    beta, gamma = parametros
    dt = t[2]-t[1]
    for _ in t[1:]:
        s1 = - (beta*dt/N)*i[-1]*s[-1] + s[-1]
        i1 = ((beta/N)*s[-1]-gamma)*dt*i[-1] + i[-1]
        r1 = (gamma*dt)*i[-1] + r[-1]
        s.append(s1)
        i.append(i1)
        r.append(r1)
    return np.stack([s, i, r]).T
val = modeloSIR(valores_iniciales,parametros,t, N)

#Modelo SIR con cuarentena y distanciamiento social
#Considerando una función k(t) que tiene imagen en[0, 1]
#Sea k(t)={0.65, si t_0<=t<=t_1, 0}, esto representa una reducción de la tasa de infectados en un 65%
t_cuarentena = 30
def modeloSIRmodificado(valores_iniciales, parametros, t, t_cuarentena, N):
    s0, i0, r0 = valores_iniciales
    s, i, r = [s0], [i0], [r0]
    beta, gamma = parametros
    dt = t[2]-t[1]
    for tp in t[1:]:
        if((tp*dt) <= t_cuarentena):
            k = 0.35
        s1 = - (k*beta*dt/N)*i[-1]*s[-1] + s[-1]
        i1 = ((k*beta/N)*s[-1]-gamma)*dt*i[-1] + i[-1]
        r1 = (gamma*dt)*i[-1] + r[-1]
        s.append(s1)
        i.append(i1)
        r.append(r1)
        k = 1
    return np.stack([s, i, r]).T
val1 = modeloSIRmodificado(valores_iniciales,parametros,t, t_cuarentena, N)

maximo_val = max(val[:, 1])
maximo_val1 = max(val1[:, 1])
reduciendo = (abs(maximo_val-maximo_val1)*100)/maximo_val
reduciendo = round(reduciendo, 2)
print(f'Reduciendo un {reduciendo} % de infectados')
plt.figure(figsize=(12, 8))
plt.plot(val)
plt.ylabel('Fracción de la Población')
plt.legend(['Susceptible', 'Infectados', 'Recuperados'])
plt.xlabel('Días')
plt.show()
plt.figure(figsize=(12, 8))
plt.plot(val[:, 1])
plt.xlabel('Días')
plt.ylabel('Fracción de la Población (Infectados)')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(val1)
plt.ylabel('Fracción de la Población')
plt.legend(['Susceptible', 'Infectados', 'Recuperados'])
plt.xlabel('Días')
plt.show()
plt.figure(figsize=(12, 8))
plt.plot(val1[:, 1])
plt.xlabel('Días')
plt.ylabel('Fracción de la Población (Infectados)')
plt.show()
