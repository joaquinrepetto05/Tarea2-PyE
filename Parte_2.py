import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cauchy

def uniforme_pseudoaleatorio(semilla, a, c, m, n):
    """
    Genera una muestra de números pseudoaleatorios con distribución uniforme en [0, 1].

    Parámetros:
        semilla (int): Semilla inicial.
        a (int): Multiplicador.
        c (int): Incremento.
        m (int): Módulo.
        n (int): Tamaño de la muestra.

    Retorna:
        numpy.ndarray: Muestra de números pseudoaleatorios uniformes en [0, 1].
    """
    muestra = np.zeros(n)
    z = semilla
    for i in range(n):
        z = (a * z + c) % m
        muestra[i] = z / m
    return muestra

def cauchy_estandar_pseudoaleatorio(muestra_uniforme):
    """
    Genera una muestra de números pseudoaleatorios con distribución de Cauchy estándar.

    Parámetros:
        muestra_uniforme (numpy.ndarray): Muestra de números pseudoaleatorios uniformes en [0, 1].

    Retorna:
        numpy.ndarray: Muestra de números pseudoaleatorios con distribución de Cauchy estándar.
    """
    return np.tan(np.pi * (muestra_uniforme - 0.5))

# Parámetros del generador congruencial lineal
semilla = 12345
a = 1664525
c = 1013904223
m = 2**32
n = 100

# Generar muestra uniforme
muestra_uniforme = uniforme_pseudoaleatorio(semilla, a, c, m, n)

# Generar muestra de Cauchy estándar
muestra_cauchy = cauchy_estandar_pseudoaleatorio(muestra_uniforme)

# Visualización de la distribución uniforme
plt.figure(figsize=(8, 6))
plt.hist(muestra_uniforme, bins='auto', density=True, alpha=0.6, color='b', label='Histograma Uniforme')
sns.kdeplot(muestra_uniforme, color='r', label='Densidad por Núcleos Uniforme')
plt.title('Muestra de Variables Aleatorias Uniformes en [0, 1]')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.show()

# Visualización de la distribución de Cauchy estándar
x = np.linspace(-5, 5, 1000)
densidad_teorica_cauchy = cauchy.pdf(x)

plt.figure(figsize=(8, 6))
plt.hist(muestra_cauchy, bins='auto', density=True, alpha=0.6, color='g', label='Histograma Cauchy')
sns.kdeplot(muestra_cauchy, color='m', label='Densidad por Núcleos Cauchy')
plt.plot(x, densidad_teorica_cauchy, 'r--', label='Densidad Teórica de Cauchy')
plt.title('Muestra de Variables Aleatorias con Distribución de Cauchy Estándar')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.xlim(-5, 5)
plt.legend()
plt.grid(True)
plt.show()
