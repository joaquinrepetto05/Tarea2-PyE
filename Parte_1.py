import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import poisson

# 1. Cargar datos
# Asegurarse de que el archivo 'cancelaciones.csv' esté en el mismo directorio o proporciona la ruta correcta.
try:
    df = pd.read_csv("cancelaciones.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'cancelaciones.csv'. Por favor, asegúrate de que el archivo esté en el mismo directorio o proporciona la ruta correcta.")
    exit()

# La columna se llama 'cancelaciones'
if 'cancelaciones' not in df.columns:
    print("Error: La columna 'cancelaciones' no se encuentra en el archivo CSV.  Por favor, verifica el nombre de la columna.")
    exit()
cancelaciones = df['cancelaciones']

# 2. Tabla de Frecuencias, Probabilidades Empíricas y Distribuciones Acumuladas
frecuencia_absoluta = cancelaciones.value_counts().sort_index()
probabilidad_empirica = frecuencia_absoluta / len(cancelaciones)
distribucion_acumulada = probabilidad_empirica.cumsum()

tabla_frecuencias = pd.DataFrame({
    'Cancelaciones': frecuencia_absoluta.index,
    'Frecuencia Absoluta': frecuencia_absoluta.values,
    'Probabilidad Empírica': probabilidad_empirica.values,
    'Distribución Acumulada': distribucion_acumulada.values
})

print("\n1. Tabla de Frecuencias, Probabilidades Empíricas y Distribuciones Acumuladas:")
print(tabla_frecuencias)

# 3. Esperanza y Varianza Empíricas
esperanza_empirica = cancelaciones.mean()
varianza_empirica = cancelaciones.var(ddof=0)  # ddof=0 para varianza poblacional

print("\n2. Esperanza y Varianza Empíricas:")
print(f"Esperanza Empírica (Media): {esperanza_empirica:.2f}")
print(f"Varianza Empírica: {varianza_empirica:.2f}")

# 4. Mediana, Rango Intercuartílico y Diagrama de Cajas
mediana = cancelaciones.median()
q1 = cancelaciones.quantile(0.25)
q3 = cancelaciones.quantile(0.75)
iqr = q3 - q1

print("\n3. Mediana, Rango Intercuartílico y Diagrama de Cajas:")
print(f"Mediana: {mediana:.2f}")
print(f"Rango Intercuartílico (IQR): {iqr:.2f}")

# Diagrama de cajas
plt.figure(figsize=(8, 6))
sns.boxplot(x=cancelaciones)
plt.title('Diagrama de Cajas: Cancelaciones Diarias')
plt.xlabel('Cancelaciones')
plt.ylabel('Cancelaciones')  # Agregado etiqueta al eje y
plt.show()


# 5. Histograma de Cancelaciones Diarias
plt.figure(figsize=(8, 6))
sns.histplot(cancelaciones, bins=range(cancelaciones.min(), cancelaciones.max() + 2), kde=False, stat='count')
plt.title('Histograma de Cancelaciones Diarias')
plt.xlabel('Cancelaciones')
plt.ylabel('Frecuencia')
plt.show()

# 6. Comparación con Distribución de Poisson
mu = esperanza_empirica
x_vals = np.arange(cancelaciones.min(), cancelaciones.max() + 1)
poisson_pmf = poisson.pmf(x_vals, mu)

plt.figure(figsize=(8, 6))
sns.histplot(cancelaciones, bins=range(cancelaciones.min(), cancelaciones.max() + 2), stat='probability', label='Empírica')
plt.plot(x_vals, poisson_pmf, 'ro-', label=f'Poisson(λ={mu:.2f})')
plt.title('Histograma con Ajuste de Poisson')
plt.xlabel('Cancelaciones')
plt.ylabel('Probabilidad')
plt.legend()
plt.show()

print("\n5. Comparación con Distribución de Poisson:")
if varianza_empirica / esperanza_empirica < 1.2 and varianza_empirica / esperanza_empirica > 0.8:
    print("El ajuste de Poisson es razonable ya que la varianza es aproximadamente igual a la media.")
else:
    print("El ajuste de Poisson no es razonable ya que la varianza no es aproximadamente igual a la media")

# 7. Probabilidades con la Distribución de Poisson
prob_menos_de_5 = poisson.cdf(4, mu)
prob_mas_de_15 = 1 - poisson.cdf(15, mu)

print("\n6. Probabilidades con la Distribución de Poisson:")
print(f"Probabilidad de menos de 5 cancelaciones (P(X < 5)): {prob_menos_de_5:.4f}")
print(f"Probabilidad de más de 15 cancelaciones (P(X > 15)): {prob_mas_de_15:.4f}")
