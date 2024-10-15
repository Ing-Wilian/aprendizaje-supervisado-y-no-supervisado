import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar los datos desde el archivo CSV
# Asegúrate de que el nombre del archivo sea correcto
file_name = 'data1.csv'  # Cambia 'nombre_del_archivo.csv' por el nombre real de tu archivo
df = pd.read_csv(file_name)

# Mostrar los datos originales
print("Datos originales:")
print(df.head())

# Seleccionar las columnas de tasas para el K-Means
# Ajusta las columnas según tu archivo
X = df[['rate-12-13', 'rate-13-14', 'rate-14-15', 'rate-15-16', 'rate-16-17', 'rate-17-18', 'rate-18-19', 'rate-19-20', 'rate-20-21']]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Cambia n_clusters según sea necesario
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Mostrar los resultados
print("\nDatos después de K-Means:")
print(df)

# Visualizar los clusters (ejemplo solo con dos dimensiones para la visualización)
plt.scatter(df['rate-12-13'], df['rate-13-14'], c=df['Cluster'], cmap='viridis', marker='o')
plt.title('K-Means Clustering')
plt.xlabel('Rate 12-13')
plt.ylabel('Rate 13-14')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
