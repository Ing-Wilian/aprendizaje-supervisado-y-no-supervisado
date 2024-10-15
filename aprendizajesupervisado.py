import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


file_name = 'data1.csv'
df = pd.read_csv(file_name)

print("Columnas disponibles en el DataFrame:")
print(df.columns)


print("Datos originales:")
print(df.head())


X = df[['rate-12-13', 'rate-13-14', 'rate-14-15', 'rate-15-16', 'rate-16-17', 'rate-17-18', 'rate-18-19', 'rate-19-20', 'rate-20-21']]
y = df['rate-21-22']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


y_pred = lin_reg.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nError cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)


plt.scatter(X_test[:, 0], y_test, color='blue', label='Datos reales')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicciones')
plt.title('Regresión Lineal - Comparación de Datos Reales y Predicciones')
plt.xlabel('Rate 12-13 (Escalado)')
plt.ylabel('Rate 21-22')
plt.legend()
plt.grid(True)
plt.show()
