import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("EXERCÍCIO 1: Altura vs Peso\n")

data = {
    'altura': [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
    'peso': [50, 52, 56, 60, 65, 70, 75, 80, 85, 90]
}
df = pd.DataFrame(data)

X = df[['altura']]
y = df['peso']

model = LinearRegression()
model.fit(X, y)

a = model.intercept_
b = model.coef_[0]
print(f"Equação: peso = {a:.2f} + {b:.2f} * altura")


y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.2f}")


peso_170 = model.predict([[170]])
print(f"Peso previsto para 170 cm: {peso_170[0]:.2f} kg")


plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Regressão linear')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.title('Altura vs Peso')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
