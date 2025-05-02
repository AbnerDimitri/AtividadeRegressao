import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("EXERCÍCIO 2: Aditivo vs Octanagem\n")

data = {
    'percentual_aditivo': [1, 2, 3, 4, 5, 6],
    'octanagem': [87, 88, 90, 92, 94, 95]
}
df = pd.DataFrame(data)

X = df[['percentual_aditivo']]
y = df['octanagem']

model = LinearRegression()
model.fit(X, y)

a = model.intercept_
b = model.coef_[0]
print(f"Equação: octanagem = {a:.2f} + {b:.2f} * aditivo")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R²: {r2:.2f}")

oct_5_5 = model.predict([[5.5]])
print(f"Octanagem prevista para 5,5% de aditivo: {oct_5_5[0]:.2f}")

plt.scatter(X, y, color='green', label='Dados reais')
plt.plot(X, y_pred, color='orange', label='Regressão linear')
plt.xlabel('Percentual de Aditivo (%)')
plt.ylabel('Octanagem')
plt.title('Aditivo vs Octanagem')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
