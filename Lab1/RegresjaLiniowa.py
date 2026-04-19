import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1) przygotowanie danych

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

print("\ninformacje o danych")
print(df.info())

print("\nbrakujace wartosci")
print(df.isnull().sum())

print("\npodstawowe statystyki")
print(df.describe())

#zależność średniego dochodu od wartości domu- przykładowe dane
plt.figure(figsize=(8, 5))
plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3)
plt.xlabel("MedInc (średni dochód)")
plt.ylabel("MedHouseVal (średnia wartość domu)")
plt.title("Zależność wartości domu od średniego dochodu")
plt.grid(True)
plt.show()


# 2) regresja scikit

# wszystkie cechy jako zmienne wejsciowe
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# podzial na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# utworzenie i trenowanie modelu
model = LinearRegression()
model.fit(X_train, y_train)

# predykcja
y_pred = model.predict(X_test)

# ocena modelu
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nmodel podstawowy, wszystkie zmienne")
print("Współczynniki:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"\nwyraz wolny: {model.intercept_:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")


# 3) analiza wplywu parametrow

# Model 1: tylko jedna cecha
X1 = df[["MedInc"]]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

mse1 = mean_squared_error(y1_test, y1_pred)
r2_1 = r2_score(y1_test, y1_pred)

print("\ntylko MedInc")
print(f"MSE: {mse1:.4f}")
print(f"R^2: {r2_1:.4f}")


# Model 2: kilka wybranych cech
X2 = df[["MedInc", "HouseAge", "AveRooms"]]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mse2 = mean_squared_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print("\nMedInc + HouseAge + AveRooms")
print(f"MSE: {mse2:.4f}")
print(f"R^2: {r2_2:.4f}")


# Model 3: wszystkie cechy
X3 = df.drop("MedHouseVal", axis=1)

X3_train, X3_test, y3_train, y3_test = train_test_split(
    X3, y, test_size=0.2, random_state=42
)

model3 = LinearRegression()
model3.fit(X3_train, y3_train)
y3_pred = model3.predict(X3_test)

mse3 = mean_squared_error(y3_test, y3_pred)
r2_3 = r2_score(y3_test, y3_pred)

print("\nwszystkie cechy")
print(f"MSE: {mse3:.4f}")
print(f"R^2: {r2_3:.4f}")


# Tabela porównawcza
comparison = pd.DataFrame({
    "Model": [
        "Tylko MedInc",
        "MedInc + HouseAge + AveRooms",
        "Wszystkie cechy"
    ],
    "MSE": [mse1, mse2, mse3],
    "R^2": [r2_1, r2_2, r2_3]
})

print("\nporównanie modeli")
print(comparison)

# 4) wizualizacja wyników
plt.figure(figsize=(8, 5))
plt.scatter(X1_test["MedInc"], y1_test, alpha=0.4, label="Dane rzeczywiste")

sort_idx = np.argsort(X1_test["MedInc"].values)
X1_sorted = X1_test["MedInc"].values[sort_idx]
y1_sorted_pred = y1_pred[sort_idx]

plt.plot(X1_sorted, y1_sorted_pred, color="red", linewidth=2, label="Linia regresji")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.title("Model 1: Regresja liniowa dla jednej zmiennej")
plt.legend()
plt.grid(True)
plt.show()


#  Model 2 - rzeczywiste vs przewidywane
plt.figure(figsize=(8, 5))
plt.scatter(y2_test, y2_pred, alpha=0.4, label="Punkty")

min_val = min(y2_test.min(), y2_pred.min())
max_val = max(y2_test.max(), y2_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2, label="Linia idealnego dopasowania")
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Model 2: Rzeczywiste vs przewidywane")
plt.legend()
plt.grid(True)
plt.show()


# Model 3 - rzeczywiste vs przewidywane
plt.figure(figsize=(8, 5))
plt.scatter(y3_test, y3_pred, alpha=0.4, label="Punkty")

min_val = min(y3_test.min(), y3_pred.min())
max_val = max(y3_test.max(), y3_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2, label="Linia idealnego dopasowania")
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Model 3: Rzeczywiste vs przewidywane")
plt.legend()
plt.grid(True)
plt.show()

# Reszty dla modelu 3
residuals3 = y3_test - y3_pred

plt.figure(figsize=(8, 5))
plt.scatter(y3_pred, residuals3, alpha=0.4)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Wartości przewidywane")
plt.ylabel("Reszty")
plt.title("Model 3: Wykres reszt")
plt.grid(True)
plt.show()

# Porównanie R^2 modeli

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["R^2"])
plt.ylabel("R^2")
plt.title("Porównanie modeli")
plt.xticks(rotation=15)
plt.grid(axis="y")
plt.show()