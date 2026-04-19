import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

X = df[["MedInc"]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) regresja liniowa

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nregresja liniowa")
print("Współczynniki:")
for feature, coef in zip(X.columns, linear_model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Wyraz wolny: {linear_model.intercept_:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"R^2: {r2_linear:.4f}")

# 3) regresja wielomianowa

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nregresja wielomianowa stopnia 2")
print("Nazwy cech wielomianowych:")
print(poly.get_feature_names_out(X.columns))
print(f"MSE: {mse_poly:.4f}")
print(f"R^2: {r2_poly:.4f}")

# stopień 3 wielomianu

poly3 = PolynomialFeatures(degree=3, include_bias=False)

X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

poly3_model = LinearRegression()
poly3_model.fit(X_train_poly3, y_train)
y_pred_poly3 = poly3_model.predict(X_test_poly3)

mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
r2_poly3 = r2_score(y_test, y_pred_poly3)

print("\nregresja wielomianowa, stopien 3")
print(f"MSE: {mse_poly3:.4f}")
print(f"R^2: {r2_poly3:.4f}")

# 5) porownanie

comparison = pd.DataFrame({
    "Model": [
        "Regresja liniowa",
        "Regresja wielomianowa stopień 2",
        "Regresja wielomianowa stopień 3"
    ],
    "MSE": [mse_linear, mse_poly, mse_poly3],
    "R^2": [r2_linear, r2_poly, r2_poly3]
})

print("\nPorownanie regresji")
print(comparison)

# przygotowanie danych do rysowania linii regresji
X_test_values = X_test["MedInc"].values
sort_idx = np.argsort(X_test_values)

X_test_sorted = X_test_values[sort_idx]
y_test_sorted = y_test.values[sort_idx]
y_pred_linear_sorted = y_pred_linear[sort_idx]
y_pred_poly_sorted = y_pred_poly[sort_idx]
y_pred_poly3_sorted = y_pred_poly3[sort_idx]

# 6) wizualizacja

# 6.1 Wykres danych + czerwona linia regresji liniowej
plt.figure(figsize=(8, 5))
plt.scatter(X_test["MedInc"], y_test, alpha=0.4, label="Dane rzeczywiste")
plt.plot(X_test_sorted, y_pred_linear_sorted, color="red", linewidth=2, label="Linia regresji liniowej")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.title("Regresja liniowa - czerwona linia regresji")
plt.legend()
plt.grid(True)
plt.show()

# 6.2 Wykres danych + czerwona krzywa regresji wielomianowej stopnia 2
plt.figure(figsize=(8, 5))
plt.scatter(X_test["MedInc"], y_test, alpha=0.4, label="Dane rzeczywiste")
plt.plot(X_test_sorted, y_pred_poly_sorted, color="red", linewidth=2, label="Krzywa regresji wielomianowej stopnia 2")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.title("Regresja wielomianowa stopnia 2 - czerwona krzywa")
plt.legend()
plt.grid(True)
plt.show()

# 6.3 Wykres danych + czerwona krzywa regresji wielomianowej stopnia 3
plt.figure(figsize=(8, 5))
plt.scatter(X_test["MedInc"], y_test, alpha=0.4, label="Dane rzeczywiste")
plt.plot(X_test_sorted, y_pred_poly3_sorted, color="red", linewidth=2, label="Krzywa regresji wielomianowej stopnia 3")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.title("Regresja wielomianowa stopnia 3 - czerwona krzywa")
plt.legend()
plt.grid(True)
plt.show()

# 6.4 Rzeczywiste vs przewidywane - regresja liniowa
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_linear, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Regresja liniowa: wartości rzeczywiste vs przewidywane")
plt.grid(True)
plt.show()

# 6.5 Rzeczywiste vs przewidywane - regresja wielomianowa stopień 2
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_poly, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Regresja wielomianowa stopień 2: rzeczywiste vs przewidywane")
plt.grid(True)
plt.show()

# 6.6 Wykres reszt - regresja liniowa
residuals_linear = y_test - y_pred_linear

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_linear, residuals_linear, alpha=0.4)
plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Wartości przewidywane")
plt.ylabel("Reszty")
plt.title("Wykres reszt - regresja liniowa")
plt.grid(True)
plt.show()

# 6.7 Wykres reszt - regresja wielomianowa stopień 2
residuals_poly = y_test - y_pred_poly

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_poly, residuals_poly, alpha=0.4)
plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Wartości przewidywane")
plt.ylabel("Reszty")
plt.title("Wykres reszt - regresja wielomianowa stopień 2")
plt.grid(True)
plt.show()

# 6.8 Porównanie modeli na wykresie słupkowym
plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["R^2"])
plt.ylabel("R^2")
plt.title("Porównanie modeli")
plt.xticks(rotation=15)
plt.grid(axis="y")
plt.show()