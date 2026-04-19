import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print("\nbrakujące wartosci")
print(df.isnull().sum())

print("\npodstawowe statystyki")
print(df.describe())


# 2) metoda najmniejszych kwadratów

class MyRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.beta = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        self.beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta


# 3) funkcja do testowania modeli

def compare_models(df, feature_cols, target_col="MedHouseVal"):
    print(f"test dla cech: {feature_cols}")
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # własny model
    my_model = MyRegression()
    my_model.fit(X_train, y_train)
    y_pred_my = my_model.predict(X_test)

    mse_my = mean_squared_error(y_test, y_pred_my)
    r2_my = r2_score(y_test, y_pred_my)

    # sklearn
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    y_pred_sk = sk_model.predict(X_test)

    mse_sk = mean_squared_error(y_test, y_pred_sk)
    r2_sk = r2_score(y_test, y_pred_sk)

    # wyniki
    print("\nwłasna implementacja")
    print("Intercept:", my_model.intercept_)
    print("Coef:", my_model.coef_)
    print("MSE:", mse_my)
    print("R^2:", r2_my)

    print("\nscikit-learn model")
    print("Intercept:", sk_model.intercept_)
    print("Coef:", sk_model.coef_)
    print("MSE:", mse_sk)
    print("R^2:", r2_sk)

    comparison = pd.DataFrame({
        "Model": ["Własna implementacja", "Scikit-learn"],
        "MSE": [mse_my, mse_sk],
        "R^2": [r2_my, r2_sk]
    })

    print("\nporównanie regresji")
    print(comparison)

    return {
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_my": y_pred_my,
        "y_pred_sk": y_pred_sk,
        "mse_my": mse_my,
        "r2_my": r2_my,
        "mse_sk": mse_sk,
        "r2_sk": r2_sk,
        "comparison": comparison,
        "my_model": my_model,
        "sk_model": sk_model
    }


# 4) porównanie dla różnych cech

# Zestaw 1 - jedna cecha
result1 = compare_models(df, ["MedInc"])

# Zestaw 2 - trzy cechy
result2 = compare_models(df, ["MedInc", "HouseAge", "AveRooms"])

# Zestaw 3 - wszystkie cechy
result3 = compare_models(df, list(df.drop(columns=["MedHouseVal"]).columns))


# 5) wizualizacja wyników

def plot_regression_line(X_test, y_test, model, feature_name, title):
    X_test_array = np.array(X_test).reshape(-1)
    y_test_array = np.array(y_test)

    sort_idx = np.argsort(X_test_array)
    X_sorted = X_test_array[sort_idx]
    y_pred_sorted = model.predict(X_sorted.reshape(-1, 1))

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test_array, y_test_array, alpha=0.4, label="Dane rzeczywiste")
    plt.plot(X_sorted, y_pred_sorted, color="red", linewidth=2, label="Linia regresji")
    plt.xlabel(feature_name)
    plt.ylabel("MedHouseVal")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(y_test, y_pred_my, y_pred_sk, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_my, alpha=0.4, label="Własna implementacja")
    plt.scatter(y_test, y_pred_sk, alpha=0.4, label="Scikit-learn")
    plt.xlabel("Wartości rzeczywiste")
    plt.ylabel("Wartości przewidywane")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_residuals(y_test, y_pred, title):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(y=0, linestyle="--", color="red")
    plt.xlabel("Wartości przewidywane")
    plt.ylabel("Reszty")
    plt.title(title)
    plt.grid(True)
    plt.show()


# wykres linii regresji dla jednej cechy
plot_regression_line(
    result1["X_test"],
    result1["y_test"],
    result1["my_model"],
    "MedInc",
    "Regresja liniowa - własna implementacja"
)

plot_regression_line(
    result1["X_test"],
    result1["y_test"],
    result1["sk_model"],
    "MedInc",
    "Regresja liniowa - scikit-learn"
)

# wykres porównania predykcji
plot_predictions(
    result1["y_test"],
    result1["y_pred_my"],
    result1["y_pred_sk"],
    "Porównanie predykcji - 1 cecha (MedInc)"
)

plot_residuals(
    result1["y_test"],
    result1["y_pred_my"],
    "Wykres reszt - własna implementacja (MedInc)"
)

plot_residuals(
    result1["y_test"],
    result1["y_pred_sk"],
    "Wykres reszt - scikit-learn (MedInc)"
)


# 6) porównanie wyników

summary = pd.DataFrame({
    "Zestaw cech": [
        "['MedInc']",
        "['MedInc', 'HouseAge', 'AveRooms']",
        "Wszystkie cechy"
    ],
    "MSE - własna": [result1["mse_my"], result2["mse_my"], result3["mse_my"]],
    "R^2 - własna": [result1["r2_my"], result2["r2_my"], result3["r2_my"]],
    "MSE - sklearn": [result1["mse_sk"], result2["mse_sk"], result3["mse_sk"]],
    "R^2 - sklearn": [result1["r2_sk"], result2["r2_sk"], result3["r2_sk"]],
})

print("\nzbiorcze porównanie")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
print(summary)

# wykres porównania R^2
plt.figure(figsize=(9, 5))
x = np.arange(len(summary))
width = 0.35

plt.bar(x - width / 2, summary["R^2 - własna"], width, label="Własna implementacja")
plt.bar(x + width / 2, summary["R^2 - sklearn"], width, label="Scikit-learn")

plt.xticks(x, summary["Zestaw cech"], rotation=15)
plt.ylabel("R^2")
plt.title("Porównanie R^2 dla różnych zestawów cech")
plt.legend()
plt.grid(axis="y")
plt.show()

# wykres porównania MSE
plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, summary["MSE - własna"], width, label="Własna implementacja")
plt.bar(x + width / 2, summary["MSE - sklearn"], width, label="Scikit-learn")

plt.xticks(x, summary["Zestaw cech"], rotation=15)
plt.ylabel("MSE")
plt.title("Porównanie MSE dla różnych zestawów cech")
plt.legend()
plt.grid(axis="y")
plt.show()