import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

print("Dokładność modelu drzewa decyzyjnego:", accuracy_tree)
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_tree))
print("Macierz konfuzji:\n", confusion_matrix(y_test, y_pred_tree))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Dokładność modelu Random Forest:", accuracy_rf)
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_rf))
print("Macierz konfuzji:\n", confusion_matrix(y_test, y_pred_rf))

feature_importances = rf_model.feature_importances_
for i, feature in enumerate(data.feature_names):
    print(f"Ważność cechy '{feature}': {feature_importances[i]:.4f}")

plt.barh(data.feature_names, feature_importances)
plt.xlabel("Ważność cechy")
plt.ylabel("Cechy")
plt.title("Ważność cech według Random Forest")
plt.show()

#2
for i in range (1,5):
    tree_model = DecisionTreeClassifier(max_depth=i, random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print("Głębokość: ", i)
    print("Dokładność modelu Random Forest dla głębokości:", accuracy_rf)
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_rf))
    print("Macierz konfuzji:\n", confusion_matrix(y_test, y_pred_rf))

#3
for i in range (50, 201, 25):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Liczba drzew: ", i)
    print("Dokładność modelu Random Forest:", accuracy_rf)
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_rf))
    print("Macierz konfuzji:\n", confusion_matrix(y_test, y_pred_rf))

#4 Grid Search dla drzewa decyzyjnego
parametry_drzewa = {
    "max_depth": [1, 2, 3, 4, 5, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3]
}

grid_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    parametry_drzewa,
    cv=5,
    scoring="accuracy"
)

grid_tree.fit(X_train, y_train)

print("Najlepsze parametry drzewa decyzyjnego:")
print(grid_tree.best_params_)

y_pred_grid_tree = grid_tree.predict(X_test)
accuracy_grid_tree = accuracy_score(y_test, y_pred_grid_tree)

print("Dokładność drzewa po GridSearchCV:", accuracy_grid_tree)
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_grid_tree))
print("Macierz konfuzji:\n", confusion_matrix(y_test, y_pred_grid_tree))


#5 Randomized Search dla Random Forest
parametry_lasu = {
    "n_estimators": [50, 75, 100, 125, 150, 175, 200],
    "max_depth": [2, 3, 4, 5, None],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3]
}

random_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    parametry_lasu,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42
)

random_rf.fit(X_train, y_train)

print("Najlepsze parametry lasu losowego:")
print(random_rf.best_params_)

y_pred_random_rf = random_rf.predict(X_test)
accuracy_random_rf = accuracy_score(y_test, y_pred_random_rf)

print("Dokładność Random Forest po RandomizedSearchCV:", accuracy_random_rf)
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred_random_rf))
print("Macierz konfuzji:\n", confusion_matrix(y_test, y_pred_random_rf))