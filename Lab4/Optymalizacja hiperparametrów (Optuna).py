import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
import optuna


def objective_svm(trial):
    C = trial.suggest_float('C', 1e-2, 1e2, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    # Tworzenie modelu
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    # Walidacja krzyżowa
    scores = cross_validate(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores['test_score'].mean()


def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    scores = cross_validate(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores['test_score'].mean()


def objective_rf_multi(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    scores = cross_validate(model, X_train, y_train, cv=3, scoring=['accuracy', 'r2'])
    acc = scores['test_accuracy'].mean()
    r2 = scores['test_r2'].mean()
    return 1 - acc, 1 - r2


data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=50)

print("Najlepsze hiperparametry dla SVM:")
print(study_svm.best_params)
print("Najlepsza dokładność:", study_svm.best_value)

optuna.visualization.plot_optimization_history(study_svm).show()
optuna.visualization.plot_param_importances(study_svm).show()
optuna.visualization.plot_contour(study_svm, params=['kernel', 'gamma']).show()

best_svm = SVC(**study_svm.best_params, random_state=42)
best_svm.fit(X_train, y_train)
y_pred_svm = best_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("Dokładność na zbiorze testowym (SVM):", svm_accuracy)

# Random Forest - porównanie małej i większej liczby prób
study_rf_10 = optuna.create_study(direction='maximize')
study_rf_10.optimize(objective_rf, n_trials=10)

study_rf_100 = optuna.create_study(direction='maximize')
study_rf_100.optimize(objective_rf, n_trials=100)

print("\nRandom Forest - 10 prób:")
print(study_rf_10.best_params)
print("Najlepsza dokładność:", study_rf_10.best_value)

print("\nRandom Forest - 100 prób:")
print(study_rf_100.best_params)
print("Najlepsza dokładność:", study_rf_100.best_value)

optuna.visualization.plot_optimization_history(study_rf_100).show()
optuna.visualization.plot_param_importances(study_rf_100).show()
optuna.visualization.plot_parallel_coordinate(study_rf_100).show()

best_rf = RandomForestClassifier(**study_rf_100.best_params, random_state=42)
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Dokładność na zbiorze testowym (Random Forest):", rf_accuracy)

if rf_accuracy > svm_accuracy:
    print("Lepszy model po optymalizacji: Random Forest")
elif svm_accuracy > rf_accuracy:
    print("Lepszy model po optymalizacji: SVM")
else:
    print("Oba modele uzyskały taką samą dokładność po optymalizacji")

print("\nNajwiększy wpływ na wynik RF pokazuje wykres plot_param_importances.")
print("Parametry z najwyższym słupkiem na wykresie mają największe znaczenie dla accuracy.")

#Optymalizacja dla wielu metryuk
study_rf_multi = optuna.create_study(directions=['minimize', 'minimize'])
study_rf_multi.optimize(objective_rf_multi, n_trials=100)

print("\nLiczba rozwiązań Pareto:", len(study_rf_multi.best_trials))
for trial in study_rf_multi.best_trials[:5]:
    print("Wartości celu:", trial.values, "Parametry:", trial.params)

optuna.visualization.plot_pareto_front(
    study_rf_multi,
    target_names=['1 - accuracy', '1 - r2']
).show()
