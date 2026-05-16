import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Wczytanie danych Wine
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

print(df.head())

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means po skalowaniu
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df['cluster'] = kmeans.labels_

# Wizualizacja K-Means na dwóch pierwszych cechach
sns.scatterplot(
    x=df[feature_names[0]],
    y=df[feature_names[1]],
    hue=df['cluster'],
    palette='Set1'
)
plt.title("Grupowanie K-Means na oryginalnych cechach")
plt.show()

# Porównanie z prawdziwymi klasami
print(pd.crosstab(df['species'], df['cluster']))

# Metoda łokcia
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Liczba klastrów')
plt.ylabel('Wewnętrzna suma kwadratów')
plt.title('Metoda łokcia')
plt.show()

# Silhouette Score dla różnych wartości k
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"k = {k}, Silhouette Score = {score:.4f}")

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Liczba klastrów k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score dla różnych wartości k')
plt.show()

# PCA do 2 wymiarów
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

df['PCA1_2D'] = X_pca_2d[:, 0]
df['PCA2_2D'] = X_pca_2d[:, 1]

# K-Means po PCA 2D
kmeans_pca_2d = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_pca_2d'] = kmeans_pca_2d.fit_predict(X_pca_2d)

sns.scatterplot(
    x=df['PCA1_2D'],
    y=df['PCA2_2D'],
    hue=df['cluster_pca_2d'],
    palette='Set1'
)
plt.title("Wyniki PCA 2D + K-Means")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# PCA do 3 wymiarów
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

df['PCA1'] = X_pca_3d[:, 0]
df['PCA2'] = X_pca_3d[:, 1]
df['PCA3'] = X_pca_3d[:, 2]

# K-Means po PCA 3D
kmeans_pca_3d = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_pca_3d'] = kmeans_pca_3d.fit_predict(X_pca_3d)

# Wizualizacja 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['PCA1'],
    df['PCA2'],
    df['PCA3'],
    c=df['cluster_pca_3d'],
    cmap='Set1',
    s=50
)

ax.set_title("Wyniki PCA 3D + K-Means")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")

plt.colorbar(scatter, ax=ax, label='Cluster')
plt.show()

# Porównanie jakości grupowania po PCA dla różnej liczby wymiarów
print("\nPorównanie Silhouette Score po PCA:")

for n in range(2, X_scaled.shape[1] + 1):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)
    explained_variance = pca.explained_variance_ratio_.sum()

    print(
        f"PCA n_components = {n}, "
        f"Silhouette Score = {score:.4f}, "
    )

# Dodatkowe porównanie: bez PCA, PCA 2D i PCA 3D
labels_original = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
score_original = silhouette_score(X_scaled, labels_original)

labels_pca_2d = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca_2d)
score_pca_2d = silhouette_score(X_pca_2d, labels_pca_2d)

labels_pca_3d = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca_3d)
score_pca_3d = silhouette_score(X_pca_3d, labels_pca_3d)

print("\nPodsumowanie:")
print(f"Bez PCA: Silhouette Score = {score_original:.4f}")
print(f"PCA 2D:  Silhouette Score = {score_pca_2d:.4f}")
print(f"PCA 3D:  Silhouette Score = {score_pca_3d:.4f}")