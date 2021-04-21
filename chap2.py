# import mglearn.datasets
from mglearn.tools import discrete_scatter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
import mglearn.datasets
import mglearn.plots

# surapprentissage = overfitting
# => on aboutit a une regle trop précise pas assez généraliste

# sousapprentisage= underfitting
# => ne tire pas assez d'information des données

# point idée => millieu entre sous-aprentissage et sur-apprentisage
"""
X, y = mglearn.datasets.make_forge()
discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print(f"X.shape: {X.shape}")

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
"""

cancer = load_breast_cancer()
print(f"cancer.keys(): {', '.join(cancer)}")
# cancer.keys(): data, target, frame, target_names,
# DESCR, feature_names, filename

print(f"Shape of cancer data: {cancer.data.shape}")
# Shape of cancer data: (569, 30)

counts = dict(zip(cancer.target_names, np.bincount(cancer.target)))
print(f"Sample counts per class: \n{counts}")
# Sample counts per class:
# {'malignant': 212, 'benign': 357}

print(f"Feature names:\n{cancer.feature_names}")
# Feature names:
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

boston = load_boston()
print(f"Data shape: {boston.data.shape}")

X, y = mglearn.datasets.load_extended_boston()
print(f"X.shape: {X.shape}")

# mglearn.plots.plot_knn_classification(n_neighbors=1)
# plt.show()

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print(f"Test set prediction: {clf.predict(X_test)}")
print(f"Test set predictiond: {clf.score(X_test, y_test):.2f}")
ig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbor, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbor).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5,
                                    ax=ax, alpha=.4)
    discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(f"{n_neighbor}")
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
# + de voisin = plus simple
# - de voisin = plus complexe

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=60)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print(f"Test set predictions: {reg.predict(X_test)}")
# coefficient de déterminaton = score
print(f"Test set R^2: {reg.score(X_test, y_test):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(0), markersize=8)
    ax.set_title(
        f"{n_neighbors} neighbor(s)\n"
        f"train score: {reg.score(X_train, y_train):.2f} "
        f"test score: {reg.score(X_test, y_test):.2f}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")
plt.show()

# knn efficase pour les volumes moyene avec peu de caractéristique
# ou des champ vide, calcule avec la distance euclidienne

# modele linaire
# grace à une formule
# somme des podérer = prediction
# y = x[0] * w[0] + ... + x[p] * w[p]+ b
# x = cacatérisqtiues, (w = pente, b = origine) = paramètre

mglearn.plots.plot_linear_regression_wave()
plt.show()

# regression linéaire = méthode des moindres carrées ordinaires
# eng: ordinary least square
# cherche a minimiser l'erreur quadratique moyenne (mean squared error)
# err = sum((y_test - y_pred)**2 for  y_test, y_pred int result)/len(result)
# On ne peu pas controler la compléxité


X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

print(f"lr.coef_: {lr.coef_}")
print(f"lr.intercept_ {lr.intercept_}")
# lr.coef_: [0.39390555]
# lr.intercept_ -0.031804343026759746


# les attrbiuts se terminant par _ sont des résultats de l'apprentisage
print(f"Traing set score: {lr.score(X_train, y_train):.2f}")
print(f"Test set score: {lr.score(X_test, y_test):.2f}")
# Traing set score: 0.67
# Test set score: 0.66

# risque de surappentissage quand il y a de nombreuseuse caractéristiques
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print(f"Traing set score: {lr.score(X_train, y_train):.2f}")
print(f"Test set score: {lr.score(X_test, y_test):.2f}")
# Traing set score: 0.95
# test set score: 0.61
# ^ surapprentisage

# Régresssion ridge
# presque identique  à la régrésion linaire sauf qu'on
# recherche le poinds minimal, donc avec chaque caractéristique
# qui valent autant
ridge = Ridge().fit(X_train, y_train)
print(f"Training set score: {ridge.score(X_train, y_train):.2f}")
print(f"Test set score: {ridge.score(X_test, y_test):.2f}")
# plus complexe = moins bon sur le jeu d'apprentissage mais meilleurs
# sur le jeu de test

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print(f"Training set score: {ridge10.score(X_train, y_train):.2f}")
print(f"Test set score: {ridge10.score(X_test, y_test):.2f}")


ridge01 = Ridge(alpha=.1).fit(X_train, y_train)
print(f"Training set score: {ridge01.score(X_train, y_train):.2f}")
print(f"Test set score: {ridge01.score(X_test, y_test):.2f}")
# alpha=0 => regression linéaire
# alpha=10 => on force grandement les coéficients à être proches de 0

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()
plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()
