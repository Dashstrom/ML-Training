import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from pandas.plotting import scatter_matrix
# import pandas as pd
# import matplotlib.pyplot as plt


iris_dataset = load_iris()

print(f"Keys: {', '.join(iris_dataset)}")
# Keys: data, target, frame, target_names, DESCR, feature_names, filename

print(f"DESCR: \n{iris_dataset['DESCR'][:225]}")
# .. _iris_dataset:
#
# Iris plants dataset
# --------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 150 (50 in each of three classes)
#     :Number of Attributes: 4 numeric, predictive attributes and the class

print(f"Target name: {iris_dataset['target_names']}")
# Target name: ['setosa' 'versicolor' 'virginica']

print(f"Feature names: {iris_dataset['feature_names']}")
# Feature names: ['sepal length (cm)', 'sepal width (cm)',
#                 'petal length (cm)', 'petal width (cm)']

print(f"Type of data: {iris_dataset['data'].__class__}")
# Type of data: <class 'numpy.ndarray'>

print(f"Shape of data: {iris_dataset['data'].shape}")
# Shape of data: (150, 4)

# élément individuel = échantillions
# propriétés = caractéristiques (features)
# forme = shape

print(f"First five rows of data:\n {iris_dataset['data'][:5]}")
# First five rows of data:
#  [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]]

print(f"Type of target: {iris_dataset['target'].__class__}")
# Type of target: <class 'numpy.ndarray'>

print(f"Shape of target: {iris_dataset['target'].shape}")
# Shape of target: (150,)
# 0 = setosa    1 = versicolor    2 = virginica

# training set = ensemble des données qui servent à entrener l'IA
# test set = ensemble des données qui servent à tester l'IA,
# elle ne doivent pas provenir du training set

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=0)
# random_state permet de tester avec une seed aléatoire


print(f"X_train shape: {X_train.shape}")
# X_train shape: (112, 4)

print(f"X_test shape: {X_test.shape}")
# X_test shape: (38, 4)

print(f"y_train shape: {y_train.shape}")
# y_train shape: (112,)

print(f"y_test shape: {y_test.shape}")
# y_test shape: (38,)

# pour représenter des données
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# scatter_matrix(iris_dataframe)
# plt.show()

# Utilisation des K plus proches voisins
knn = KNeighborsClassifier(n_neighbors=1)
# knn pourt k-nearest neighbors

print(knn.fit(X_train, y_train))

X_new = np.array([[5, 29, 1, 0.2]])
print(f"X_new.shape: {X_new.shape}")

prediction = knn.predict(X_new)
print(f"Prediction: {prediction}")
print(f"Prediction target name: {iris_dataset['target_names'][prediction]}")

# calcule de l'exactitude: pourcentage de réussite

# y_pred = knn.predict(X_test)
# print(f"Test set score: {np.mean(y_pred == y_test):.2f}")
print(f"Test set score: {knn.score(X_test, y_test):.2f}")
