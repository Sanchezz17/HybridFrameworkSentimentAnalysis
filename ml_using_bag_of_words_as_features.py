import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN


def knn_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) \
        -> np.ndarray:
    model = KNN(n_neighbors=10)
    model.fit(X_train, y_train)
    return model.predict(X_test)
