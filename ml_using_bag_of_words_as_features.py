import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def knn_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = KNN(n_neighbors=10)
    return classification(model, X_train, y_train, X_test)


def svc_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = SVC(kernel='linear', random_state=0)
    return classification(model, X_train, y_train, X_test)


def svc_rbf_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = SVC(kernel='rbf', random_state=0)
    return classification(model, X_train, y_train, X_test)


def naive_bayes_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = GaussianNB()
    return classification(model, X_train, y_train, X_test)


def dt_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    return classification(model, X_train, y_train, X_test)


def random_forest_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = RandomForestClassifier(max_depth=5, random_state=0)
    return classification(model, X_train, y_train, X_test)


def ada_boost_classification(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model = AdaBoostClassifier()
    return classification(model, X_train, y_train, X_test)


def classification(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_test)


def classify_and_print_accuracy(method, method_name: str,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = method(X_train, y_train, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{method_name} ML using bag of words as features accuracy: {accuracy}")


def classify_and_print_accuracy_all_methods(X_train: np.ndarray, y_train: np.ndarray,
                                            X_test: np.ndarray, y_test: np.ndarray) -> None:
    classify_and_print_accuracy(knn_classification, "KNN", X_train, y_train, X_test, y_test)

    classify_and_print_accuracy(svc_classification, "SVC", X_train, y_train, X_test, y_test)

    classify_and_print_accuracy(svc_rbf_classification, "SVC RBF", X_train, y_train, X_test, y_test)

    classify_and_print_accuracy(naive_bayes_classification, "Naive Bayes", X_train, y_train, X_test, y_test)

    classify_and_print_accuracy(dt_classification, "DT", X_train, y_train, X_test, y_test)

    classify_and_print_accuracy(random_forest_classification, "Random forest", X_train, y_train, X_test, y_test)

    classify_and_print_accuracy(ada_boost_classification, "AdaBoost", X_train, y_train, X_test, y_test)
