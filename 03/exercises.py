from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
score_untuned = knn_clf.score(X_test, y_test)
print('Before tuning: ', score_untuned)

param_grid = [
    {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [3, 4, 5, 6, 7, 8],
    }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train[:10_000], y_train[:10_000])

estimator = grid_search.best_estimator_.fit(X_train, y_train)
score_tuned = estimator.score(X_test, y_test)
print('After tuning: ', score_tuned)
print('Params: ', grid_search.best_params_)