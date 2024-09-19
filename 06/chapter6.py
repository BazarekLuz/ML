import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from graphviz import Source

iris = load_iris(as_frame=True)
X_iris = iris.data[['petal length (cm)', 'petal width (cm)']].values
y_iris = iris.target

tree_classifier = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_classifier.fit(X_iris, y_iris)

export_graphviz(
    tree_classifier,
    out_file='iris_tree.dot',
    feature_names=['petal length (cm)', 'petal width (cm)'],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

print(tree_classifier.predict_proba([[6, 1.5]]).round(3))


# make moons classification
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_classifier1 = DecisionTreeClassifier(random_state=42)
tree_classifier2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
tree_classifier1.fit(X_moons, y_moons)
tree_classifier2.fit(X_moons, y_moons)

X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)

print(tree_classifier1.score(X_moons_test, y_moons_test))
print(tree_classifier2.score(X_moons_test, y_moons_test))

# decision tree regressor
np.random.seed(42)
X_quad = np.random.rand(200, 1) - 0.5
y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

tree_regressor = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_regressor.fit(X_quad, y_quad)

export_graphviz(
    tree_regressor,
    out_file='x1_tree.dot',
    feature_names=['x1'],
    rounded=True,
    filled=True
)

# pca
pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_classifier_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_classifier_pca.fit(X_iris_rotated, y_iris)

# exercise 7
X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_moons_train, X_moons_test, y_moons_train, y_moons_test = train_test_split(X_moons, y_moons, random_state=42)

param_distrib = {
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': list(range(1, 7)),
    'min_samples_split': [2, 3, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_distrib, cv=3)
grid_search.fit(X_moons_train, y_moons_train)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)

y_pred = grid_search.predict(X_moons_test)
accuracy_score(y_moons_test, y_pred)

