import numpy as np
from sklearn.datasets import make_moons, load_iris
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_classifier = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_classifier.fit(X_train, y_train)

for name, classifier in voting_classifier.named_estimators_.items():
    print(name, '=', classifier.score(X_test, y_test))

votings_predict = voting_classifier.predict(X_test[:1])
predicts = [clf.predict(X_test[:1]) for clf in voting_classifier.estimators_]

print(voting_classifier.score(X_test, y_test))

voting_classifier.voting = "soft"
voting_classifier.named_estimators['svc'].probability = True
voting_classifier.fit(X_train, y_train)
voting_classifier.score(X_test, y_test)

print(voting_classifier.score(X_test, y_test))

# bagging classifier
bag_classifier = BaggingClassifier(DecisionTreeClassifier(),
                                   n_estimators=500,
                                   max_samples=100,
                                   n_jobs=-1,
                                   random_state=42,
                                   )
bag_classifier.fit(X_train, y_train)
print(bag_classifier.score(X_test, y_test))

bag_classifier = BaggingClassifier(DecisionTreeClassifier(),
                                   n_estimators=500,
                                   oob_score=True,
                                   n_jobs=-1,
                                   random_state=42,
                                   )
bag_classifier.fit(X_train, y_train)

print("OOB: ", bag_classifier.oob_score_)

y_pred = bag_classifier.predict(X_test)
print("Prediction accuracy: ", accuracy_score(y_test, y_pred))

print(bag_classifier.oob_decision_function_[:3])

# random forests
random_forest_classifier = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
random_forest_classifier.fit(X_train, y_train)
y_pred_random_forest = random_forest_classifier.predict(X_test)

# feature importance
iris = load_iris(as_frame=True)
rnd_classifier = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_classifier.fit(iris.data, iris.target)
for score, name in zip(rnd_classifier.feature_importances_, iris.data.columns):
    print(round(score, 2), name)

# ADA boost
ada_classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30, learning_rate=0.5, random_state=42
)
ada_classifier.fit(X_train, y_train)

# print(ada_classifier.score(X_test, y_test))

# gradient boosting
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

tree_regressor1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_regressor1.fit(X, y)

y2 = y - tree_regressor1.predict(X)
tree_regressor2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_regressor2.fit(X, y2)

y3 = y2 - tree_regressor2.predict(X)
tree_regressor3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_regressor3.fit(X, y3)

X_new = np.array([[-0.4], [0.], [0.5]])
print(sum(tree.predict(X_new) for tree in (tree_regressor1, tree_regressor2, tree_regressor3)))

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)

gbrt_best = GradientBoostingRegressor(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=500,
    n_iter_no_change=10,
    random_state=42
)
gbrt_best.fit(X, y)

# stacking
stacking_classifier = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5
)
stacking_classifier.fit(X_train, y_train)

print(stacking_classifier.score(X_test, y_test))
