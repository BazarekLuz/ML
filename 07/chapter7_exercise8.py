import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

X, y = fetch_openml('mnist_784', as_frame=False, return_X_y=True, parser='auto')

X_train, y_train = X[:50_000], y[:50_000]
X_valid, y_valid = X[50_000:60_000], y[50_000:60_000]
X_test, y_test = X[60_000:], y[60_000:]


random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_classifier = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
mlp_classifier = MLPClassifier(random_state=42)

estimators = [random_forest_classifier, extra_trees_classifier, svm_classifier, mlp_classifier]
for estimator in estimators:
    print('Training the ', estimator)
    estimator.fit(X_train, y_train)

scores = [estimator.score(X_valid, y_valid) for estimator in estimators]
print(scores)

named_estimators = [
    ('random_forest_classifier', random_forest_classifier),
    ('extra_trees_classifier', extra_trees_classifier),
    ('svm_classifier', svm_classifier),
    ('mlp_classifier', mlp_classifier),
]

voting_classifier = VotingClassifier(estimators=named_estimators)
voting_classifier.fit(X_train, y_train)

print(voting_classifier.score(X_valid, y_valid))

y_valid_encoded = y_valid.astype(np.int64)

encoded_scores = [estimator.score(X_valid, y_valid_encoded) for estimator in voting_classifier.estimators_]

voting_classifier.set_params(svm_classifier='drop')
svm_clf_trained = voting_classifier.named_estimators_.pop("svm_classifier")
voting_classifier.estimators_.remove(svm_clf_trained)

print(voting_classifier.score(X_valid, y_valid))

final_scores = [estimator.score(X_test, y_test.astype(np.int64)) for estimator in voting_classifier.estimators_]
print("Final scores: ", final_scores)
