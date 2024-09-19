from scipy.stats import loguniform, uniform
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

wine = load_wine(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=42)

linear_classifier = make_pipeline(StandardScaler(), LinearSVC(dual=True, random_state=42))
linear_classifier.fit(X_train, y_train)

score = cross_val_score(linear_classifier, X_train, y_train).mean()
print(score)

svm_classifier = make_pipeline(StandardScaler(), SVC(random_state=42))

svm_score = cross_val_score(svm_classifier, X_train, y_train)

param_distrib = {
    "svc__gamma": loguniform(0.001, 0.1),
    "svc__C": uniform(1, 10)
}

random_search = RandomizedSearchCV(svm_classifier, param_distrib, n_iter=100, random_state=42, cv=5)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
print(random_search.best_score_)

print(random_search.score(X_test, y_test))
