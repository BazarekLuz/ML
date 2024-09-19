# again iris
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC

iris = load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = (iris.target == 2)

svm_classifier = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
svm_classifier.fit(X, y)

X_new = [[5.5, 1.7], [5.0, 1.5]]
# print(svm_classifier.predict(X_new))

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_classifier = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42)
)

polynomial_svm_classifier.fit(X, y)

poly_kernel_svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, coef0=1, C=5))
poly_kernel_svm_classifier.fit(X, y)

rbf_kernel_svm_classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=5, C=0.001))
rbf_kernel_svm_classifier.fit(X, y)


