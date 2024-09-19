from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

iris = load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = iris.target.values

print(X, y)

setosa_or_versicolor = (y == 0) | (y == 1)
print(setosa_or_versicolor)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

C = 5
standard_scaler = StandardScaler()
X_scaled = standard_scaler.fit_transform(X)

lin_classifier = LinearSVC(C=C, dual=True, random_state=42).fit(X_scaled, y)
svc_classifier = SVC(C=C, kernel='linear').fit(X_scaled, y)
sgd_classifier = SGDClassifier(alpha=0.05, random_state=42).fit(X_scaled, y)

def compute_decision_boundary(model):
    w = -model.coef_[0, 0] / model.coef_[0, 1]
    b = -model.intercept_[0] / model.coef_[0, 1]
    return standard_scaler.inverse_transform([[-10, -10 * w + b], [10, 10 * w + b]])


lin_line = compute_decision_boundary(lin_classifier)
svc_line = compute_decision_boundary(svc_classifier)
sgd_line = compute_decision_boundary(sgd_classifier)

# Plot all three decision boundaries
plt.figure(figsize=(11, 4))
plt.plot(lin_line[:, 0], lin_line[:, 1], "k:", label="LinearSVC")
plt.plot(svc_line[:, 0], svc_line[:, 1], "b--", linewidth=2, label="SVC")
plt.plot(sgd_line[:, 0], sgd_line[:, 1], "r-", label="SGDClassifier")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs") # label="Iris versicolor"
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo") # label="Iris setosa"
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="upper center")
plt.axis([0, 5.5, 0, 2])
plt.grid()

plt.show()