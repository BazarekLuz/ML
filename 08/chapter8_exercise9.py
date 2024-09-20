from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', as_frame=False)
X_train = mnist.data[:60_000]
y_train = mnist.target[:60_000]

X_test = mnist.data[60_000:]
y_test = mnist.target[60_000:]

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


