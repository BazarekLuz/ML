import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.svm import SVC


def loadTitanic():
    tarballPath = Path('datasets/titanic.tgz')
    if not tarballPath.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = 'https://github.com/ageron/data/raw/main/titanic.tgz'
        urllib.request.urlretrieve(url, tarballPath)
        with tarfile.open(tarballPath) as titanicTarball:
            titanicTarball.extractall(path='datasets')
    return pd.read_csv('datasets/titanic/train.csv'), pd.read_csv('datasets/titanic/test.csv')


train, test = loadTitanic()

train = train.set_index('PassengerId')
test = test.set_index('PassengerId')

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

cat_pipeline = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(sparse_output=False)
)

num_attributes = ['Age', 'SibSp', 'Parch', 'Fare']
cat_attributes = ['Pclass', 'Sex', 'Embarked']

preprocess_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes),
])

X_train = preprocess_pipeline.fit_transform(train)
y_train = train['Survived']

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print("Forest classifier mean score: ", forest_scores.mean())

svm_clf = SVC(gamma='auto')
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print("SVM classifier mean score: ", svm_scores.mean())