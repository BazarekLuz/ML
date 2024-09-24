import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, expon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVR

os.environ['OMP_NUM_THREADS'] = '1'

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def loadHousingData():
    tarballPath = Path("datasets/housing.tgz")
    if not tarballPath.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarballPath)
        with tarfile.open(tarballPath) as housingTarball:
            housingTarball.extractall(path="datasets")
    return pd.read_csv("datasets/housing/housing.csv")


housing = loadHousingData()

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

# y = housing[['median_house_value']]
# X = housing.drop(['median_house_value'], axis=1)
housing = strat_train_set.copy()
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy='median')
housing_numerics = housing.select_dtypes(include=[np.number])

numerics_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'),
)

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)


preprocessing = ColumnTransformer([
    ('wspolczynnik_sypialni', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('pokoje_na_rodzine', ratio_pipeline(), ['total_rooms', 'households']),
    ('liczba_osob_na_dom', ratio_pipeline(), ['population', 'households']),
    ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population', 'households', 'median_income']),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ('cat', cat_pipeline, make_column_selector(dtype_include=object))
])

param_distribs = {
    'svr__kernel': ['linear', 'rbf'],
    'svr__C': loguniform(20, 200_000),
    'svr__gamma': expon(scale=1.0),
}

svr_pipeline = make_pipeline(
    preprocessing,
    SVR()
)

random_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions=param_distribs,
    n_iter=50,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42,
)
random_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
svr_random_search_rmse = -random_search.best_score_
print(svr_random_search_rmse)
print(random_search.best_params_)