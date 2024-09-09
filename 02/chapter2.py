from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint


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


def shuffleAndSplitData(data, testRatio):
    shuffledIndices = np.random.permutation(len(data))
    testSetSize = int(len(data) * testRatio)
    testIndices = shuffledIndices[:testSetSize]
    trainIndices = shuffledIndices[testSetSize:]
    return data.iloc[trainIndices], data.iloc[testIndices]


trainSet, testSet = shuffleAndSplitData(housing, 0.2)


# print(len(trainSet), len(testSet))
# housing.hist(bins=50, figsize=(12, 8))
# plt.show()


def isIdInTestSet(id, testRatio):
    return crc32(np.int64(id)) < testRatio * 2 ** 32


def splitDataWithIdHash(data, testRatio, idColumn):
    ids = data[idColumn]
    inTestSet = ids.apply(lambda id_: isIdInTestSet(id_, testRatio))
    return data.loc[~inTestSet], data.loc[inTestSet]


housingWithId = housing.reset_index()
# trainSet, testSet = splitDataWithIdHash(housingWithId, 0.2, "index")

trainSet, testSet = train_test_split(housing, test_size=0.2, random_state=42)
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Kategoria dochodów")
# plt.ylabel("Liczba dystryktów")
# plt.show()
# print(housing["income_cat"])

# splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# stratSplits = []
# for trainIndex, testIndex in splitter.split(housing, housing["income_cat"]):
#     stratTrainSetN = housing.iloc[trainIndex]
#     stratTestSetN = housing.iloc[testIndex]
#     stratSplits.append([stratTrainSetN, stratTestSetN])
#
# stratTrainSet, stratTestSet = stratSplits[0]
# print(stratTestSet.head())

stratTrainSet, stratTestSet = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
# print(stratTestSet["income_cat"].value_counts() / len(stratTestSet))

for set_ in (stratTrainSet, stratTestSet):
    set_.drop("income_cat", axis=1, inplace=True)

housing = stratTrainSet.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
#              s=housing["population"] / 100, label="population",
#              c="median_house_value", cmap="jet", colorbar=True,
#              legend=True, sharex=False, figsize=(10,7))
# plt.show()
# print(corrMatrix["median_house_value"].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

housing["pokoje_na_rodzine"] = housing["total_rooms"] / housing["households"]
housing["wspolczynnik_sypialni"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["liczba_osob_na_dom"] = housing["population"] / housing["households"]

corrMatrix = housing.corr(numeric_only=True)
# print(corrMatrix["median_house_value"].sort_values(ascending=False))

housing = stratTrainSet.drop("median_house_value", axis=1)
housingLabels = stratTrainSet["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housingNum = housing.select_dtypes(include=[np.number])
imputer.fit(housingNum)

# print(imputer.statistics_)
# print(housingNum.median().values)

X = imputer.transform(housingNum)

housingTr = pd.DataFrame(X, columns=housingNum.columns, index=housingNum.index)
# print(housingTr.head())

housingCat = housing[["ocean_proximity"]]
# print(housingCat.head(10))
# print(housing.head())

ordinalEncoder = OrdinalEncoder()
housingCatEncoded = ordinalEncoder.fit_transform(housingCat)
# print(housingCatEncoded[:10])
# print(ordinalEncoder.categories_)

catEncoder = OneHotEncoder()
housingCat1Hot = catEncoder.fit_transform(housingCat)
# print(catEncoder.categories_)

minMaxScaler = MinMaxScaler(feature_range=(-1, 1))
housingNumMinMaxScaled = minMaxScaler.fit_transform(housingNum)
# print(housingNumMinMaxScaled)

stdScaler = StandardScaler()
housingNumStdScaled = stdScaler.fit_transform(housingNum)
# print(housingNumMinMaxScaled)

ageSimil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
# print(ageSimil_35)

targetScaler = StandardScaler()
scaledLabels = targetScaler.fit_transform(housingLabels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaledLabels)
someNewData = housing[["median_income"]].iloc[:5]

scaledPredictions = model.predict(someNewData)
predictions = targetScaler.inverse_transform(scaledPredictions)
# print(predictions)

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housingLabels)
predictions = model.predict(someNewData)
# print(predictions)

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
logPop = log_transformer.transform(housing[["population"]])
# print(logPop)

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])


# print(sf_simil)

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Podobienstwo {i} skupienia" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similrities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housingLabels)
# print(similrities[:3].round(2))

from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="median")),
    ('standardize', StandardScaler()),
])

import sklearn

sklearn.set_config(display="diagram")

from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housingNum)
# print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housingNum.index
)
# print(df_housing_num_prepared.head())

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)
df_housing_prepared = pd.DataFrame(
    housing_prepared, columns=preprocessing.get_feature_names_out(),
    index=housingNum.index
)


# print(df_housing_prepared.head().to_string())

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )


log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing = ColumnTransformer([
    ("wspolczynnik_sypialni", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("pokoje_na_rodzine", ratio_pipeline(), ["total_rooms", "households"]),
    ("liczba_osob_na_dom", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object))
], remainder=default_num_pipeline)

housing_prepared = preprocessing.fit_transform(housing)
# print(preprocessing.get_feature_names_out())
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housingLabels)
housing_predictions = lin_reg.predict(housing)
# print(housing_predictions[:5].round(-2))
# print(housingLabels.iloc[:5].values)

lin_rmse = mean_squared_error(housingLabels, housing_predictions, squared=False)
# print(lin_rmse)

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housingLabels)

# housing_predictions = tree_reg.predict(housing)
# tree_rmse = mean_squared_error(housingLabels, housing_predictions, squared=False)
# print(tree_rmse)

# tree_rmses = -cross_val_score(tree_reg, housing, housingLabels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(tree_rmses).describe())

# forest_reg = make_pipeline(preprocessing,
#                            RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing, housingLabels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmses).describe())

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
# grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
# grid_search.fit(housing, housingLabels)
# print(grid_search.best_params_)

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}
rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42
)
rnd_search.fit(housing, housingLabels)

final_model = rnd_search.best_estimator_
# feature_importances = final_model["random_forest"].feature_importances_
# print(sorted(zip(feature_importances,
#                  final_model["preprocessing"].get_feature_names_out()),
#              reverse=True))

# X_test = stratTestSet.drop("median_house_value", axis=1)
# y_test = stratTestSet["median_house_value"].copy()
#
# final_predictions = final_model.predict(X_test)
#
# final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
# print(final_rmse)

import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")
