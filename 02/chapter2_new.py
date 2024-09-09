from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32

from pandas.plotting import scatter_matrix
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV


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

# check out info about data
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# generate histograms for each of variables
housing.hist(bins=50, figsize=(12, 8))
plt.show()


# create test set out of the dataset, but make it a repeatable test set, so it is independent on each run of program
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# add index column to the housing data set
housing_with_id = housing.reset_index()
# split data set to train and test sets
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# add categorical variable based on median income
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Kategoria dochodów")
plt.ylabel("Liczba dystryktów")
plt.show()

# strata - warstwy (losowanie warstwowe), the best way to divide data set to train and test, with stratify parameter
# WILL USE THIS SPLIT DATA
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

# remove the categorical variable to return to primary version of data after splitting train and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# make copy of train set
housing = strat_train_set.copy()

# ----------------------------------------------------------------------------------------------------------------------
# EXPLORE DATA

# visualization of geo data
housing.plot(kind="scatter",
             x="longitude",
             y="latitude",
             grid=True,
             s=housing["population"] / 100,
             label="population",
             c="median_house_value",
             cmap="jet",
             colorbar=True,
             legend=True,
             sharex=False,
             figsize=(10, 7)
             )
plt.show()

# find correlation between variables
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# add new attributes, because raw don't tell much
housing["pokoje_na_rodzine"] = housing["total_bedrooms"] / housing["households"]
housing["wspolczynnik_sypialni"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["liczba_osob_na_dom"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# PREPARE DATA
# copy again the data and separate labels to other variable
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# populate missing data in total_bedrooms (some records are missing), can be done only on numeric types
imputer = SimpleImputer(strategy="median")
housing_numerics = housing.select_dtypes(include=[np.number])

# impute for each of variables because of safety
imputer.fit(housing_numerics)
print(imputer.statistics_)
X = imputer.transform(housing_numerics)

# put missing values to X
housing_tr = pd.DataFrame(X, columns=housing_numerics.columns, index=housing_numerics.index)
print(housing_tr.info())

# categorical variable
# won't use this encoder
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(housing_cat_encoded[:8])
print(ordinal_encoder.categories_)

# will use this encoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# scaling variables is needed to make models work properly
# it has some drawbacks, can use other scalers instead
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_numerics)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_numerics)

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# this is a manual way to inverse transform outcome, because of the logarithmic nature of data
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# a class for the previous thing
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
print(predictions)

# custom transformers
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[['population']])
print(log_pop)

rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[['housing_median_age']])
print(age_simil_35)

# similarity between every district and san francisco
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[['latitude', 'longitude']])
print(sf_simil)

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio = ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
print(ratio)

# cluster similarity estimator
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

#
# cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
# similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
#                                            sample_weight=housing_labels)
#
# housing_renamed = housing.rename(columns={
#     "latitude": "Latitude", "longitude": "Longitude",
#     "population": "Population",
#     "median_house_value": "Median house value (ᴜsᴅ)"})
# housing_renamed["Max cluster similarity"] = similarities.max(axis=1)
#
# housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
#                      s=housing_renamed["Population"] / 100, label="Population",
#                      c="Max cluster similarity",
#                      cmap="jet", colorbar=True,
#                      legend=True, sharex=False, figsize=(10, 7))
# plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
#          cluster_simil.kmeans_.cluster_centers_[:, 0],
#          linestyle="", color="black", marker="X", markersize=20,
#          label="Cluster centers")
# plt.legend(loc="upper right")
# plt.show()

# transform pipelines
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy="median")),
    ('standardize', StandardScaler())
])

housing_num_prepared = num_pipeline.fit_transform(housing_numerics)
print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_numerics.index
)
print(df_housing_num_prepared)

# column transformer, pass numeric and categorical pipelines and data
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
               "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

print(preprocessing)

# auto naming transformers with this function
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared)

df_housing_prepared = pd.DataFrame(
    housing_prepared, columns=preprocessing.get_feature_names_out(),
    index=housing.index
)
print(df_housing_prepared)


# ----------------------------------------------------------------------------------------------------------------------
# summarized - make a whole pipeline to do end to end example
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
print(housing_prepared)
print(preprocessing.get_feature_names_out())

# ----------------------------------------------------------------------------------------------------------------------
# choose model and start learning

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

# predict results
housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)

lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(lin_rmse)

# check out different, more complex model
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(tree_rmse)

# cross validation to make more trustworthy results
tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())

# linear regression and this tree are more or less equally bad, so lets try different model
forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))

# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmses).describe())

# still bad, so lets make some grid search in search of best hyperparams
full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {"preprocessing__geo__n_clusters": [5, 8, 10],
     "random_forest__max_features": [4, 6, 8]},
    {"preprocessing__geo__n_clusters": [10, 15],
     "random_forest__max_features": [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
# grid_search.fit(housing, housing_labels)
#
# print(grid_search.best_params_)
#
# cv_res = pd.DataFrame(grid_search.cv_results_)
# cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
# code from website
# cv_res = cv_res[["param_preprocessing__geo__n_clusters",
#                  "param_random_forest__max_features", "split0_test_score",
#                  "split1_test_score", "split2_test_score", "mean_test_score"]]
# score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
# cv_res.columns = ["n_clusters", "max_features"] + score_cols
# cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
#
# print(cv_res.head())

# random search of hyperparams
param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions=param_distribs,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42
)

rnd_search.fit(housing, housing_labels)

print(rnd_search.best_params_)