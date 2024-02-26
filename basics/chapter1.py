import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind="scatter",
             grid=True,
             x="GDP per capita (USD)",
             y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# model = LinearRegression()
model = KNeighborsRegressor(n_neighbors=3)

model.fit(X, y)

X_new = [[23_434]]
print(model.predict(X_new))
