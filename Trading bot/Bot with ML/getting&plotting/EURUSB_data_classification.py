import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Experimentation.functions import *
from sklearn import *
import plotly as py
from plotly import subplots
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def feature_importance_plot(model):
    n_features = len(features_name)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features_name)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.savefig('feature importance.png')

# Load our CSV Data

dataframe = pd.read_csv('DATA/masterframe.csv')

dataframe.columns = ['date', 'momentum3K', 'momentum3D', 'momentum4K', 'momentum4D', 'momentum5K',
                     'momentum5D', 'momentum8K', 'momentum8D', 'momentum9K', 'momentum9D',
                     'momentum10K', 'momentum10D', 'stoch3K', 'stoch3D', 'stoch4K',
                     'stoch4D', 'stoch5K', 'stoch5D', 'stoch8K', 'stoch8D', 'stoch9K',
                     'stoch9D', 'stoch10K', 'stoch10D', 'will6R', 'will7R', 'will8R',
                     'will9R', 'will10R', 'proc12close', 'proc13close', 'proc14close',
                     'proc15close', 'wadl15close', 'adosc2AD', 'adosc3AD', 'adosc4AD',
                     'adosc5AD', 'cci15close', 'bollinger15upper', 'bollinger15mid',
                     'bollinger15lower', 'heiken15open', 'heiken15high', 'heiken15close',
                     'heiken15low', 'paverage2open', 'paverage2high', 'paverage2low',
                     'paverage2close', 'slope3high', 'slope4high', 'slope5high',
                     'slope10high', 'slope20high', 'slope30high', 'open', 'high', 'low',
                     'close', 'prices']

pr = dataframe['prices']
pr[pr > 0] = 1
pr[pr < 0] = -1
pr[pr == 0] = 0

pr = pr.fillna(0)

# print(pr)

dataframe = dataframe.set_index(pd.to_datetime(dataframe.date))

dataframe = dataframe[['momentum3K', 'momentum3D', 'momentum4K', 'momentum4D', 'momentum5K',
                       'momentum5D', 'momentum8K', 'momentum8D', 'momentum9K', 'momentum9D',
                       'momentum10K', 'momentum10D', 'stoch3K', 'stoch3D', 'stoch4K',
                       'stoch4D', 'stoch5K', 'stoch5D', 'stoch8K', 'stoch8D', 'stoch9K',
                       'stoch9D', 'stoch10K', 'stoch10D', 'will6R', 'will7R', 'will8R',
                       'will9R', 'will10R', 'proc12close', 'proc13close', 'proc14close',
                       'proc15close', 'wadl15close', 'adosc2AD', 'adosc3AD', 'adosc4AD',
                       'adosc5AD', 'cci15close', 'bollinger15upper', 'bollinger15mid',
                       'bollinger15lower', 'heiken15open', 'heiken15high', 'heiken15close',
                       'heiken15low', 'paverage2open', 'paverage2high', 'paverage2low',
                       'paverage2close', 'slope3high', 'slope4high', 'slope5high',
                       'slope10high', 'slope20high', 'slope30high']]

dataframe.fillna(0)

data = dataframe.to_numpy()

# print("data for X_train: {}".format(data))


X_train, X_test, y_train, y_test = train_test_split(data, pr,random_state=0)

# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))

features_name = ['momentum3K', 'momentum3D', 'momentum4K', 'momentum4D', 'momentum5K',
                 'momentum5D', 'momentum8K', 'momentum8D', 'momentum9K', 'momentum9D',
                 'momentum10K', 'momentum10D', 'stoch3K', 'stoch3D', 'stoch4K',
                 'stoch4D', 'stoch5K', 'stoch5D', 'stoch8K', 'stoch8D', 'stoch9K',
                 'stoch9D', 'stoch10K', 'stoch10D', 'will6R', 'will7R', 'will8R',
                 'will9R', 'will10R', 'proc12close', 'proc13close', 'proc14close',
                 'proc15close', 'wadl15close', 'adosc2AD', 'adosc3AD', 'adosc4AD',
                 'adosc5AD', 'cci15close', 'bollinger15upper', 'bollinger15mid',
                 'bollinger15lower', 'heiken15open', 'heiken15high', 'heiken15close',
                 'heiken15low', 'paverage2open', 'paverage2high', 'paverage2low',
                 'paverage2close', 'slope3high', 'slope4high', 'slope5high',
                 'slope10high', 'slope20high', 'slope30high']

# trading_dataframe = pd.DataFrame(X_train, columns=features_name)
""""
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 500)
i = 0
for n_neighbors in neighbors_settings:
    print(i)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
    i += 1
plt.plot(neighbors_settings, training_accuracy)
plt.plot(neighbors_settings, test_accuracy)
plt.show()

"""

forest = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20)
forest.fit(X_train, y_train)
print("Score forest:")
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))


gbrt = GradientBoostingClassifier(random_state=42, learning_rate=0.1)
gbrt.fit(X_train, y_train)

print("Score Boosting classifier:")
print(gbrt.score(X_train, y_train))
print(gbrt.score(X_test, y_test))
feature_importance_plot(gbrt)


min_on = X_train.min(axis=0)
range_on = (X_train - min_on).max(axis=0)
X_train_scaled = (X_train - min_on) / range_on

"""
print(X_train_scaled.min(axis=0))
print(X_train_scaled.max(axis=0))
"""

X_test_scaled = (X_test - min_on) / range_on

svc = SVC(C=1)
svc.fit(X_train_scaled, y_train)
print("Score SVC:")
print(svc.score(X_train_scaled, y_train))
print(svc.score(X_test_scaled, y_test))



""""
knn = KNeighborsClassifier(n_neighbors=500)

knn.fit(X_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=500, p=2,
                     weights='uniform')

y_pred = knn.predict(X_test)

print("Train score:{:.2f}".format(knn.score(X_train, y_train)))

print("Test set predictions: \n {}".format(y_pred))

print("Test score : {:.2f}".format(knn.score(X_test, y_test)))

"""
