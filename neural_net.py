from sklearn.base import BaseEstimator
from sknn.mlp import Layer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sknn.mlp import Classifier as MLP
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
path = "validation_set/"



X_train = pd.read_csv(path+"X_train.csv",index_col=0)
X_test = pd.read_csv(path+"X_test.csv",index_col=0)
y_train = pd.Series.from_csv(path+"y_train.csv")
y_test = pd.Series.from_csv(path+"y_test.csv")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
#########################################
scaler1=StandardScaler()
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
nn = MLP(
layers=[
Layer("Rectifier", units=6000),
        Layer("Softmax")],
        learning_rule = 'momentum', learning_rate=0.01, batch_size = 150,dropout_rate = 0.2,
        n_iter=20,
        verbose = 1, 
        valid_size = 0.1, 
        n_stable = 15,
        debug = True,
        #    regularize = 'L2'
)

#nn = MLP(layers=[
#        Layer("Rectifier", units=100, pieces=2),
#        Layer("Softmax")],
#    learning_rate=0.001,    n_iter=25)

clf = Pipeline([
    ("scaler", scaler1),
    ('neural network', nn)
    ])

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print accuracy_score(pred, y_test)
#print accuracy_score(clf.predict(X_train),y_train)
