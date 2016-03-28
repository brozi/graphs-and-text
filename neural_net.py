from sklearn.base import BaseEstimator
from sknn.mlp import Layer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sknn.mlp import Classifier as MLP
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
validation = True
path =""
if(validation):
    path = "validation_set/"
else:
    path = "train_set/"

X_train = pd.read_csv(path+"X_train.csv",index_col=0)
X_test = pd.read_csv(path+"X_test.csv",index_col=0)
y_train = pd.Series.from_csv(path+"y_train.csv")
if(validation):
    y_test = pd.Series.from_csv(path+"y_test.csv")

#X_train = X_train.drop(["Number of times to cited"], axis = 1)
#X_test = X_test.drop(["Number of times to cited"], axis = 1)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
#########################################
scaler1=StandardScaler()
scaler = MinMaxScaler(feature_range=(0.0, 1.0))
val_percent = 0.1
nn = MLP(
layers=[
    Layer("Rectifier", units=100),
    Layer("Rectifier", units=100),
        Layer("Softmax")],
        learning_rule = 'momentum', learning_rate=0.005, batch_size = 30,dropout_rate =0.1,
        n_iter=100,
        verbose = 1, 
        valid_size = val_percent, 
        n_stable = 30,
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

clf.fit(X_train.values, y_train.values)
pred = clf.predict(X_test.values).flatten()
if(validation):
    print "hold out validation:",accuracy_score(pred, y_test)
if val_percent>0:
    valid = clf.named_steps["neural network"].valid_set
    print "nn validation:",accuracy_score(clf.named_steps["neural network"].predict(valid[0]),valid[1][:,1])

#print accuracy_score(clf.predict(X_train),y_train)

def make_submission(predicted_label, name = 'submit.csv'):
    submit_d = d = {'id' : pd.Series(np.arange(predicted_label.shape[0]).astype(int)),
                    'category' : pd.Series(predicted_label).astype(int)}
    submit = pd.DataFrame(submit_d, columns=["id","category"])
    submit.to_csv(name,index=False)
    return submit


if(not(validation)):
    submit = make_submission(pred,name = "submit_nn.csv")
