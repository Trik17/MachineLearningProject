# coding: utf-8
# dataframe management
import pandas as pd
# numerical computation
import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import pickle

print("loading the data preprocessed from files")
Y_train = np.loadtxt("Y_train.csv", delimiter=",")
Y_test = np.loadtxt("Y_test.csv", delimiter=",")
X_train = pd.read_csv('X_train.csv', sep='\t', encoding='utf-8')
X_test = pd.read_csv('X_test.csv', sep='\t', encoding='utf-8')
X_train = X_train.drop(columns=['Unnamed: 0'])
X_test = X_test.drop(columns=['Unnamed: 0'])

print("\nModel 1 training ")
logReg = LogisticRegression(solver='liblinear', random_state=123, C=0.1, penalty='l2', multi_class='ovr')
logReg.fit(X_train, Y_train)
print("\nModel 2 training ")
model2 = svm.SVC(kernel='rbf', C=10, gamma=0.01)
model2.fit(X_train, Y_train)
print("\nModel 3 training ")
model3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=100, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=5,
                                min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
                                oob_score=False, random_state=129,
                                warm_start=False)
model3.fit(X_train, Y_train)
print("\nModel 4 training ")
model4 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                           colsample_bytree=0.3, gamma=0.5, learning_rate=0.001,
                           max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
                           n_estimators=6000, n_jobs=-2, nthread=None, objective='reg:logistic',
                           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                           seed=None, silent=True, subsample=0.6)
model4.fit(X_train, Y_train)

print("\nModel 5 training ")
model5 = RidgeClassifier(random_state=123)
model5 = model5.fit(X_train, Y_train)

print(
    "\nModel 6 is a NN but is present only in the jupiter notebook \n because of its very bad performances (due to the very small dataset)")

print("\nModel 7 training ")
model7 = AdaBoostClassifier(model3, n_estimators=2, random_state=123)
model7.fit(X_train, Y_train)

# --- Feature selection
print("\nCalculating most important features")
model_simple = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                      max_depth=100, max_features='auto', max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                      min_samples_leaf=1, min_samples_split=5,
                                      min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
                                      oob_score=False, random_state=None, verbose=0,
                                      warm_start=False)
model_simple = model_simple.fit(X_train, Y_train)

# Get numerical feature importances
importances = list(model_simple.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

f = []
for i in range(120):
    f.append(feature_importances[i][0])
f = np.array(f)

# --

model3_featureSelection = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                                 max_depth=100, max_features='auto', max_leaf_nodes=None,
                                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                                 min_samples_leaf=1, min_samples_split=5,
                                                 min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
                                                 oob_score=False, random_state=223,
                                                 warm_start=False)
model3_featureSelection.fit(X_train[f], Y_train)

xgb_featureSelection = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                         colsample_bytree=0.3, gamma=0.5, learning_rate=0.001,
                                         max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
                                         n_estimators=6000, n_jobs=-1, nthread=None, objective='reg:logistic',
                                         random_state=123, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                         seed=None, silent=True, subsample=0.6)
xgb_featureSelection.fit(X_train[f], Y_train)
print("\n Training Model 3 and Model 4 with the first 120 most important features")

print("\nSaving the model on files .pickle")

with open('model1.pickle', 'wb') as handle:
    pickle.dump(logReg, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model2.pickle', 'wb') as handle:
    pickle.dump(model2, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model3.pickle', 'wb') as handle:
    pickle.dump(model3, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model4.pickle', 'wb') as handle:
    pickle.dump(model4, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model5.pickle', 'wb') as handle:
    pickle.dump(model5, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model7.pickle', 'wb') as handle:
    pickle.dump(model7, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model3_featureSelection.pickle', 'wb') as handle:
    pickle.dump(model3_featureSelection, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model4_featureSelection.pickle', 'wb') as handle:
    pickle.dump(xgb_featureSelection, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('featureSelection.pickle', 'wb') as handle:
    pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)