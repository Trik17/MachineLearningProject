# coding: utf-8

# dataframe management
import pandas as pd
# numerical computation
import numpy as np
# visualization library
import seaborn as sns
# import matplotlib and allow it to plot inline
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
#from sklearn.linear_model import Ridge,Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import scipy
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import hw5
import pickle



print("loading the data preprocessed from files")
Y_test=np.loadtxt("Y_test.csv",delimiter=",")
X_test= pd.read_csv('X_test.csv', sep='\t', encoding='utf-8')
X_test=X_test.drop(columns=['Unnamed: 0'])



print("\nLoading the models from files .pickle")

with open('model1.pickle', 'rb') as handle:
    logReg = pickle.load(handle)
with open('model2.pickle', 'rb') as handle:
    model2 = pickle.load(handle)
with open('model3.pickle', 'rb') as handle:
    model3 = pickle.load(handle)
with open('model4.pickle', 'rb') as handle:
    model4 = pickle.load(handle)
with open('model5.pickle', 'rb') as handle:
    model5 = pickle.load(handle)
with open('model7.pickle', 'rb') as handle:
    model7 = pickle.load(handle)
with open('model3_featureSelection.pickle', 'rb') as handle:
    model3_featureSelection = pickle.load(handle)
with open('model4_featureSelection.pickle', 'rb') as handle:
    xgb_featureSelection = pickle.load(handle)
with open('featureSelection.pickle', 'rb') as handle:
    f = pickle.load(handle)


def predictBaseline(test):
    mostfrequent = 1
    res=[]
    for i in range(len(test)):
        res.append(mostfrequent)
    return np.array(res)
print("\n\nThe baseline is to predict the most frequent class, it reaches the accuracy:")
predictions=predictBaseline(X_test)
print(np.mean(predictions==Y_test))

print("\nModel 1")
print("Accuracy:")
predict_test = logReg.predict(X_test)
print(np.mean(predict_test == Y_test))

print("\nModel 2")
print("Accuracy:")
prediction2=model2.predict(X_test)
print(np.mean(prediction2 == Y_test))

print("\nModel 3")
print("Accuracy:")
predict3=model3.predict(X_test)
print(np.mean(predict3 == Y_test))
print("F1 score:")
print(f1_score(Y_test,predict3))

print("\nModel 4")
print("Accuracy:")
predict4=model4.predict(X_test)
print(np.mean(predict4 == Y_test))
print("F1 score:")
print(f1_score(Y_test,predict4))

print("\nModel 5")
print("Accuracy:")
predict5 = model5.predict(X_test)
print(np.mean(predict5==Y_test))

print("\nModel 6 is a NN but is present only in the jupiter notebook \n because of its very bad performances (due to the very small dataset)")

print("\nModel 7")
print("Accuracy:")
pred7=model7.predict(X_test)
print(np.mean(pred7==Y_test))
print("F1 score:")
print(f1_score(Y_test,pred7))

print("\nModel 3 with features selection")
print("Accuracy:")
predict3_featureSelection=model3_featureSelection.predict(X_test[f])
print(np.mean(predict3_featureSelection == Y_test))

print("\nModel 4 with features selection")
print("Accuracy:")
pred=xgb_featureSelection.predict(X_test[f])
print(np.mean(pred == Y_test))


# for model persistence:
#https://scikit-learn.org/stable/modules/model_persistence.html
#save=pickle.dumps(model3)
#model3= pickle.loads(s)
#model3.predict(X_test)
#-----------------
#with open('model2.pickle', 'wb') as handle:
#    pickle.dump(model2, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('model1.pickle', 'rb') as handle:
#    b = pickle.load(handle)

