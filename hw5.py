
# coding: utf-8

# In[1]:


# dataframe management
import pandas as pd     
# numerical computation
import numpy as np
# visualization library
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})   
# import matplotlib and allow it to plot inline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score


# In[2]:


import importlib #importlib.reload(WhatToReimport)
import hw5
importlib.reload(hw5)


# In[3]:


d=hw5.Dataset()


# # Data exploration

# In[4]:


d.data.describe()


# In[5]:


d.data.shape


# # Preprocessing

# ## Missing values of the target feature

# In[6]:


nulls = d.data.isnull().sum()
sorted([(x,y) for (x,y) in zip(nulls.index, nulls) if y>0], key=lambda x: x[1], reverse=True)


# We have to manage all these missing values. <br>
# First of all I will remove all the rows that have the target feature "Empathy" to null because they have no use. 

# In[7]:


#removing the rows in which the Empathy attrivute is null
#they are not necessary for train or testing
nullsEmpathy = d.data["Empathy"].isnull().sum()
#nullsEmpathy = 5
print("Number of rows with Empathy that is null: "+str(nullsEmpathy))
d.data = d.data[d.data["Empathy"].notna()]
print("Number of rows with Empathy that is null after: "+str(d.data["Empathy"].isnull().sum()))


# ## Dealing with the categorical variables

# Now I have to deal with the categorical variables. <br>
# The first thing that I have to do is to impute the missing values of them. I will use the mode() (which is the most common value for each feature) to impute them.

# In[8]:


categorical=d.data.select_dtypes(include="object", exclude="float")


# In[9]:


d.data = d.data.select_dtypes(exclude="object")


# In[10]:


categorical.mode().loc[0]


# In[11]:


print(categorical.isnull().sum())
categorical = categorical.fillna(categorical.mode().loc[0])
print(categorical.isnull().sum())


# ### From categorical to scale

# From various attempts it turns out that one-hot encoding of all th variables leads to bad results. <br>
# From the theory we can understand this result because one hot encoding leads to have too many features and, moreover, the values of this categorical attributes are actually in a scale of values even if they are strings, to the best thing to do is to turn them in integers with a scale. (As done below)<br>
# I will do one-hot encoding only fot the binary features where the two values represents different things.

# In[12]:


categorical.shape


# In[13]:


categorical.columns


# In[14]:


categorical.columns=['Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet_usage',
       'Gender', 'Left_right_handed', 'Education', 'Only_child',
       'Village_town', 'House_block_of_flats']


# In[15]:


categorical.describe()


# In[16]:


categorical.Smoking.unique()


# In[17]:


for row in categorical.itertuples():#range(len(categorical["Smoking"])):
    #print(row)
    #if(i==607 or i==722 or i==845 or i==858 or i==921 ):
    #    continue
    #print(row.Smoking)
    #print(row.Index)
    if(row.Smoking=="never smoked"):
        categorical['Smoking'][row.Index]=1
        continue
    if(row.Smoking=="tried smoking"):
        categorical['Smoking'][row.Index]=2
        continue
    if(row.Smoking=="former smoker"):
        categorical['Smoking'][row.Index]=3
        continue
    if(row.Smoking=="current smoker"):
        categorical['Smoking'][row.Index]=4
        continue


# In[18]:


categorical.Smoking.unique()


# In[19]:


categorical.Alcohol.unique()


# In[20]:


for row in categorical.itertuples():
    if(row.Alcohol=="never"):
        categorical['Alcohol'][row.Index]=1
        continue
    if(row.Alcohol=="social drinker"):
        categorical['Alcohol'][row.Index]=2
        continue
    if(row.Alcohol=="drink a lot"):
        categorical['Alcohol'][row.Index]=3
        continue


# In[21]:


categorical.Alcohol.unique()


# In[22]:


categorical.Punctuality.unique()


# In[23]:


for row in categorical.itertuples():
    if(row.Punctuality=="i am often running late"):
        categorical['Punctuality'][row.Index]=1
        continue
    if(row.Punctuality=="i am always on time"):
        categorical['Punctuality'][row.Index]=2
        continue
    if(row.Punctuality=="i am often early"):
        categorical['Punctuality'][row.Index]=3
        continue


# In[24]:


categorical.Punctuality.unique()


# In[25]:


categorical.Lying.unique()


# In[26]:


for row in categorical.itertuples():
    if(row.Lying=="everytime it suits me"):
        categorical['Lying'][row.Index]=1
        continue
    if(row.Lying=="sometimes"):
        categorical['Lying'][row.Index]=2
        continue
    if(row.Lying=="only to avoid hurting someone"):
        categorical['Lying'][row.Index]=3
        continue
    if(row.Lying=="never"):
        categorical['Lying'][row.Index]=4
        continue


# In[27]:


categorical.Lying.unique()


# In[28]:


categorical.Internet_usage.unique()


# In[29]:


for row in categorical.itertuples():
    if(row.Internet_usage=="most of the day"):
        categorical['Internet_usage'][row.Index]=1
        continue
    if(row.Internet_usage=="few hours a day"):
        categorical['Internet_usage'][row.Index]=2
        continue
    if(row.Internet_usage=="less than an hour a day"):
        categorical['Internet_usage'][row.Index]=3
        continue
    if(row.Internet_usage=="no time at all"):
        categorical['Internet_usage'][row.Index]=4
        continue


# In[30]:


categorical.Internet_usage.unique()


# In[31]:


categorical.Education.unique()


# In[32]:


for row in categorical.itertuples():
    if(row.Education=="currently a primary school pupil"):
        categorical['Education'][row.Index]=1
        continue
    if(row.Education=="primary school"):
        categorical['Education'][row.Index]=2
        continue
    if(row.Education=="secondary school"):
        categorical['Education'][row.Index]=3
        continue
    if(row.Education=="college/bachelor degree"):
        categorical['Education'][row.Index]=4
        continue
    if(row.Education=="masters degree"):
        categorical['Education'][row.Index]=5
        continue
    if(row.Education=="doctorate degree"):
        categorical['Education'][row.Index]=6
        continue


# In[33]:


categorical.Education.unique()


# In[34]:


categorical.describe()


# #### One-hot encoding of categorical variables that are left

# In[35]:


categorical.shape


# In[36]:


categorical["Smoking"]=categorical["Smoking"].astype("float64")
categorical["Alcohol"]=categorical["Alcohol"].astype("float64")
categorical["Punctuality"]=categorical["Punctuality"].astype("float64")
categorical["Lying"]=categorical["Lying"].astype("float64")
categorical["Internet_usage"]=categorical["Internet_usage"].astype("float64")
categorical["Education"]=categorical["Education"].astype("float64")


# In[37]:


categorical.dtypes


# In[38]:


categorical2=categorical.select_dtypes(include="object", exclude="float64")
categorical = categorical.select_dtypes(exclude="object")


# In[39]:


categoricalDummied = pd.get_dummies(categorical2)


# In[40]:


categoricalDummied.shape


# #### Imputation of missing values for the numerical features

# I wil use the mean value of each attribute to impute the value of missing values for numerical features

# In[41]:


d.data=d.data.fillna(d.data.mean())


# ## Outliers: Boxplot and Winsorizing

# In[42]:


d.data.quantile(.99).sort_values(ascending=False).head(8)


# In[43]:


def q(col, quant, f):
    t = d.data[col].quantile(quant)
    print(f'col {col} at {quant}-th quantile => {t}')
    d.data.loc[f(d.data[col], t), col] = t


# In[44]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1})
sns.boxplot(y="Height", data=d.data)


# In[45]:


q("Height", .99, lambda x, y: x > y)
q("Height", .1, lambda x,y: x < y)
sns.boxplot(y="Height", data=d.data)


# In[46]:


sns.boxplot(y="Weight", data=d.data)


# In[47]:


q("Weight", .99, lambda x, y: x > y)
q("Weight", .1, lambda x,y: x < y)
sns.boxplot(y="Weight", data=d.data)


# In[48]:


sns.boxplot(y="Age", data=d.data)


# In[49]:


q("Age", .95, lambda x, y: x > y)
#q("Age", .1, lambda x,y: x < y)
sns.boxplot(y="Age", data=d.data)


# ## Normalization of Numerical Variables

# In[50]:


scaler = MinMaxScaler(feature_range=(1, 5), copy=True)
scaled_df = scaler.fit_transform(d.data)
scaled_df = pd.DataFrame(scaled_df, columns=d.data.columns)


# In[51]:


d.data=scaled_df


# In[52]:


d.data= pd.concat([d.data,categoricalDummied,categorical],axis=1,join='inner')


# In[53]:


d.data.describe()


# In[54]:


nulls = d.data.isnull().sum()
sorted([(x,y) for (x,y) in zip(nulls.index, nulls) if y>0], key=lambda x: x[1], reverse=True)


# In[55]:


d.data.shape


# In[56]:


# compute the skewness but only for non missing variables (we already imputed them but just in case ...)
skewed_feats = d.data.apply(lambda x: skew(x.dropna()))
skewness = pd.DataFrame({"Variable":skewed_feats.index, "Skewness":skewed_feats.data})
skewness = skewness.sort_values('Skewness', ascending=[0])
f, ax = plt.subplots(figsize=(23,15))
plt.xticks(rotation='90')
sns.barplot(x=skewness['Variable'], y=skewness['Skewness'])
plt.ylim(0,1.5)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Skewness', fontsize=15)
plt.title('', fontsize=15)


# # Training and Testing sets

# In[57]:


X = d.data.drop(columns=['Empathy'])
Y = d.data['Empathy']


# I have now to trasform the target feature from a scale from 1 to 5 to a binary variable 0 (if the vale is 1,2 or 3) and 1 (4 ot 5).

# In[58]:


def getBinary(x):
    res=[]
    for i in range(len(x)):
        if(x[i]<=3):
            res.append(0)
        else:
            res.append(1)
    res = np.array(res)
    return res


# # Baseline

# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=40)


# I will use as baseline a dump predictor that predict always the most frequent.

# In[60]:


Y_train=getBinary(Y_train.values)
Y_test=getBinary(Y_test.values)
Y=getBinary(Y.values)


# In[61]:


def trainBaseline(x):
    return scipy.stats.mode(x)[0][0]


# In[62]:


mode=trainBaseline(Y_train)


# In[63]:


def predictBaseline(test,mostfrequent):
    res=[]
    for i in range(len(test)):
        res.append(mostfrequent)
    return np.array(res)


# In[64]:


predictions=predictBaseline(X_test,mode)


# In[65]:


np.mean(predictions==Y_test)


# Usign the 20% of my dataset as testing set, this base predictor has an accuracy of 66%

# # Model1

# I start trying a Logistic Regression algoritm, I choose as solver the 'liblinear' one that should be one of the most suitable for binary classification in small databases. <br>
# I'll use the L2 norm for the penalization and a value of C very small. C is the inverse of regularization strength, like in support vector machines, smaller values specify stronger regularization.

# In[66]:


from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(solver='liblinear',random_state=123, C=0.1,penalty='l2', multi_class='ovr')
logReg.fit(X_train, Y_train)


# In[67]:


predict_train = logReg.predict(X_train) 


# In[68]:


np.mean(predict_train == Y_train)


# In[69]:


predict_test = logReg.predict(X_test) 


# In[70]:


np.mean(predict_test == Y_test)


# We have already achieved an accuracy of 82% on the training set and an accuracy of 70% on the testing one.

# Now I will try to have a better idea of the real accuracy that that model can reach using a Stratified K-Fold cross validation.

# In[71]:


X.shape


# In[72]:


np.mean(cross_val_score(logReg, X, Y, cv=150))


# # Model2

# In[73]:


def svc_param_selection(X_t, Y_t, n):
    params = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1]}
    gs = GridSearchCV(svm.SVC(kernel='rbf'), params, cv=n,n_jobs=-1)
    gs.fit(X_t, Y_t)
    return gs.best_params_


# In[74]:


svc_param_selection(X_train,Y_train,20)


# In[75]:


model2=svm.SVC(kernel='rbf',C= 10, gamma= 0.01)
model2.fit(X_train,Y_train)


# In[76]:


prediction2=model2.predict(X_test)


# In[77]:


np.mean(prediction2 == Y_test)


# # Model 3

# In[57]:


# Heavy to run:
rfc = RandomForestClassifier()
params = {'n_estimators': [4, 15,20,50,100,200,250,300,350], 
        #'n_estimators': [4, 6, 9,15,20,50,100,150,200,250,300,350], 
              #'max_features': ['log2', 'sqrt','auto'],
              'max_features': ['auto'],
              #'criterion': ['entropy', 'gini'],
              'criterion': ['gini'],
              'max_depth': [3, 5, 10,50,100,250], 
              'min_samples_split': [2,3, 5,10,15],
              'min_samples_leaf': [1,5,8,18]
              #'max_depth': [2, 3, 5, 10,15,20,25,30,50,100], 
              #'min_samples_split': [2, 3, 5,10],
              #'min_samples_leaf': [1,5,8]
             }
gs = GridSearchCV(clf, params,iid=False,cv=10,n_jobs=-1)
gs = gs.fit(X_train, Y_train)
rfc = gs.best_estimator_
rfc.fit(X_train, Y_train)


# In[58]:


gs.best_params_


# In[59]:


prediction3=rfc.predict(X_test)


# In[60]:


np.mean(prediction3 == Y_test)


# Redone with preprocessed data:

# In[78]:


model3=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=129,  
            warm_start=False)
model3.fit(X_train,Y_train)
predict3=model3.predict(X_test)
np.mean(predict3 == Y_test)


# In[79]:


f1_score(Y_test,predict3)


# # Model 4

# In[80]:


xg_reg = xgb.XGBClassifier(n_estimators = 300)


# In[81]:


xg_reg.fit(X_train,Y_train)

preds = xg_reg.predict(X_test)


# In[82]:


xg_reg.score(X_test,Y_test) 


# In[74]:


#heavy to run
xgboostClass = xgb.XGBClassifier()
parameters = {'n_estimators': [4, 15,20,50,100,200,250,300,350,500,1000],
              'objective': ['reg:logistic','binary:logistic'],
              'max_depth': [3, 5, 10,50,100,250],
              'learning_rate': [0.1,0.5,0.3,0.8,0.01,0.003],
              "subsample": [0.6, 0.4],
              "colsample_bytree": [0.7, 0.3],
              "gamma": [0, 0.5 ]
             }


# In[75]:


grid_obj = GridSearchCV(xgboostClass, parameters,iid=False,cv=10,n_jobs=-1)
grid_obj = grid_obj.fit(X_train, Y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, Y_train)
grid_obj.best_estimator_


# In[79]:


predict4=clf.predict(X_test)
np.mean(predict4 == Y_test) #


# Redone with preprocessed data:

# In[83]:


model4 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.5, learning_rate=0.001,
       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
       n_estimators=6000, n_jobs=1, nthread=None, objective='reg:logistic',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.6)


# In[84]:


model4.fit(X_train, Y_train)


# In[85]:


predict4=model4.predict(X_test)


# In[86]:


np.mean(predict4 == Y_test)


# ## Model 5

# In[87]:


model5 = RidgeClassifier(random_state=123)
model5 = model5.fit(X_train, Y_train)
predict5 = model5.predict(X_test)
np.mean(predict5==Y_test)


# # Model 6

# In[88]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(6, activation=tf.nn.relu),
  tf.keras.layers.Dense(8, activation=tf.nn.tanh),
  tf.keras.layers.Dense(8, activation=tf.nn.tanh),
  tf.keras.layers.Dense(8, activation=tf.nn.tanh),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train.values, Y_train , epochs=200)


# In[89]:


predictionNN=model.predict(X_test.values)
res=[]
for i in range(len(predictionNN)):
    if(predictionNN[i]<=0.5):
        res.append(0)
    else:
        res.append(1)
predictionNN = np.array(res)
np.mean(predictionNN==Y_test)


# # Model 7

# In[90]:


model7=AdaBoostClassifier(model3,n_estimators=2,random_state=123)
model7.fit(X_train,Y_train)


# In[91]:


pred7=model7.predict(X_test)
np.mean(pred7==Y_test)


# In[92]:


np.mean(cross_val_score(model7, X_test, Y_test, cv=15))


# In[93]:


f1_score(Y_test,pred7)


# # Dimensionality reduction

# ## PCA

# In[94]:


X.shape


# In[95]:


X_train.shape


# In[96]:


pca = PCA(n_components=130)
X_trainPCA = pca.fit_transform(X_train)
X_testPCA = pca.fit_transform(X_test)
print(X_trainPCA.shape)


# In[97]:


model=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=223,  
            warm_start=False)
model.fit(X_trainPCA,Y_train)
predict=model.predict(X_testPCA)
np.mean(predict == Y_test)


# Tuning of PCA

# In[98]:


for i in [50,80,100,115,130,150]:
    pca = PCA(n_components=i)
    X_trainPCA = pca.fit_transform(X_train)
    X_testPCA = pca.fit_transform(X_test)
    print("number of feature of PCA: "+str(X_trainPCA.shape[1]))
    model=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=223,  
            warm_start=False)
    model.fit(X_trainPCA,Y_train)
    predict6=model.predict(X_testPCA)
    print(np.mean(predict6 == Y_test))
    print("\n")


# ## Feature Selection

# In[99]:


model_simple = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
model_simple = model_simple.fit(X_train, Y_train)


# In[100]:


# Get numerical feature importances
importances = list(model_simple.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[101]:


# list of x locations for plotting
x_values = list(range(len(importances)))
    
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]

# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)

fig = plt.figure(figsize = (23,10))
# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')

# Draw line at 96% of importance retained
plt.hlines(y = 0.96, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')

# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')

# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');
    


# In[102]:


f=[]
for i in range(120):
    f.append(feature_importances[i][0])
f=np.array(f)


# In[103]:


cov=X_train[f[:20]].corr(method='pearson')
#cm = sns.clustermap(cov, annot=True, center=0, cmap="Blues", figsize=(100, 100))
#cm.cax.set_visible(False)
fig, ax = plt.subplots(figsize=(30,30)) 
cm=sns.heatmap(cov, annot=True, center=0, cmap="Blues")


# In[104]:


X_train[f].shape


# In[105]:


model3_featureSelection=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=100, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=None,
            oob_score=False, random_state=223,  
            warm_start=False)
model3_featureSelection.fit(X_train[f],Y_train)
predict3_featureSelection=model3_featureSelection.predict(X_test[f])
np.mean(predict3_featureSelection == Y_test)


# In[106]:


xgb_featureSelection = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.5, learning_rate=0.001,
       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
       n_estimators=6000, n_jobs=-1, nthread=None, objective='reg:logistic',
       random_state=123, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.6)
xgb_featureSelection.fit(X_train[f], Y_train)
pred=xgb_featureSelection.predict(X_test[f])
np.mean(pred == Y_test)

