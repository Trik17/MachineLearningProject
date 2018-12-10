# coding: utf-8
# dataframe management
import pandas as pd
# numerical computation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import hw5
d=hw5.Dataset()
print('preprocessing\n')
print('removing the rows in which the Empathy attrivute is null')
nullsEmpathy = d.data["Empathy"].isnull().sum()
d.data = d.data[d.data["Empathy"].notna()]
categorical=d.data.select_dtypes(include="object", exclude="float")
d.data = d.data.select_dtypes(exclude="object")

print('use mode to imput missing values of categorical')
categorical = categorical.fillna(categorical.mode().loc[0])

print('\nturning the categorical features that are scales into numerical scales of values')

categorical.columns=['Smoking', 'Alcohol', 'Punctuality', 'Lying', 'Internet_usage',
       'Gender', 'Left_right_handed', 'Education', 'Only_child',
       'Village_town', 'House_block_of_flats']
for row in categorical.itertuples():
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

categorical["Smoking"]=categorical["Smoking"].astype("float64")
categorical["Alcohol"]=categorical["Alcohol"].astype("float64")
categorical["Punctuality"]=categorical["Punctuality"].astype("float64")
categorical["Lying"]=categorical["Lying"].astype("float64")
categorical["Internet_usage"]=categorical["Internet_usage"].astype("float64")
categorical["Education"]=categorical["Education"].astype("float64")

print("\nOne-hot encoding of categorical variables that are left")
categorical2=categorical.select_dtypes(include="object", exclude="float64")
categorical = categorical.select_dtypes(exclude="object")
categoricalDummied = pd.get_dummies(categorical2)

print("Imputation of missing values for the numerical features")
d.data=d.data.fillna(d.data.mean())

print("\Outliers: Winsorizing")
def q(col, quant, f):
    t = d.data[col].quantile(quant)
    print(f'col {col} at {quant}-th quantile => {t}')
    d.data.loc[f(d.data[col], t), col] = t

q("Height", .99, lambda x, y: x > y)
q("Height", .1, lambda x,y: x < y)
q("Weight", .99, lambda x, y: x > y)
q("Weight", .1, lambda x,y: x < y)
q("Age", .95, lambda x, y: x > y)

print("\nNormalization of Numerical Variables")
scaler = MinMaxScaler(feature_range=(1, 5), copy=True)
scaled_df = scaler.fit_transform(d.data)
scaled_df = pd.DataFrame(scaled_df, columns=d.data.columns)
d.data=scaled_df
d.data= pd.concat([d.data,categoricalDummied,categorical],axis=1,join='inner')

print("\ncreating training (used also as dev thanks to cross-validation) and testing sets")
X = d.data.drop(columns=['Empathy'])
Y = d.data['Empathy']
print("\ntransorm Y from 1-5 into binary")
def getBinary(x):
    res=[]
    for i in range(len(x)):
        if(x[i]<=3):
            res.append(0)
        else:
            res.append(1)
    res = np.array(res)
    return res
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=40)
Y_train=getBinary(Y_train.values)
Y_test=getBinary(Y_test.values)
Y=getBinary(Y.values)


print("\nSaving the csv files of the training and test sets after preproessing ")
np.savetxt("Y_train.csv", Y_train, delimiter=",")
np.savetxt("Y_test.csv", Y_test, delimiter=",")
X_train.to_csv("X_train.csv", sep='\t', encoding='utf-8')
X_test.to_csv("X_test.csv", sep='\t', encoding='utf-8')

