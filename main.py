import math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyClassifier
from IPython.display import clear_output
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error , mean_absolute_percentage_error , mean_squared_error , accuracy_score
from sklearn.ensemble import RandomForestClassifier

#Questions to be answered
    #Can survey questions from the BRFSS provide accurate predictions of whether an individual has diabetes?
    #What risk factors are most predictive of diabetes risk?
    #Can we use a subset of the risk factors to accurately predict whether an individual has diabetes?
    #Can we create a short form of questions from the BRFSS using feature selection to accurately predict
    #if someone might have diabetes or is at high risk of diabetes?

diabetes_1 = pd.read_csv(r"C:\Users\ASUS TUFF GAMING\Downloads\Data Diabetes\diabetes_binary_health_indicators_BRFSS2015.csv")
print('Data Exploration')
print(diabetes_1.describe())
print(diabetes_1.head())
print(diabetes_1.info())
print(diabetes_1.isnull().sum())
diabetes_1.duplicated().sum()
diabetes_1.drop_duplicates(inplace = True)

#There are no null data and all duplicates are removed.
#Data Visualization

diabetes_columns = diabetes_1.columns.tolist()
print(diabetes_columns)

for i in diabetes_columns:
    diabetes_1[i] = diabetes_1[i].astype(int)

#for x in diabetes_columns:
        #sns.histplot(diabetes_1[x])
        #plt.title(x)
        #plt.show()

diabetes_columns = diabetes_1.columns.tolist()
print(diabetes_columns)

for a in diabetes_columns:
    print(diabetes_1[a].unique().tolist())

for i in diabetes_columns:
    diabetes_1[i] = diabetes_1[i].astype(int)

print(diabetes_1.info())

sns.heatmap(diabetes_1.corr(),linewidths=1,cmap=plt.cm.Reds, annot=True);
plt.title('Heatmap for correlation between columns of training data')
plt.show()

X = diabetes_1.iloc[:,1:]
Y = diabetes_1.iloc[:,0]

BestFeatures = SelectKBest(score_func=chi2, k=10)
fit = BestFeatures.fit(X,Y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

f_Scores = pd.concat([df_columns,df_scores],axis=1)
f_Scores.columns = ['Feature','Score']

print(f_Scores)

#columns_elimination_1 = ["Fruits" , "Veggies" , "Sex" , "CholCheck" , "AnyHealthcare", "NoDocbcCost", "Smoker"]
new_data = diabetes_1
#drop(columns_elimination_1, axis=1)

X = new_data.drop("Diabetes_binary",axis=1)
Y = new_data["Diabetes_binary"]

from imblearn.under_sampling import NearMiss
nm = NearMiss(version = 1 , n_neighbors = 10)
x_sm,y_sm= nm.fit_resample(X,Y)

print(y_sm.value_counts())
X_train, X_test, Y_train, Y_test = train_test_split(x_sm,y_sm, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

#Alright Let's Go! LAZY REGRESSION!

#clf = LazyClassifier(verbose=0,
                     #ignore_warnings=True,
                     #custom_metric=None,
                     #predictions=False,
                     #random_state=12,
                     #classifiers='all')

#models, predictions = clf.fit(X_train , X_test , Y_train , Y_test)
#clear_output()

#print(models[:10])

#Best Models According to Lazy Regression = RandomForestClassifier, LGBMC Classifier, ExtraTreesClasifier, XBGC Classifier, Bagging Classifier

rf = RandomForestClassifier()
#grid_search = GridSearchCV(rf, param_grid, cv=5)
#grid_search.fit(X_train, Y_train)
#best_params = grid_search.best_params_
rf = RandomForestClassifier(n_estimators=100, max_features=16 , max_depth=16, random_state = 69)
rf.fit(X_train, Y_train)
y_pred=rf.predict(X_test)
print('Training set score: {:.4f}'.format(rf.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(rf.score(X_test, Y_test)))
mse =mean_squared_error(Y_test, y_pred)
print('Mean Squared Error : '+ str(mse))
rmse = math.sqrt(mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error : '+ str(rmse))
matrix = classification_report(Y_test,y_pred )
print(matrix)

from lightgbm import LGBMClassifier
lgbm = LGBMClassifier (objective='binary',
    boosting='gbdt',
    learning_rate = 0.09,
    max_depth = 8,
    num_leaves = 85,
    n_estimators = 300,
    bagging_fraction = 0.8,
    feature_fraction = 0.9)

lgbm.fit(X_train, Y_train)
y_pred_2 = lgbm.predict(X_test)
print('Training set score: {:.4f}'.format(lgbm.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(lgbm.score(X_test, Y_test)))
mse =mean_squared_error(Y_test, y_pred_2)
print('Mean Squared Error : '+ str(mse))
rmse = math.sqrt(mean_squared_error(Y_test, y_pred_2))
print('Root Mean Squared Error : '+ str(rmse))
matrix = classification_report(Y_test,y_pred_2)
print(matrix)

from xgboost import XGBClassifier

xgb= XGBClassifier(max_depth= 6, min_child_weight = 3)
xgb.fit(X_train, Y_train)
print(xgb.score(X_train, Y_train))
print(xgb.score(X_test, Y_test))
y_pred_train_xgb = xgb.predict(X_train)
acc_train_xgb = accuracy_score(Y_train, y_pred_train_xgb)

y_pred_test_xgb = xgb.predict(X_test)
acc_test_xgb = accuracy_score(Y_test, y_pred_test_xgb)
print(acc_train_xgb)
print(acc_test_xgb)

print('Training set score: {:.4f}'.format(xgb.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(xgb.score(X_test, Y_test)))
mse =mean_squared_error(Y_test, y_pred)
print('Mean Squared Error : '+ str(mse))
rmse = math.sqrt(mean_squared_error(Y_test, y_pred_test_xgb))
print('Root Mean Squared Error : '+ str(rmse))
matrix = classification_report(Y_test,y_pred_test_xgb)
print(matrix)

from sklearn.svm import SVC
esvece = SVC(kernel='linear', C=1.0)
esvece.fit(X_train, Y_train)
y_pred_3 = esvece.predict(X_test)
print('Training set score: {:.4f}'.format(esvece.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(esvece.score(X_test, Y_test)))
mse =mean_squared_error(Y_test, y_pred_3)
print('Mean Squared Error : '+ str(mse))
rmse = math.sqrt(mean_squared_error(Y_test, y_pred_3))
print('Root Mean Squared Error : '+ str(rmse))
matrix = classification_report(Y_test,y_pred_3)
print(matrix)