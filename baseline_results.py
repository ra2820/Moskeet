import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support,classification_report,mean_squared_error,accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV



#data = pd.read_csv('./7_AMCA_Cleaned.csv')
#data = pd.read_csv('./Culex_features.csv')
#data = pd.read_csv('./Entire_data.csv')
data = pd.read_csv('./All_features.csv')
#data = pd.read_csv('./All_features_limited2.csv')


# displays performance of classifier 
# input : test labels (y_test), predicted labels (y_pred)
# output : accuracy, precision, recall, f1, MSE, confusion matrix
def display_results(y_test,y_pred): 
  accuracy = balanced_accuracy_score(y_test,y_pred)
  #accuracy = accuracy_score(y_test,y_pred)
  report = classification_report(y_test,y_pred,zero_division=0)
  error = mean_squared_error(y_test,y_pred)
  cm = confusion_matrix(y_test, predictions)
  
  print("Accuracy:",accuracy)
  print()
  print("Other metrics:")
  print(report)
  print()
  print("MSE:")
  print(error)
  print()
  print("Confusion Matrix:")
  print(cm)
  print()
  print()
  plt.figure(figsize = (16,10))
  x_axis_labels = list(labelencoder.classes_) # labels for x-axis
  y_axis_labels = list(labelencoder.classes_) # labels for y-axis

  sns.heatmap(cm, xticklabels = x_axis_labels, yticklabels = y_axis_labels, annot=True,fmt='g')


# Cleaning the data 

#Transform labels
labelencoder = LabelEncoder()
data=data.dropna()
data['target'] = labelencoder.fit_transform(data['Genre'])

integer_mapping = {l: i for i, l in enumerate(labelencoder.classes_)}


y = data['target']

"""
# Drop the Activity time for now + target labels
X = data.drop('ActivityTime',1)
X_2= X.drop('Genre',1)
X_3 = X_2.drop('target',1)
"""

#Keeping activity time
X=data.drop('Genre',1)
X_3 = X.drop('target',1)

#Scale features
scaler=StandardScaler()
scaler.fit(X_3)
scaled_data=scaler.transform(X_3)
scaled_dataframe = pd.DataFrame(scaled_data,columns=X_3.columns)


# 75-25 train vs test split
x_train, x_test, y_train, y_test = train_test_split(scaled_dataframe, y, test_size=0.25, random_state=0)


##### Logistic Regression #####
logisticRegr = LogisticRegression(max_iter=300)
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)

print("LOGISTIC REGRESSION RESULTS")
print(integer_mapping)
display_results(y_test,predictions)


##### SVM #####

#Create svm Classifier
clf = svm.SVC(kernel='rbf')

#Train the model 
clf.fit(x_train, y_train)

#Predictions for test dataset
y_pred = clf.predict(x_test)

print("SVM RESULTS")
print(integer_mapping)
display_results(y_test,y_pred)



##### Random forest ##### 

clf=RandomForestClassifier(n_estimators=100)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

print("RANDOM FOREST RESULTS")
print(integer_mapping)
display_results(y_test,y_pred)


##### MLP #####

clf = MLPClassifier(random_state=1, max_iter=1000).fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("MLP RESULTS")
print(integer_mapping)
display_results(y_test,y_pred)



##### AdaBoost Classifier #####

adaboost = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
model = adaboost.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("ADABOOST RESULTS")
print(integer_mapping)
display_results(y_test,y_pred)


##### Decision Tree #####
clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("DECISION TREE RESULTS")
print(integer_mapping)
display_results(y_test,y_pred)



"""

x_train_2, x_dev, y_train_2, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

def hyperparam_search(X_dev,y_dev,regressor,param_grid): 
    grid = GridSearchCV(regressor,param_grid = param_grid,scoring='r2', verbose=3, cv=3,n_jobs=6)
    fitting = grid.fit(X_dev,y_dev)
    parameters = fitting.best_params_
    results = fitting.cv_results_

    print("parameters")
    print(parameters)
    print("results")
    print(results)
    print('-----------------------------------------')
    print()


hyperparam_search(x_dev,y_dev,LogisticRegression(solver='liblinear'),param_grid=dict(penalty=['l1', 'l2'])) #l2
hyperparam_search(x_dev,y_dev,svm.SVC(),param_grid={'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear']} )
hyperparam_search(x_dev,y_dev,AdaBoostClassifier(),param_grid={'n_estimators':[10,20,50,100]})
"""