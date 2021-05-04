import pandas as pd 
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)






data = pd.read_csv('./7_AMCA_Cleaned.csv')
#data = pd.read_csv('./Culex_features.csv')




labelencoder = LabelEncoder()
data=data.dropna()
data['target'] = labelencoder.fit_transform(data['Genre'])


y = data['target']
#print(y.shape)

X = data.drop('ActivityTime',1)
X_2= X.drop('Genre',1)
X_3 = X_2.drop('target',1)


scaler=StandardScaler()
scaler.fit(X_3)
scaled_data=scaler.transform(X_3)
scaled_dataframe = pd.DataFrame(scaled_data,columns=X_3.columns)




x_train, x_test, y_train, y_test = train_test_split(scaled_dataframe, y, test_size=0.25, random_state=0)





# Logistic regression 

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
#print(predictions)
score = logisticRegr.score(x_test, y_test)
print(score)



#SVM 
from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

from sklearn import metrics

print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))







#Random Forest


from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

print("Accuracy:",metrics.balanced_accuracy_score(y_test,y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)






# MLP 
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))






#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)

model = abc.fit(x_train, y_train)


y_pred = model.predict(x_test)
print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))





#DecisionTree
from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))





"""

x_train_2, x_dev, y_train_2, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=0)



from sklearn.model_selection import GridSearchCV
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
"""