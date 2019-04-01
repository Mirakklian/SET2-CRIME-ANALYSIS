import pandas as pd    # panda's nickname is pd

import numpy as np
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# from  sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import validation_curve
# from sklearn.preprocessing import binarize
# import matplotlib.pyplot as plot
# from sklearn import linear_model
from IPython.display import Image
# from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
# import sklearn.preprocessing as prep
import pydotplus
# import math
# from sklearn.linear_model import RidgeCV
# import sklearn.linear_model as linear_model
# from sklearn.model_selection import GridSearchCV
# from sklearn.cluster import KMeans
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

communities_crime_df = pd.read_csv('D:\\VIT-2ND SEM\\PROJECT\\SET\\communities-crime-clean.csv')

# Sanity test we have good data
print(communities_crime_df.head())


def setHighCrime(df):
    '''Function to set value of highCrime depending on ViolentCrimesPerPop'''
    if df['ViolentCrimesPerPop'] > 0.1:
        return True
    else:
        return False


# Adding a new field "highCrime"
communities_crime_df['highCrime'] = communities_crime_df.apply(setHighCrime, axis=1)

# Calculating the percentage of positive and negative instances in the dataset
percentage_intances = communities_crime_df.groupby('highCrime').size() * 100 / len(communities_crime_df)
print(percentage_intances)
print("------------------")
print("Percentage Positive Instance = {}\nPercentage Negative Instance = {} ".format(percentage_intances[1],percentage_intances[0]))

#Dropping non-predictive fields as well as ViolentCrimesPerPop field
X = communities_crime_df.drop('ViolentCrimesPerPop', axis=1).drop('state', axis=1).drop('communityname', axis=1).drop('fold', axis=1).drop('highCrime', axis=1)
features = list(X.columns)
y = communities_crime_df["highCrime"]


# First, we tried by not defining the max depth
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X,y)
predicted = dt_clf.predict(X[features])
recall_score = metrics.recall_score(communities_crime_df['highCrime'], predicted)
precision_score = metrics.precision_score(communities_crime_df['highCrime'], predicted)
accuracy_score = metrics.accuracy_score(communities_crime_df['highCrime'], predicted)
print("Training Accuracy = {} Precision = {} Recall = {}".format(accuracy_score,precision_score,recall_score))


for depth in range(1,4):
    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    if tree_clf.fit(X,y).tree_.max_depth < depth:
        break
    score = np.mean(cross_val_score(tree_clf, X, y,scoring='accuracy', cv=10, n_jobs=1))
    print("Depth: %i Accuracy: %.3f" % (depth,score))

dt_clf = DecisionTreeClassifier(max_depth=3)
dt_clf.fit(X,y)
#Predicting
pred_dt= dt_clf.predict(X)
dt_accuracy= metrics.accuracy_score(communities_crime_df['highCrime'], pred_dt)
dt_precision= metrics.precision_score(communities_crime_df['highCrime'], pred_dt)
dt_recall= metrics.recall_score(communities_crime_df['highCrime'], pred_dt)
print("Accuracy for DT =",dt_accuracy)
print("Precision for DT =",dt_precision)
print("Recall for DT =",dt_precision)

#
# dot_data = tree.export_graphviz(dt_clf, out_file=None,feature_names=list(X))
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())

# Applying 10 fold cross validation
dt_cv_accuracy = cross_val_score(dt_clf, X, y, cv=10).mean()
dt_cv_precision= cross_val_score(dt_clf, X, y, cv=10, scoring='precision').mean()
dt_cv_recall = cross_val_score(dt_clf, X, y, cv=10, scoring='recall').mean()
print("Cross Validation Accuracy DT:", dt_cv_accuracy)
print("Cross Validation Recall DT:", dt_cv_precision)
print("Cross Validation Precision DT:", dt_cv_recall)

#=======================================================================================NB
# Using GaussianNB
gaussian_clf = GaussianNB()
gaussian_clf.fit(X, y)

# Applying 10 fold cross validation
gaussian_accuracy = cross_val_score(gaussian_clf, X, y, cv=10).mean()
gaussian_precision= cross_val_score(gaussian_clf, X, y, cv=10, scoring='precision').mean()
gaussian_recall = cross_val_score(gaussian_clf, X, y, cv=10, scoring='recall').mean()
print("Accuracy for gaussian Naive Bayes :", gaussian_accuracy)
print("Recall for gaussian:", gaussian_recall)
print("Precision for gaussian:", gaussian_precision)

#========================================compare
labels = ["DT" ,"GaussianNB"]
acc_list = [dt_cv_accuracy,gaussian_accuracy]
pre_list = [dt_cv_precision,gaussian_precision]
re_list = [dt_cv_recall,gaussian_recall]

x_axis_range = range(2)
plt.xticks(x_axis_range, labels, rotation='vertical')
# plt.legend()

plt.plot(x_axis_range,acc_list,'ro',color="Red",label="Accuracy")
plt.plot(x_axis_range,pre_list,'>',color="green",label="Precision")
plt.plot(x_axis_range,re_list,'<',color="green",label="Recall")

plt.xlabel('Model')
plt.ylabel('Metrics')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
legend = plt.legend()
plt.show()

#========================================================RF
clf = RandomForestClassifier(random_state=100, max_depth=3)

rf_accuracy = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
rf_precision = cross_val_score(clf, X, y, cv=10, scoring='precision').mean()
rf_recall = cross_val_score(clf, X, y, cv=10, scoring='recall').mean()

print ('Accuracy for RandomForestClassifier is:- ', rf_accuracy)
print ('Precision for RandomForestClassifier is', rf_precision)
print ('Recall for RandomForestClassifier is', rf_recall)

# Applying polynomial Kernel SVC============================================================
poly_clf = svm.SVC(kernel='poly', degree=2, C= 50)

# For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly.
# non clean dataset
poly_accuracy_d = cross_val_score(poly_clf, X, y, cv=10, scoring='accuracy').mean()
poly_precision_d = cross_val_score(poly_clf, X, y, cv=10, scoring='precision').mean()
poly_recall_d = cross_val_score(poly_clf, X, y, cv=10, scoring='recall').mean()

print ('Accuracy for SVM is:', poly_accuracy_d)
print ('Precision for SVM is: ', poly_precision_d)
print ('Recall for SVM is', poly_recall_d)

#compare b/w NB and Random Forest

labels = ["RF" ,"GaussianNB"]
acc_list = [rf_accuracy,gaussian_accuracy]
pre_list = [rf_precision,gaussian_precision]
re_list = [rf_recall, gaussian_recall]

x_axis_range = range(2)
plt.xticks(x_axis_range, labels, rotation='vertical')
# plt.legend()

plt.plot(x_axis_range,acc_list,'ro',color="Red",label="Accuracy")
plt.plot(x_axis_range,pre_list,'>',color="green",label="Precision")
plt.plot(x_axis_range,re_list,'<',color="Blue",label="Recall")

plt.xlabel('Model')
plt.ylabel('Metrics')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
legend = plt.legend()
plt.show()