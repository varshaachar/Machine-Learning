# Varsha Achar
# CS 7641 - Supervised Learning

# To suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Function to plot the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# #################################################

# WINE QUALITY DATASET

print('Beginning Wine Quality data set...')
data = pd.read_csv('winequality-data.csv')
X = data.iloc[:,:11]
y = data.iloc[:,11]

############### DECISION TREE ########################
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# Find cross validation scores
cvs_score = []
depth = []
for i in range(1, 20):
    clf = DecisionTreeClassifier(random_state=0, max_depth=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    depth.append(i)
    cvs_score.append(cvs.mean())

# Plot to find depth at best CV
fig = plt.figure()
plt.plot(depth, cvs_score)
fig.suptitle('Cross Validation Score vs Depth', fontsize=13)
plt.xlabel('Depth', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

print('Max cvs', max(cvs_score))

# This is to compare the gini and the entropy criterion and plot the values for different depths
list1=[]
list2=[]
for depth in range(1, 50):
    clf = DecisionTreeClassifier(random_state=0,criterion='gini', max_depth=depth)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    list1.append(cvs.mean())

    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=depth)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    list2.append(cvs.mean())

fig = plt.figure()
plt.plot(range(len(list2)),list2, 'g')
plt.plot(range(len(list1)),list1, 'b')
fig.suptitle('Comparing Gini and Entropy', fontsize=13)
plt.xlabel('Depth', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()


# Choose depth 18 as best depth and gini as best
clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=18)
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train Decision Tree is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of Decision Tree is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of Decision Tree is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "Decision Tree with max depth 18", X, y, ylim=[0,1])

#-------------------------------------------------------------------------------------#

# NEURAL NETWORK

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

clf = MLPClassifier(solver='adam', alpha=1e-1, hidden_layer_sizes=(100, 10, 5, 20), random_state=0, activation='tanh')
cvs = cross_val_score(clf, X_train, y_train, cv=5)
print('CVS', cvs.mean())
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time taken to train Neural Network is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of Neural Network is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of Neural Network is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "Neural Network", X, y, ylim=[0,1])

#-------------------------------------------------------------------------#

# ADA BOOSTING

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# Find cross validation scores
cvs_score = []
ne = []
for i in [5, 10, 50, 100]:
    clf = AdaBoostClassifier(n_estimators=i, learning_rate=0.01)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    ne.append(i)
    cvs_score.append(cvs.mean())

# Plot to find best hyperparameters
fig = plt.figure()
plt.plot(ne, cvs_score)
fig.suptitle('Cross Validation Score vs Num estimators', fontsize=13)
plt.xlabel('Number of estimators', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

print('Max cvs', max(cvs_score))

cvs_score_10 = []
alpha_10 = []
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    clf = AdaBoostClassifier(n_estimators=10, learning_rate=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    alpha_10.append(i)
    cvs_score_10.append(cvs.mean())

cvs_score_50 = []
alpha_50 = []
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    clf = AdaBoostClassifier(n_estimators=50, learning_rate=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    alpha_50.append(i)
    cvs_score_50.append(cvs.mean())

cvs_score_100 = []
alpha_100 = []
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    clf = AdaBoostClassifier(n_estimators=100, learning_rate=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    alpha_100.append(i)
    cvs_score_100.append(cvs.mean())


# Plot to find best hyperparameters - alpha
fig = plt.figure()
plt.plot(alpha_10, cvs_score_10, 'b')
plt.plot(alpha_50, cvs_score_50, 'r')
plt.plot(alpha_100, cvs_score_100, 'g')
fig.suptitle('Cross Validation Score vs Alpha', fontsize=13)
plt.xlabel('Alpha', fontsize=8)
plt.ylabel('Cross Validation Score', fontsize=8)
plt.show()

# Choose num_estimators = 10 and learning rate = 0.01
clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.01)
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train ADA Boosting is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of ADA Boosting is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of ADA Boosting is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "ADA Boost Classifier with n_e = 10", X, y, ylim=[0,1])

#------------------------------------------------------------------------------#
# SVM CLASSIFIER

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

cvs_score = []
gamma = []
# Find best hyperparameter
for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]:
    gamma.append(g)
    clf = svm.SVC(kernel='rbf', gamma=g)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    cvs_score.append(cvs.mean())

print('CV Score RBF', max(cvs_score))

# Plot to find best hyperparameters
fig = plt.figure()
plt.plot(gamma, cvs_score)
fig.suptitle('Cross Validation Score vs Gamma', fontsize=13)
plt.xlabel('Gamma', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

# Choose RBF kernel with gamma=2
clf = svm.SVC(kernel='rbf', gamma=0.2)
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train SVM Classifier is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of SVM Classifier is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of SVM Classifier is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "SVM Classifier with RBF kernel", X, y, ylim=[0,1])


#------------------------------------------------------------------------------------#

# K NEAREST NEIGHBORS
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# Find cross validation scores
cvs_score = []
k = []
for K in range(1, 50):
    clf = KNeighborsClassifier(K, weights="distance", algorithm="auto", metric="manhattan")
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    k.append(K)
    cvs_score.append(cvs.mean())

# Plot to find best hyperparameters
fig = plt.figure()
plt.plot(k, cvs_score)
fig.suptitle('Cross Validation Score vs K', fontsize=13)
plt.xlabel('K', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

print('Max cvs', max(cvs_score))

# Choose K = 46
clf = KNeighborsClassifier(42, weights="distance", algorithm="auto", metric="manhattan")
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train KNN Classifier is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of KNN Classifier is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of KNN Classifier is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "KNN with K=42", X, y, ylim=[0,1])

#########################################


###############################
# Titanic dataset
print('Beginning Titanic data set...')
data = pd.read_csv('titanic-data.csv')
X = data.iloc[:,2:]
y = data.iloc[:,1]

############### DECISION TREE ########################
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# Find cross validation scores
cvs_score = []
depth = []
for i in range(1, 20):
    clf = DecisionTreeClassifier(random_state=0, max_depth=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    depth.append(i)
    cvs_score.append(cvs.mean())

# Plot to find depth at best CV
fig = plt.figure()
plt.plot(depth, cvs_score)
fig.suptitle('Cross Validation Score vs Depth', fontsize=13)
plt.xlabel('Depth', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

print('Max cvs', max(cvs_score))

# This is to compare the gini and the entropy criterion and plot the values for different depths
list1=[]
list2=[]
for depth in range(1, 25):
    clf = DecisionTreeClassifier(random_state=0,criterion='gini', max_depth=depth)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    list1.append(cvs.mean())

    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=depth)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    list2.append(cvs.mean())

fig = plt.figure()
plt.plot(range(len(list2)),list2, 'g')
plt.plot(range(len(list1)),list1, 'b')
fig.suptitle('Comparing Gini and Entropy', fontsize=13)
plt.xlabel('Depth', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()


# Choose depth 5 as best depth and gini as best
clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5)
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

print("Time take to train Decision Tree is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of Decision Tree is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of Decision Tree is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "Decision Tree with max depth 5", X, y, ylim=[0,1])

cnf_matrix = confusion_matrix(y_test, test_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Survived', 'Dead'],
                      title='Confusion matrix for Decision Trees')
plt.show()

#------------------------------------------------------------#

# ADA BOOSTING

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# Find cross validation scores
cvs_score = []
ne = []
for i in [100, 150, 200, 250, 500, 1000, 1500, 2000]:
    clf = AdaBoostClassifier(n_estimators=i, learning_rate=0.01)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    ne.append(i)
    cvs_score.append(cvs.mean())

# Plot to find best hyperparameters
fig = plt.figure()
plt.plot(ne, cvs_score)
fig.suptitle('Cross Validation Score vs Num estimators', fontsize=13)
plt.xlabel('Number of estimators', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

print('Max cvs', max(cvs_score))

cvs_score_10 = []
alpha_10 = []
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    clf = AdaBoostClassifier(n_estimators=1000, learning_rate=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    alpha_10.append(i)
    cvs_score_10.append(cvs.mean())

cvs_score_50 = []
alpha_50 = []
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    clf = AdaBoostClassifier(n_estimators=1500, learning_rate=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    alpha_50.append(i)
    cvs_score_50.append(cvs.mean())

cvs_score_100 = []
alpha_100 = []
for i in [0.0001, 0.001, 0.01, 0.1, 1]:
    clf = AdaBoostClassifier(n_estimators=2000, learning_rate=i)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    alpha_100.append(i)
    cvs_score_100.append(cvs.mean())


# Plot to find best hyperparameters - alpha
fig = plt.figure()
plt.plot(alpha_10, cvs_score_10, 'b')
plt.plot(alpha_50, cvs_score_50, 'r')
plt.plot(alpha_100, cvs_score_100, 'g')
fig.suptitle('Cross Validation Score vs Alpha', fontsize=13)
plt.xlabel('Alpha', fontsize=8)
plt.ylabel('Cross Validation Score', fontsize=8)
plt.show()

# Choose num_estimators = 1500 and learning rate = 0.01
clf = AdaBoostClassifier(n_estimators=1500, learning_rate=0.01)
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train ADA Boosting is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of ADA Boosting is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of ADA Boosting is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "ADA Boost Classifier with n_e = 1500", X, y, ylim=[0,1])

cnf_matrix = confusion_matrix(y_test, test_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Survived', 'Dead'],
                      title='Confusion matrix for Boosting')
plt.show()

#----------------------------------------------------------#

# NEURAL NETWORK

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

clf = MLPClassifier(solver='adam', alpha=1e-1, hidden_layer_sizes=(25,2), random_state=0, activation='tanh')
cvs = cross_val_score(clf, X_train, y_train, cv=5)
print('CVS', cvs.mean())
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time taken to train Neural Network is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of Neural Network is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of Neural Network is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "Neural Network", X, y, ylim=[0,1])

cnf_matrix = confusion_matrix(y_test, test_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Survived', 'Dead'],
                      title='Confusion matrix for Neural Network')
plt.show()

#--------------------------------------------------------------------------#
# SVM CLASSIFIER

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

cvs_score = []
gamma = []
# Find best hyperparameter
for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]:
    gamma.append(g)
    clf = svm.SVC(kernel='rbf', gamma=g)
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    cvs_score.append(cvs.mean())

print('CV Score RBF', max(cvs_score))

# Plot to find best hyperparameters
fig = plt.figure()
plt.plot(gamma, cvs_score)
fig.suptitle('Cross Validation Score vs Gamma', fontsize=13)
plt.xlabel('Gamma', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

# Choose RBF kernel with gamma=0.1
clf = svm.SVC(kernel='rbf', gamma=0.1)
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train SVM Classifier is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of SVM Classifier is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of SVM Classifier is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "SVM Classifier with RBF kernel", X, y, ylim=[0,1])

cnf_matrix = confusion_matrix(y_test, test_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Survived', 'Dead'],
                      title='Confusion matrix for SVM')
plt.show()

#-----------------------------------------------------------------------------#

# K NEAREST NEIGHBORS
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

# Find cross validation scores
cvs_score = []
k = []
for K in range(1, 50):
    clf = KNeighborsClassifier(K, weights="uniform", algorithm="auto", metric="manhattan")
    cvs = cross_val_score(clf, X_train, y_train, cv=5)
    k.append(K)
    cvs_score.append(cvs.mean())

# Plot to find best hyperparameters
fig = plt.figure()
plt.plot(k, cvs_score)
fig.suptitle('Cross Validation Score vs K', fontsize=13)
plt.xlabel('K', fontsize=7)
plt.ylabel('Cross Validation Score', fontsize=7)
plt.show()

print('Max cvs', max(cvs_score))

# Choose K = 14
clf = KNeighborsClassifier(14, weights="distance", algorithm="auto", metric="manhattan")
start_time = datetime.datetime.now()
clf = clf.fit(X_train, y_train)
end_time = datetime.datetime.now()
delta = end_time - start_time
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)
print("Time take to train KNN Classifier is " + str(int(delta.total_seconds() * 1000)) + " ms")
print("The training accuracy of KNN Classifier is ", str(accuracy_score(y_train, train_predict)))
print("The testing accuracy of KNN Classifier is " + str(accuracy_score(y_test, test_predict)))

plot_learning_curve(clf, "KNN with K=14", X, y, ylim=[0,1])

cnf_matrix = confusion_matrix(y_test, test_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Survived', 'Dead'],
                      title='Confusion matrix for KNN')
plt.show()