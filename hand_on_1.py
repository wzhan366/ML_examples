import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# '''data overview'''
# print dataset.head()
# # shape
print dataset.shape
# # descriptions
# print dataset.describe()
#
# # class distribution
# print dataset.groupby('class').size()
#
# '''data visualization'''
# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()
#
# # histograms
# dataset.hist()
# plt.show()
#
# # scatter plot matrix
# '''this mainly looks up the interactions between the variables'''
# scatter_matrix(dataset)
# plt.show()

'''model selection'''
# dataset split
# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, \
Y_train, Y_validation = cross_validation.train_test_split(X, Y,
                                                          test_size=validation_size,
                                                          random_state=seed)
print X_train.shape

# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(C=10, gamma=0.1)))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print msg

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make predictions on validation dataset
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
# print accuracy_score(Y_validation, predictions)
# print confusion_matrix(Y_validation, predictions)
# print classification_report(Y_validation, predictions)
#
#
# svc = SVC(C=10, gamma=0.1)
# svc.fit(X_train, Y_train)
# predictions = svc.predict(X_validation)
# print accuracy_score(Y_validation, predictions)
# print confusion_matrix(Y_validation, predictions)
# print classification_report(Y_validation, predictions)
