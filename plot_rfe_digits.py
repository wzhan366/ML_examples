"""
=============================
Recursive feature elimination
=============================

A recursive feature elimination example showing the relevance of pixels in
a digit classification task.
http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py

My idea:

this one basically gives us a way to select the processed feature and get the most
useful one, which have the same function as PCA in previous case.



"""
# print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import pdb
# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=10)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)
pdb.set_trace()
# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
