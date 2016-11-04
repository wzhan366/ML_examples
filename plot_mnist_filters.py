"""
=====================================
Visualization of MLP weights on MNIST
=====================================
My idea:

highlight:
1 the input actually is a vector, which confused me about the input node size
 - which means it puts too much features into the system...
2 the alpha is regulation parameter
3 the main issue is finding the optimal weights, here we should use the 
	random initial states
4 cross validation should play a role in this case


"""
# print(__doc__)
import pdb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target # X -> (70000, 784), y -> (70000,)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
pdb.set_trace()
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='lbfgs', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-5, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))




fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()
