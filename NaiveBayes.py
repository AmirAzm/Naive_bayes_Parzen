import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import collections


class NB(object):
    def __init__(self, bandwidth=1):
        self.priors = []
        self.h = bandwidth

    def prior(self, x):
        numD = x.shape[0]
        cnt = collections.Counter(x)
        for c in range(len(cnt)):
            self.priors.append(cnt[c] / numD)

    def fit(self, x, y):
        self.Class = np.sort(np.unique(y))
        self.prior(y)
        self.x_bars = [x[y == yi] for yi in np.sort(np.unique(y))]
        print(self.x_bars)

    def predict(self, X):
        self.Likelihood = []
        for x in X:
            self.like = np.zeros(len(self.x_bars))
            for i in range(len(self.x_bars)):
                n = len(self.x_bars[i])
                coef = 1 / ((n) * (2 * np.pi**self.x_bars[i].shape[1] / 2) * (self.h**self.x_bars[i].shape[1]))
                for j in self.x_bars[i]:
                    diff = x - j
                    self.like[i] += np.exp(-(np.dot(diff.T, diff)) / (2 * self.h**2))
                self.like[i] *= coef
            self.Likelihood.append(self.like)
        self.Post = []
        for like in self.Likelihood:
            self.Post.append(like * self.priors)
        self.proba = []
        for post in self.Post:
            self.proba.append(post / post.sum())
        self.proba = np.array(self.proba)
        self.pred = np.array([np.argmax(self.proba, 1)]).reshape(-1)
        return self.pred, self.proba


Text = open('Log.txt', 'w')
data = load_iris()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, shuffle=True)
kf = KFold(n_splits=5)
band = [0.2, 0.4, 0.5, 0.8, 1, 2]
c = 0
for b in band:
    for train_i, test_i in kf.split(x_train):
        print('State :', c + 1, file=Text)
        print('Bandwidth :', b, file=Text)
        x_train_train, x_train_test = x_train[train_i], x_train[test_i]
        y_train_train, y_train_test = y_train[train_i], y_train[test_i]
        clf = NB(bandwidth=b)
        clf.fit(x_train, y_train)
        pred, prob = clf.predict(x_test)
        print('Predicted Vector :', pred, file=Text)
        print('Predicted Probability :', prob, file=Text)
        print(classification_report(y_test, pred), file=Text)
        print(confusion_matrix(y_test, pred), file=Text)
        print('######################################################', file=Text)
        c += 1
print('Test Results', file=Text)
clf = NB(bandwidth=0.5)
clf.fit(x_train, y_train)
pred, prob = clf.predict(x_test)
print('Predicted Vector :', pred, file=Text)
print('Predicted Probability :', prob, file=Text)
print(classification_report(y_test, pred), file=Text)
print(confusion_matrix(y_test, pred), file=Text)
Text.close()
