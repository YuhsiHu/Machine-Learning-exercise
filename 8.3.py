# -*- coding: utf-8 -*-

# 利用AdaBoost和Bagging对西瓜数据集3.0a实现不剪枝决策树的集成学习
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

D = np.array([
    [0.697, 0.460, 1],
    [0.774, 0.376, 1],
    [0.634, 0.264, 1],
    [0.608, 0.318, 1],
    [0.556, 0.215, 1],
    [0.403, 0.237, 1],
    [0.481, 0.149, 1],
    [0.437, 0.211, 1],
    [0.666, 0.091, 0],
    [0.243, 0.267, 0],
    [0.245, 0.057, 0],
    [0.343, 0.099, 0],
    [0.639, 0.161, 0],
    [0.657, 0.198, 0],
    [0.360, 0.370, 0],
    [0.593, 0.042, 0],
    [0.719, 0.103, 0]])
train_d, label_d = D[:, [-3, -2]], D[:, -1]
# max_depth限定决策树是否为决策树桩，n_estimator表示不同数量的基学习器集成，下面以Bagging为例，AdaBoost同理
clf1 = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=3)
clf2 = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=5)
clf3 = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=11)
for clf in [clf1, clf2, clf3]:
    clf.fit(train_d, label_d)

x_min, x_max = train_d[:, 0].min() - 1, train_d[:, 0].max() + 1
y_min, y_max = train_d[:, 1].min() - 1, train_d[:, 1].max() + 1
xset, yset = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
clf_set, label_set = [clf1, clf2, clf3], []
for clf in clf_set:
    out_label = clf.predict(np.c_[xset.ravel(), yset.ravel()])
    out_label = out_label.reshape(xset.shape)
    label_set.append(out_label)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
(ax0, ax1, ax2) = axes.flatten()
for k, ax in enumerate((ax0, ax1, ax2)):
    ax.contourf(xset, yset, label_set[k], cmap=plt.cm.Set3)
    for i, n, c in zip([0, 1], ['bad', 'good'], ['black', 'red']):
        idx = np.where(label_d == i)
        ax.scatter(train_d[idx, 0], train_d[idx, 1], c=c, label=n)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper left')
    ax.set_ylabel('sugar')
    ax.set_xlabel('densty')
    ax.set_title('decision boundary for %s' % (k + 1))
plt.show()
