#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

# Sigmoid函数近似单位阶跃函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# 代价函数，越小代表越有可能是这一类
def J_cost(X, y, beta):
    '''
    :param X:  sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return: the result of formula 3.27
    '''
    # 将两个矩阵连接
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    # 行转为列
    beta = beta.reshape(-1, 1)
    # 行转为列
    y = y.reshape(-1, 1)
    # formula 3.27
    Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))

    return Lbeta.sum()

# 梯度下降公式求导数
def gradient(X, y, beta):
    '''
    compute the first derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.30
    ----------------------------------
    :param X: sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return:
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))

    gra = (-X_hat * (y - p1)).sum(0)

    return gra.reshape(-1, 1)

# 求二阶导数
def hessian(X, y, beta):
    '''
    compute the second derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.31
    ----------------------------------
    :param X: sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return:
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    p1 = sigmoid(np.dot(X_hat, beta))

    m, n = X.shape
    P = np.eye(m) * p1 * (1 - p1)

    assert P.shape[0] == P.shape[1]
    return np.dot(np.dot(X_hat.T, P), X_hat)


def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    '''
    update parameters with gradient descent method
    --------------------------------------------
    :param beta:
    :param grad:
    :param learning_rate:
    :return:
    '''
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        # 打印cost
        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))

    return beta


def update_parameters_newton(X, y, beta, num_iterations, print_cost):
    '''
    update parameters with Newton method
    :param beta:
    :param grad:
    :param hess:
    :return:
    '''

    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        hess = hessian(X, y, beta)
        beta = beta - np.dot(np.linalg.inv(hess), grad)

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta

# 初始化beta
def initialize_beta(n):
    beta = np.random.randn(n + 1, 1) * 0.5 + 1
    return beta


def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
    '''
    :param X:
    :param y:~
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :param method: str 'gradDesc' or 'Newton'
    :return:
    '''
    # 获得矩阵X维数，m为行数，n为列数
    m, n = X.shape
    # 初始化beta
    beta = initialize_beta(n)

    if method == 'gradDesc':
        return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)
    elif method == 'Newton':
        return update_parameters_newton(X, y, beta, num_iterations, print_cost)
    else:
        raise ValueError('Unknown solver %s' % method)


def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))

    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0

    return p1


if __name__ == '__main__':
    data_path = 'data/3.0a.csv'
    data = pd.read_csv(data_path).values

    # 3.0a表中，第4列，好瓜为1，坏瓜为0
    is_good = data[:, 3] == 1
    is_bad = data[:, 3] == 0

    X = data[:, 1:3].astype(float)
    y = data[:, 3]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    # 绘制表中西瓜散点图到密度—含糖量图上
    plt.scatter(data[:, 1][is_good], data[:, 2][is_good], c='k', marker='o')
    plt.scatter(data[:, 1][is_bad], data[:, 2][is_bad], c='r', marker='x')

    # X轴为密度，Y轴为含糖量
    plt.xlabel('Density')
    plt.ylabel('Sugar content')

    # 可视化模型结果
    beta = logistic_model(X, y, print_cost=True, method='gradDesc', learning_rate=0.3, num_iterations=1000)
    w1, w2, intercept = beta

    # y1直线表达式
    x1 = np.linspace(0, 1)
    y1 = -(w1 * x1 + intercept) / w2

    print(w1, w2)
    print(w1)
    print(w2)

    # 绘制到图上
    ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')

    # 使用scikit learn的分类进行效果对比
    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 注意sklearn的逻辑回归中，C越大表示正则化程度越低。
    lr.fit(X, y)

    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(J_cost(X, y, lr_beta))

    # 可视化sklearn LogisticRegression 模型结果
    w1_sk, w2_sk = lr.coef_[0, :]

    x2 = np.linspace(0, 1)
    y2 = -(w1_sk * x2 + lr.intercept_) / w2

    ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')

    plt.legend(loc='upper right')
    plt.show()