import math
import random
import numpy as np


def g_delta(w, xi, yi, delta):
    xi = np.array([xi])
    yi = np.array([yi])
    prediction = np.dot(xi, w)
    if (prediction - yi) <= -delta:
        return np.square(prediction - yi + delta)
    elif abs(yi - prediction) < delta:
        return 0
    else:
        return np.square(prediction - yi - delta)


def g_delta_derivative(w, xi, yi, delta):
    xi = np.array([xi])
    yi = np.array([yi])
    prediction = np.dot(xi, w)
    if (prediction - yi) <= -delta:
        return 2 * np.dot(xi.T, prediction - yi + delta)
    elif abs(prediction - yi) < delta:
        return np.zeros(w.shape)
    else:
        return 2 * np.dot(xi.T, prediction - yi - delta)


def f_w(w, x, y, delta, lam):
    n = np.size(x, 0)
    g_delta_x = np.array([g_delta(w, xi, yi, delta) for xi, yi in zip(x, y)])
    return (1/n) * np.sum(g_delta_x) + np.sum(np.square(w))


def f_w_derivative(w, x, y, delta, lam):
    n = np.size(x, 0)
    g_delta_derivative_x = np.array([g_delta_derivative(w, xi, yi, delta) for xi, yi in zip(x, y)])
    return (1 / n) * np.sum(g_delta_derivative_x, 0) + 2 * lam * w


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = np.zeros(num_iter)
    n = np.size(data, 0)
    x = np.c_[np.ones(n), data]
    for it in range(num_iter):
        prediction = f_w_derivative(w, x, y, delta, lam)
        w = w - eta * prediction
        history_fw[it] = f_w(w, x, y, delta, lam)
    return w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    history_fw = np.zeros(num_iter)
    n = np.size(data, 0)
    x = np.c_[np.ones(n), data]
    for it in range(1, num_iter):
        ix = random.choice(np.arange(np.size(x, 0))) if i == -1 else i
        prediction = f_w_derivative(w, np.array([x[ix]]), np.array([y[ix]]), delta, lam)
        w = w - eta/math.sqrt(it) * prediction
        history_fw[it] = f_w(w, x, y, delta, lam)
    return w, history_fw
