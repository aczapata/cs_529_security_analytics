import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import Problem5 as gd

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part
    data = np.load("data.npy")
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    std_scale = preprocessing.StandardScaler().fit(x)
    x = std_scale.transform(x)
    w = np.array([[0], [1]])

    w1, fw1_history = gd.bgd_l2(x, y, w, eta=0.05, delta=0.1, lam=0.001, num_iter=50)
    w2, fw2_history = gd.bgd_l2(x, y, w, eta=0.1, delta=0.01, lam=0.001, num_iter=50)
    w3, fw3_history = gd.bgd_l2(x, y, w, eta=0.1, delta=0, lam=0.001, num_iter=100)
    w4, fw4_history = gd.bgd_l2(x, y, w, eta=0.1, delta=0, lam=0, num_iter=100)

    fig, ax = plt.subplots(4, 2, figsize=(15,12))
    ax[0, 0].plot(fw1_history, 'red')
    ax[0, 0].set_title('GD: eta=0.05, delta=0.1, lam=0.001, num_iter=50')
    ax[0, 0].set_xlabel('Iterations')
    ax[0, 0].set_ylabel('fw')

    ax[0, 1].plot(fw2_history, 'red')
    ax[0, 1].set_title('GD: eta=0.1, delta=0.01, lam=0.001, num_iter=50')
    ax[0, 1].set_xlabel('Iterations')
    ax[0, 1].set_ylabel('fw')

    ax[1, 0].plot(fw3_history, 'red')
    ax[1, 0].set_title('GD: eta=0.1, delta=0, lam=0.001, num_iter=100')
    ax[1, 0].set_xlabel('Iterations')
    ax[1, 0].set_ylabel('fw')

    ax[1, 1].plot(fw4_history, 'red')
    ax[1, 1].set_title('GD: eta=0.1, delta=0, lam=0, num_iter=100')
    ax[1, 1].set_xlabel('Iterations')
    ax[1, 1].set_ylabel('fw')

    w5, fw5_history = gd.sgd_l2(x, y, w, eta=1, delta=0.1, lam=0.5, num_iter=800)
    w6, fw6_history = gd.sgd_l2(x, y, w, eta=1, delta=0.01, lam=0.1, num_iter=800)
    w7, fw7_history = gd.sgd_l2(x, y, w, eta=1, delta=0, lam=0, num_iter=40)
    w8, fw8_history = gd.sgd_l2(x, y, w, eta=1, delta=0, lam=0, num_iter=800)

    ax[2, 0].plot(fw5_history, 'red')
    ax[2, 0].set_title('SGD: eta=1, delta=0.1, lam=0.5, num_iter=800')
    ax[2, 0].set_xlabel('Iterations')
    ax[2, 0].set_ylabel('fw')

    ax[2, 1].plot(fw6_history, 'red')
    ax[2, 1].set_title('SGD: eta=1, delta=0.01, lam=0.1, num_iter=800')
    ax[2, 1].set_xlabel('Iterations')
    ax[2, 1].set_ylabel('fw')

    ax[3, 0].plot(fw7_history, 'red')
    ax[3, 0].set_title('SGD: eta=1, delta=0, lam=0, num_iter=40')
    ax[3, 0].set_xlabel('Iterations')
    ax[3, 0].set_ylabel('fw')

    ax[3, 1].plot(fw8_history, 'red')
    ax[3, 1].set_title('SGD: eta=1, delta=0, lam=0, num_iter=800')
    ax[3, 1].set_xlabel('Iterations')
    ax[3, 1].set_ylabel('fw')

    fig.subplots_adjust(hspace=1.5)
    plt.show()
