import cvxpy as cp
import numpy as np
import csv
import matplotlib.pyplot as plt

#load data and store it in matrix

train_cat = np.matrix(np.loadtxt('train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('train_grass.txt', delimiter = ','''))

#calculate the sample mean vector, sample covariance matrices and sample priors for train_cat and train_grass
def cal_factors():
    u_cat = np.mean(train_cat)
    u_train = np.mean(train_grass)
    cov_cat = np.cov(train_cat)
    cov_train = np.cov(train_grass)
    return (u_cat,u_train), (cov_cat, cov_train)


if __name__ == '__main__':
    u_train, cov_train = cal_factors()