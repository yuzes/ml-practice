#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:26:37 2018

@author: YuzeSu
"""
import math
import time
import numpy as np
from matplotlib import pyplot as plt
import gzip, pickle
with gzip.open('mnist.pkl.gz') as f:
    train_set, valid_set, test_set = pickle.load(f)
    
 

def computeCov_vec_base(X, miu):
    #vector based method
    start = time.time()
    n = float(len(X))
    d = X.shape[1]
    cov = np.zeros((d,d))
    for xi in X:
        sub = np.subtract(xi, miu)
        t = np.transpose(sub)
        cov += np.outer(t, sub)
    print 'vecotor base : ', (time.time() - start)
    return cov / n

def computeCov_mat_base(X,miu):
    n = float(X.shape[0])
    print 'n = ', n
    v = np.subtract(X, miu)
    t = np.transpose(v)
    cov = np.zeros((784, 784))
    start = time.time()
    cov = np.matmul(t, v)
    print 'matrix base : ', (time.time() - start)
    return cov / n

def reconstruct_error(X,mean):
    #eig_val,eig_vec = np.linalg.eig(np.matmul(X.T,X))
    U,s,V = np.linalg.svd(X)
    print s.shape
    errorArray = np.zeros(100)
    ks = [i for i in range(100)]
    eig_sum = 0
    for i in range(s.shape[0]):
        eig_sum += s[i]
    ksum = 0.0
    half = False
    tw = False
    for k in range(100):
        ksum += s[k]
        errorArray[k] = 1 - (ksum / eig_sum)
        if(errorArray[k] < 0.5 and not half):
            print 'error less than 0.5 from k = ', k
            half = True
        if(errorArray[k] < 0.2 and not tw):
            print 'error less than 0.2 from k = ', k
            tw = True
        
    
    plt.plot(ks, errorArray, 'r', label='Reconstruction Error')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()
    
def display_eigen_vec(X):
    print X.shape
    U,s,V = np.linalg.svd(X)
    for i in range(10):
        norm = np.linalg.norm(V[i], ord=2)
        img = np.reshape(V[i] / norm, (28,28))
        plt.figure()
        plt.imshow(img)
    
def main():
    """
    generate samples
    """
    index = np.random.choice(len(train_set[0]), 1000, replace=False)
    samples = [train_set[0][i] for i in index]
    i = 0
    
    """
    for sample in samples:
        norm = np.linalg.norm(sample, ord=2)
        temp = np.reshape(sample / norm, (28,28))
        plt.figure()
        plt.imshow(temp)
    """
    
    
    data = train_set[0]
    print data
    
    sum_vec = np.zeros(train_set[0].shape[1], dtype = float)
    for vec in data:
        sum_vec += vec
    mean_vec = sum_vec / 50000.0
    #mean_vec_2d = np.reshape(mean_vec, (28,28))
    #plt.imshow(mean_vec_2d)
    #vec_base = computeCov_vec_base(data, mean_vec)
    mat_base = computeCov_mat_base(data, mean_vec)
    #reconstruct_error(mat_base, mean_vec)
    display_eigen_vec(mat_base)
    
    """
    ensure two method for computing cov generate the same result
    diff = 0
    for i in range(vec_base.shape[0]):
        for j in range(vec_base.shape[1]):
            diff += math.fabs(vec_base[i][j] - mat_base[i][j])
    print 'difference : ', diff / (vec_base.shape[0] * vec_base.shape[0])
    """
     

if __name__ == "__main__":
    main()
