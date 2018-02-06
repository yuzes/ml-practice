#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 01:15:19 2018

@author: YuzeSu
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
import random
import math


 
GLOBAL_EPOCS = 2000
DEVELOPMENT_EPOCS = 400
           
                
"""
training algorithm
"""

def voted_percept(testNum, ys, xs, weights, EPOCS):
    weight_vec = np.zeros((EPOCS, xs.shape[1]))
    u = np.zeros(len(weights), dtype = float)
    c = 1
    r = list(range(ys.shape[0]))
    for i in range(EPOCS):
        random.shuffle(r)
        for j in r:
            sign = 1 if np.dot(weights, xs[j]) > 0 else -1
            if not sign == ys[j]:
                weights = weights + ys[j] * xs[j]
                u = u + ys[j] * xs[j] * c
            c += 1
        weight_vec[i] = weights - 1.0 / c * u
        
    return weight_vec
    

def percept(testNum, ys, xs, weights, EPOCS):
    """
    learn
    """
    weight_vec = np.zeros((EPOCS, xs.shape[1]))
    r = list(range(ys.shape[0]))
    for i in range(EPOCS):
        random.shuffle(r)
        for j in r:
            sign = 1 if np.dot(weights, xs[j]) > 0 else -1
            if not sign == ys[j]:
                weights = weights + ys[j] * xs[j]
        weight_vec[i] = weights
    return weight_vec


"""
test on training and test data
"""
def test(testNum, ys, xs, test_ys, test_xs, weight_vec, weight_vec_v, EPOCS):
    """
    test
    """
    """
    construct test data
    """
    print >> sys.stderr, "\ttest data : ", testNum
    errorRate = np.zeros(EPOCS, dtype = float)
    l2_norm = np.linalg.norm(weight_vec[len(weight_vec) - 1], ord=2)
    print >> sys.stderr, "L2-norm : " + str(l2_norm)
    
    
    testError = np.zeros(EPOCS, dtype = float)
    count = 0
    
    #noise feature
    for w in weight_vec[len(weight_vec) - 1]:
        if math.fabs(w * 200.0) < l2_norm:
            print >> sys.stderr, "\tnoise feature : ", count, " with feature value ", w
        count += 1
    margin = sys.maxint   
    for e_x in xs:
        temp_margin = math.fabs(np.dot(e_x, weight_vec[len(weight_vec) - 1]))
        margin = temp_margin if margin > temp_margin else margin
    print >> sys.stderr, "\tgeometric margin : ", margin / np.linalg.norm(weight_vec[len(weight_vec) - 1], ord=2)
    
    for i in range(EPOCS):
        numError = 0.0;
        for k in range(ys.shape[0]):
            sign = 1 if np.dot(weight_vec[i], xs[k]) > 0 else -1
            if not sign == ys[k]:
                numError += 1.0
        errorRate[i] = numError / ys.shape[0]
        
        numTestError = 0.0
        for k in range(test_ys.shape[0]):
            sign = 1 if np.dot(weight_vec[i], test_xs[k]) > 0 else -1
            if not sign == test_ys[k]:
                numTestError += 1.0
        testError[i] = numTestError / test_ys.shape[0]
        #print "num error in EPOC ", i, " : training Error : ", int(numError), " test error : ", int(numTestError)
    
    
    voted_error_rate = np.zeros(EPOCS, dtype = float)
    voted_test_error = np.zeros(EPOCS, dtype = float)
    
    for i in range(EPOCS):
        numError = 0.0;
        for k in range(ys.shape[0]):
            sign = 1 if np.dot(weight_vec_v[i], xs[k]) > 0 else -1
            if not sign == ys[k]:
                numError += 1.0
        voted_error_rate[i] = numError / ys.shape[0]
        
        numTestError = 0.0
        for k in range(test_ys.shape[0]):
            sign = 1 if np.dot(weight_vec_v[i], test_xs[k]) > 0 else -1
            if not sign == test_ys[k]:
                numTestError += 1.0
        voted_test_error[i] = numTestError / test_ys.shape[0]
    
    itr = list(range(EPOCS))
    
    
    """
    plot
    """
    trainFile = "A2." + str(testNum) + ".train.tsv"
    testFile = "A2." + str(testNum) + ".test.tsv"
    #g_error = testError - errorRate
    testError = smooth(testError)
    errorRate = smooth(errorRate)
    voted_error_rate = smooth(voted_error_rate)
    voted_test_error = smooth(voted_test_error)
    
    
    
    plt.plot(itr, testError, 'r', alpha = .7, linewidth=1, label='TestError')
    plt.plot(itr, errorRate, 'b', alpha = .7, linewidth=1, label='TrainingError')
    plt.plot(itr, voted_error_rate, 'g', alpha = .7, linewidth=1, label='Voted TrainingError')
    plt.plot(itr, voted_test_error, 'y', alpha = .7, linewidth=1, label='Voted TestError')
    #plt.plot(itr, g_error, 'g', alpha = .7, linewidth=1, label='Generalization Error')
    
    
    plt.xlabel('epocs')
    plt.ylabel('error')
    plt.ylim(ymin = 0.0)
    plt.title(trainFile + ' & ' + testFile)
    plt.legend()
    plt.savefig(trainFile+' & '+testFile+'.png' , dpi = 800)
    plt.show()
    
def smooth(vec):
    l = len(vec)
    for i in range(l):
        s = 0.0
        count = 0
        for j in range(i-10, i+10):
            if j > 0 and j < l:
                s += vec[j]
                count += 1
        vec[i] = s / count
    return vec
        
"""
test error on training, test and development data
"""  
def test_development(testNum, ys, xs, test_ys, test_xs, d_xs, d_ys, weight_vec, EPOCS):
    """
    test
    """
    """
    construct test data
    """
    print >> sys.stderr, "\ttest data : ", testNum
    errorRate = np.zeros(EPOCS, dtype = float)
    l2_norm = np.linalg.norm(weight_vec[len(weight_vec) - 1], ord=2)
    print >> sys.stderr, "L2-norm : " + str(l2_norm)
    print d_xs.shape
    developError = np.zeros(EPOCS, dtype = float)
    testError = np.zeros(EPOCS, dtype = float)
    count = 0
    
    #noise feature
    for w in weight_vec[len(weight_vec) - 1]:
        if math.fabs(w * 200.0) < l2_norm:
            print >> sys.stderr, "\tnoise feature : ", count, " with feature value ", w
        count += 1
    margin = sys.maxint   
    for e_x in xs:
        temp_margin = math.fabs(np.dot(e_x, weight_vec[len(weight_vec) - 1]))
        margin = temp_margin if margin > temp_margin else margin
    print >> sys.stderr, "\tgeometric margin : ", margin
    
    for i in range(EPOCS):
        numError = 0.0;
        for k in range(ys.shape[0]):
            sign = 1 if np.dot(weight_vec[i], xs[k]) > 0 else -1
            if not sign == ys[k]:
                numError += 1.0
        errorRate[i] = numError / ys.shape[0]
        
        numTestError = 0.0
        for k in range(test_ys.shape[0]):
            sign = 1 if np.dot(weight_vec[i], test_xs[k]) > 0 else -1
            if not sign == test_ys[k]:
                numTestError += 1.0
        testError[i] = numTestError / test_ys.shape[0]
        
        numDevelopError = 0.0
        for k in range(d_ys.shape[0]):
            sign = 1 if np.dot(weight_vec[i], d_xs[k]) > 0 else -1
            if not sign == d_ys[k]:
                numDevelopError += 1.0
        developError[i] = numDevelopError / d_ys.shape[0]
        #print "num error in EPOC ", i, " : training Error : ", int(numError), " test error : ", int(numTestError)
    itr = list(range(EPOCS))
    
    
    """
    plot
    """
    trainFile = "A2." + str(testNum) + ".train.tsv"
    testFile = "A2." + str(testNum) + ".test.tsv"
    plt.plot(itr, testError, 'r', alpha = .7, linewidth=1, label='TestError')
    plt.plot(itr, errorRate, 'b', alpha = .7, linewidth=1, label='TrainingError')
    print developError
    plt.plot(itr, developError, 'g', alpha = .7, linewidth=1, label='Development Error')
    
    
    plt.xlabel('epocs')
    plt.ylabel('error')
    plt.ylim(ymin = 0.0)
    plt.title(trainFile + ' & ' + testFile)
    plt.legend()
    plt.savefig(trainFile+' & '+testFile+'.png' , dpi = 800)
    plt.show()
    
 
def main():
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    weight_vec = []
    weight_vec_v = []
    for i in range(start, end + 1):
        #fetch training file
        trainFile = "A2." + str(i) + ".train.tsv"
        ys = np.loadtxt(trainFile, usecols = 0)
        fileP = np.loadtxt(trainFile)
        numCol = fileP.shape[1]
        xs = np.loadtxt(trainFile, usecols = range(1, numCol))
        
        
        weights = np.zeros(xs.shape[1], dtype = float)
        
        """
        development
        """
        #d_ys = ys[0:(len(ys)/5)]
        print ys.shape
        #d_xs = xs[0:(len(xs)/5), :]
        print xs.shape
        """
        """
        
        
        #weight_vec = percept(i, d_ys, d_xs, weights, DEVELOPMENT_EPOCS)
        
        weight_vec_v = voted_percept(i, ys, xs, weights, GLOBAL_EPOCS)
        weight_vec = percept(i, ys, xs, weights, GLOBAL_EPOCS)
        #fetch test file
        testFile = "A2." + str(i) + ".test.tsv"
        test_ys = np.loadtxt(testFile, usecols = 0)
        test_xs = np.loadtxt(testFile, usecols = range(1, numCol))
        test(i, ys, xs, test_ys, test_xs, weight_vec, weight_vec_v, GLOBAL_EPOCS)
        #test_development(i, ys, xs, test_ys, test_xs, d_xs, d_ys, weight_vec, DEVELOPMENT_EPOCS)


"""
optimize training algorithm for data 6, 7, 8
"""      
def main678():
    start = 6
    end = 8
    weight_vec = []
    weight_vec_v = []
    trainFile = "A2." + str(6) + ".train.tsv"
    ys = np.loadtxt(trainFile, usecols = 0)
    fileP = np.loadtxt(trainFile)
    numCol = fileP.shape[1]
    xs = np.loadtxt(trainFile, usecols = range(1, numCol))
    weights = np.zeros(xs.shape[1], dtype = float)
    for i in range(start, end + 1):
        #fetch training file
        trainFile = "A2." + str(i) + ".train.tsv"
        ys = np.loadtxt(trainFile, usecols = 0)
        fileP = np.loadtxt(trainFile)
        numCol = fileP.shape[1]
        xs = np.loadtxt(trainFile, usecols = range(1, numCol))
        
        
        #weights = np.zeros(xs.shape[1], dtype = float)
        
        """
        development
        """
        develop_ys = ys[0:(len(ys)/1)]
        print ys.shape
        develop_xs = xs[0:(len(xs)/1), :]
        print xs.shape
        """
        """
        
        weight_vec = percept(i, develop_ys, develop_xs, weights, GLOBAL_EPOCS)
    for i in range(start, end + 1):    
        
        #weight_vec = percept(i, ys, xs, weights, DEVELOPMENT_EPOCS)
        #fetch test file
        testFile = "A2." + str(i) + ".test.tsv"
        test_ys = np.loadtxt(testFile, usecols = 0)
        test_xs = np.loadtxt(testFile, usecols = range(1, numCol))
        test(i, ys, xs, test_ys, test_xs, weight_vec, GLOBAL_EPOCS)
        
if __name__ == "__main__":
    main()
    #main678()
    
    
    
    