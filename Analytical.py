import numpy as np
import scipy as sp

from train import load_datafile
import random
import argparse
import sklearn.datasets
import logging
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
import sklearn
#from torch.utils.data import Dataset, DataLolader
import utils
import os

import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def my_kernel(X, Y=None):
    #k1 = sklearn.metrics.pairwise.rbf_kernel(X, Y=Y,  gamma = 1/sigma(X))
    K = pairwise_kernels(X, Y=Y, metric='rbf', gamma = 1/ sigma(X)** 2)
    return K

def nu(X):
    vect_wrt_origin = sklearn.metrics.pairwise.euclidean_distances(X,[[0]* len(X[0]) ])
    return 1/ (np.sum(vect_wrt_origin) / len(vect_wrt_origin) )

def G(tot_view,L,U, K ,nu, lamb =0.1 ):
    G = np.matmul(np.transpose(L), L) + nu*K + 2* lamb*(tot_view - 1)* np.matmul(np.transpose(U), U)
    return G

def sigma(X):
    sig = np.sum(sklearn.metrics.pairwise.euclidean_distances(X,X)) / len(X)
    return sig

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dataset/housing.data', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2004, help="random seed for initialization")
parser.add_argument('--log_dir', default='analytical_log1', help="file containing logs")

parser.add_argument('--outputDim', type=int, default=1, help="provides outputdimension")

parser.add_argument("--test_trainsplit", default=0.10)
parser.add_argument("--lamb", type=int, default=0.1)

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


if (__name__ == "__main__"):
    
    args = parser.parse_args()
    log_model_dir = 'experiments/'
    utils.set_logger(os.path.join(log_model_dir, args.log_dir))

    X, Y = load_datafile(args.dataset, args.outputDim)
    test_ind = int(args.test_trainsplit*len(X)) #train_test split
    unlab = int(len(X)*0.10) # unlaballed point ind

    

    n = random.randint(len(X[0])//2, len(X[0])-1) #random disjoint split of dataset
    print("split of attribute at")
    print(n)

    X = StandardScaler().fit_transform(X)
    #shuffle_along_axis(X, 1)
    
    logging.info("Loading the datasets...")
    logging.info("model...")

    logging.info("Loading the datasets...")
#print(n)
#============Need better preprocessing===================
    X1 = X[:-test_ind,:n]
    X2 = X[:-test_ind,n:]
    Y_test = Y[-test_ind:,:]
    X_test = X[-test_ind:,:]
    Y = Y[:-(test_ind+unlab),:] #no. of unlabelled point 
    
    
    print(X.shape)
    
    ymin = Y.min(axis=0)
    ymax = Y.max(axis=0)
    y_max_min = (ymax)

#============Need Better way to do it====================

    K1 = my_kernel(X1) + np.identity(len(X1))
    K2 = my_kernel(X2) + np.identity(len(X1)) # to make them pos_def

    L1 = K1[:-unlab,:] 
    U1 = K1[-unlab:,:]
   
    L2 = K2[:-unlab,:]
    U2 = K2[-unlab:,:]

    G1 = G(2,L1,U1, K1 , nu(X1), lamb = args.lamb)
    G2 = G(2,L2,U2, K2 , nu(X2), lamb = args.lamb)
    print(G1.shape)
    print(K1.shape)

    c1_a = np.linalg.inv(G1 - 4* args.lamb**2 * np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(U1), U2), np.linalg.inv(G2)),np.transpose(U2)),U1))
    c2_b = np.matmul(np.transpose(L1),Y)+ 2* args.lamb* np.matmul(np.matmul(np.matmul( np.matmul( np.transpose(U1), U2), np.linalg.inv(G2)),np.transpose(L2)),Y)

    c1 = np.matmul(c1_a, c2_b )
    k_test = my_kernel(X1, Y=X_test[:,:n])
    #alpha = np.atleast_1d(1.0)

    #_solve_cholesky_kernel(K, y, alpha)
    pred_y = np.dot(np.transpose(k_test[:,:]), c1.reshape(-1))

    ssl_error = rmse(Y_test, pred_y) 
    
    Normalised_ssl_error = ssl_error/ y_max_min

    logging.info("Normalisedd RMSE (SSL): {:05.2f}".format(Normalised_ssl_error[0]))

#====================== SL implementation====================
    from sklearn.kernel_ridge import KernelRidge

    clf = KernelRidge(alpha=1, kernel = 'rbf', gamma = 1/ sigma(X)** 2)
    X_sl, Y_sl = load_datafile(args.dataset, args.outputDim)
    
    X_sl = X_sl[:-(test_ind+unlab), :]
    Y_sl = Y_sl[:-(test_ind+unlab), :]
    X_sl = StandardScaler().fit_transform(X_sl)

    X_test_sl = X[-test_ind:, :]
    Y_test_sl = Y[-test_ind:, :]

    
    clf.fit(X_sl, Y_sl)
    sl_pred = clf.predict(X_test_sl)
    sl_error = rmse(Y_test_sl, sl_pred) / y_max_min
    logging.info("Normalisedd RMSE (SL): {:05.2f}".format(sl_error[0]))
