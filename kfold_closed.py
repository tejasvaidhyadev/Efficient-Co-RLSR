import numpy as np
from train import load_datafile
import random
import argparse
import sklearn.datasets
import logging
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
import sklearn
import re
#from torch.utils.data import Dataset, DataLolader
import utils
import os

import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def my_kernel(X, Y=None):

    params = {"gamma": 1/ sigma(X)** 2,
                      "degree": 3,
                      "coef0": 1}

    #k1 = sklearn.metrics.pairwise.rbf_kernel(X, Y=Y,  gamma = 1/sigma(X))
    K = pairwise_kernels(X, Y=Y, metric='rbf', filter_params=True, **params)

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



def train_n_fit(X1, X2, Y, unlab, X_test, Y_test, n, y_max_min):
    
    K1 = my_kernel(X1)+  np.identity(len(X1))
    K2 = my_kernel(X2)+  np.identity(len(X1)) # to make them pos_def
 
    L1 = K1[:-unlab,:] 
    U1 = K1[-unlab:,:]
   
    L2 = K2[:-unlab,:]
    U2 = K2[-unlab:,:]

    G1 = G(2, L1, U1, K1 , nu(X1), lamb = args.lamb)
    G2 = G(2, L2, U2, K2 , nu(X2), lamb = args.lamb)
    # print(G1.shape)
    # print(K1.shape)

    c1_a = np.linalg.inv(G1 - 4* args.lamb**2 * np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(U1), U2), np.linalg.inv(G2)),np.transpose(U2)),U1))
    c2_b = np.matmul(np.transpose(L1),Y)+ 2* args.lamb* np.matmul(np.matmul(np.matmul( np.matmul( np.transpose(U1), U2), np.linalg.inv(G2)),np.transpose(L2)),Y)

    c1 = np.matmul(c1_a, c2_b )

    k_test = my_kernel(X1, Y=X_test[:,:n])
    #alpha = np.atleast_1d(1.0)

    #_solve_cholesky_kernel(k_test, y, alpha)

    pred_y = np.matmul(np.transpose(c1), k_test)
    #pred_y = np.dot(np.transpose(k_test[:,:]), c1.reshape(-1))
    ssl_error = rmse(Y_test, pred_y) 
    
    Normalised_ssl_error = ssl_error/ y_max_min

    #logging.info("Normalisedd RMSE (SSL): {:05.2f}".format(Normalised_ssl_error[0]))
    return Normalised_ssl_error


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./tool/uci-download-process/data/regression/housing/', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2004, help="random seed for initialization")
parser.add_argument('--log_dir', default='analytical_log1', help="file containing logs")

parser.add_argument('--outputDim', type=int, default=1, help="provides outputdimension")

parser.add_argument("--test_trainsplit", default=0.10)
parser.add_argument("--lamb", type=int, default=0.1)
parser.add_argument("--stan_scale", type=bool, default=True)





if (__name__ == "__main__"):
    args = parser.parse_args()
    log_model_dir = 'experiments/'
    utils.set_logger(os.path.join(log_model_dir, args.log_dir))


    # performing inverse cross validation step
    k_fold_testset = [f for f in os.listdir(args.dataset) if re.match('train', f) ] # folds for lab and unlab data
    k_fold_testset.sort()
    k_fold_trainset = [f for f in os.listdir(args.dataset) if re.match('test', f)] # folds for train 
    k_fold_trainset.sort()
    

    foldnorm_ssl_error = []
    foldnorm_sl_error =[]

    for index, train in enumerate(k_fold_trainset): # index for test and label for given fold
        X_train, Y_train = load_datafile(args.dataset+train, args.outputDim)
        X_test_unlab, Y_test_unlab = load_datafile(args.dataset+k_fold_testset[index], args.outputDim)
        unlab = int(len(X_test_unlab)*0.60) # unlaballed datapoint ind

        # Randomly splitting set into 2 disjoint 
        n = random.randint(len(X_train[0])//2, len(X_train[0])-1) # For experimentation it's range is changed
        

        print("split of attribute at")
        print(n)
        if args.stan_scale == True:
            X_train = StandardScaler().fit_transform(X_train)
            X_test_unlab = StandardScaler().fit_transform(X_test_unlab)

        # default view is 2 
        X1 = X_train[ :, :n]
        X2 = X_train[:, n:]
        Y = Y_train[:,:]#
        
        X_test = X_test_unlab[unlab:,:]#
        Y_test = Y_test_unlab[unlab:,:]#
        
        X_unlab = X_test_unlab[:unlab,:] 

        X1_unlab = X_unlab[ :, :n] 
        X2_unlab = X_unlab[ :, n:]

        X1_cat_unlab = np.concatenate((X1, X1_unlab), axis=0)#
        X2_cat_unlab = np.concatenate((X2, X2_unlab), axis=0)#

        if len(X1_cat_unlab) != len(X1) + len(X1_unlab) and len(X2_cat_unlab) != len(X2) + len(X2_unlab):
            Exception("ophs, problem with cat (between unlab and X1)")
        y_max_min = Y.max(axis=0)

        Normalised_ssl_error = train_n_fit(X1_cat_unlab, X2_cat_unlab, Y, unlab, X_test, Y_test, n, y_max_min)
        foldnorm_ssl_error.append(Normalised_ssl_error[0])
        logging.info("Normalisedd RMSE (SSL): {:05.2f}".format(Normalised_ssl_error[0]))

#====================== SL implementation====================
        from sklearn.kernel_ridge import KernelRidge

        clf = KernelRidge(alpha=1, kernel = 'rbf', gamma = 1/ sigma(X_train)** 2)  

        clf.fit(X_train, Y_train)
        sl_pred = clf.predict(X_test)
        Normalised_sl_error = rmse(Y_test, sl_pred) / y_max_min
        foldnorm_sl_error.append(Normalised_sl_error)
        logging.info("Normalisedd RMSE (SL): {:05.2f}".format(Normalised_sl_error[0]))
    

    logging.info("Avg Normalisedd RMSE (SSL): {:05.2f}".format(np.mean(foldnorm_ssl_error)))
    logging.info("Avg Normalisedd RMSE (SL): {:05.2f}".format(np.mean(foldnorm_sl_error)))

    
    
