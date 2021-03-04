"""Train and evaluate the model"""
import os
import torch
import random
import utils
import logging
import argparse
import torch.nn as nn
from tqdm import trange
from torch.autograd import Variable
import numpy as np
import sklearn.preprocessing
from itertools import chain
import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Baseline import baseline_linear
from dataloader import PrepareData
from model import linearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import csv


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dataset/housing.data', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2018, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/")
parser.add_argument('--outputDim', type=int, default=1, help="provides outputdimension")

parser.add_argument('--learningRate',type=int,default=0.00005, help="define the learningrate")
parser.add_argument("--epochs", type=int,default=100, help="Number of Epoches")
parser.add_argument("--test_trainsplit", default=0.3)
parser.add_argument("--batch_size",type=int, default=64)

# for now we are not storing pretrained weigths
def train_epoches(model,model2, train_set1,train_set2,unknow_point, epochs, criterion ,optimizer, restore_dir=None):
    all_losses = []
    for epoch in range(epochs):

        optimizer.zero_grad()
        ## ToDO 
        # make below part parametric
        # get output from the model, given the inputs
        batch_losses = []
        for ix, ((Xb1, yb1), (Xb2, yb2)) in enumerate(zip(train_set1,train_set2)):

            _X1 = Variable(Xb1).float()
            _y1 = Variable(yb1).float()
            _X2 = Variable(Xb2).float()
            _y2 = Variable(yb2).float()
            
            outputs1 = model(_X1)
            outputs2 = model2(_X2)
            # get loss for the predicted output

            loss1 = criterion(outputs1, _y1)+criterion(outputs2, _y2)
            
            l2_reg = torch.tensor(0.)
            for param in chain(model.parameters(), model2.parameters()):
                l2_reg += torch.norm(param)
            loss = loss1+ 1 * l2_reg + 1*(criterion(model(unknow_point[:100,:len(_X1[1])]), model2(unknow_point[:100,-(len(_X2[1])):]))) #co-regularsation term
        
        #nn.utils.clip_grad_norm_(parameters=chain(model.parameters(), model2.parameters()), max_norm=params.clip_grad) 
        #adding loss for each functions 
        # get gradients w.r.t to parameters
            loss.backward()
        
        # update parameters
            optimizer.step()
            batch_losses.append(loss.item())
            all_losses.append(loss.item())

        #print(batch_losses)
        meanbatchloss = np.sqrt(np.mean(batch_losses)).round(3)
        logging.info("co-RMS-loss: {:05.2f}".format(meanbatchloss))
    logging.info("Training completed for printing loss uncomment 75 and 76 linn in train.")
        #print('epoch {}, loss {}'.format(epoch, meanbatchloss/2))
        # to keep the loss fair divide by m
        ## avoiding m factor in batch loss


def load_datafile( dataset_path, multiReg = 1):
    np.random.seed(1)
    
    try:
        D = np.loadtxt(dataset_path)
    except:
            D = np.genfromtxt(dataset_path, delimiter=",", filling_values=np.nan)
    #D = np.loadtxt(dataset_path)
            col_mean = np.nanmean(D, axis = 0) 
            inds = np.where(np.isnan(D))
            D[inds] = np.take(col_mean, inds[1])

    np.random.shuffle(D)
    X = D[:,:-(multiReg)] 
    Y = D[:,-(multiReg):]
    print(Y.shape)
    return X,Y


def test_epochs(model,model2, test_set1,test_set2, criterion ,optimizer, restore_dir=None):
    
    model.eval()
    model2.eval()
    all_losses = []

    for (Xb1, yb1),(Xb2 , yb2) in zip(test_set1,test_set2):

        _X1 = Variable(Xb1).float()
        _y1 = Variable(yb1).float()
        _X2 = Variable(Xb2).float()
        _y2 = Variable(yb2).float()

        with torch.no_grad(): # we don't need gradients in the testing phase
            predicted = model(_X1)

        with torch.no_grad(): # we don't need gradients in the testing phase
            predicted2 = model2(_X2)
    
        L1 = criterion(predicted,_y1)
        L2 =criterion(predicted2,_y2)
        total_error = (L1 + L2)/2
        #loss_avg.update(total_error.item())
        all_losses.append(total_error.item())
    
    meanbatchloss = np.sqrt(np.mean(all_losses)).round(3)
    
    #print(total_error)
    logging.info("combine-rms on test set: {:05.2f}".format(meanbatchloss))
    
    print("done")

if (__name__ == "__main__"):
    args = parser.parse_args()
    log_model_dir = 'experiments/'
    batch_size = args.batch_size

    #TODO: Load the parameters from json file
    
    # Use GPUs if available
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set the logger
    utils.set_logger(os.path.join(log_model_dir, 'train.log'))

    # Create the input data pipeline
    # Initialize the DataLoader
    data_dir = args.dataset

    X, Y = load_datafile(args.dataset, args.outputDim)
    # collecting indices for test and train sets
    train_idx, test_idx = train_test_split(list(range(X.shape[0])), test_size=args.test_trainsplit)
    
    # need to automate for random points
    n = random.randint(1,len(X[0]))
    print("split of attribute at")
    print(n)
    #print(n)
    X1 = X[:,:n]
    X2 = X[:,n:]
    unseen_point = X[:,:]
    unseen_point = StandardScaler().fit_transform(unseen_point)

    ds1 = PrepareData(X1, y=Y, scale_X=True)
    ds2 = PrepareData(X2, y=Y, scale_X=True)
    
    # Load training data and test data
    # TODO find better way of doing 
    train_set1 = DataLoader(ds1, batch_size=batch_size,
                       sampler=SubsetRandomSampler(train_idx))
    train_set2 = DataLoader(ds2, batch_size=batch_size,
                      sampler=SubsetRandomSampler(train_idx))
    test_set1 = DataLoader(ds1, batch_size=batch_size,
                      sampler=SubsetRandomSampler(test_idx))
    test_set2 = DataLoader(ds2, batch_size=batch_size,
                      sampler=SubsetRandomSampler(test_idx))
    unseen_point = torch.from_numpy(unseen_point.astype(np.float32))

    #X1_train,X2_train, Y_train, X1_test,X2_test, Y_test, unseen_point = load_housing(data_dir,False)     
    logging.info("Loading the datasets...")
 
    logging.info("model...")

    # Prepare model
    #model.to(params.device)
    inputDim1 = len(X1[1])        # takes variable 'x'
    inputDim2 = len(X2[1]) 
    outputDim = args.outputDim      # takes variable 'y'
    learningRate = args.learningRate 
    epochs = args.epochs
    
    
    ## TODO
    # need to make parametric to input other kernals
    model = linearRegression(inputDim1, outputDim)
    model2= linearRegression(inputDim2, outputDim)

    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(chain(model.parameters(), model2.parameters()), lr=learningRate)

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(args.epochs))
    train_epoches(model,model2, train_set1,train_set2, unseen_point, epochs, criterion ,optimizer, restore_dir=None)
    logging.info("Starting testing for epoch(s)")    
    test_epochs(model,model2, test_set1,test_set2, criterion ,optimizer, restore_dir=None)

    baseline_linear(X, Y,epochs,batch_size,learningRate, criterion ,outputDim, args.test_trainsplit)
