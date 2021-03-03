
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

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import PrepareData
from model import linearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def baseline_linear( X, Y,num_epochs,batch_size,learningRate, criterion ,output, test_size ):
    
    trainbase, testbase = train_test_split(list(range(X.shape[0])), test_size=test_size)
    ds = PrepareData(X, y=Y, scale_X=True)
    input_size =len(X[1])
    train_set = DataLoader(ds, batch_size=batch_size,
                       sampler=SubsetRandomSampler(trainbase))
    
    test_set = DataLoader(ds, batch_size=batch_size,
    sampler=SubsetRandomSampler(testbase))

    model = linearRegression(input_size, output)

    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    all_losses = []
    logging.info("Training of Baseline starting")    

    for e in range(num_epochs):
        batch_losses = []

        for ix, (Xb, yb) in enumerate(train_set):

            _X = Variable(Xb).float()
            _y = Variable(yb).float()

        #==========Forward pass===============

            preds = model(_X)
            lossbas = criterion(preds, _y)
            l2_reg = torch.tensor(0.)
            for param in chain(model.parameters()):
                l2_reg += torch.norm(param)
            lossbase = lossbas+1 * l2_reg 

        #==========backward pass==============

            optimizer.zero_grad()
            lossbase.backward()
            optimizer.step()

        batch_losses.append(lossbase.item())
        meanbatchloss = np.sqrt(np.mean(batch_losses)).round(3)
        


#        if e % 10 == 0:
#           print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, meanbatchloss))
    logging.info("Training of Baseline successful. To print losses uncomment the line 67 and 68")  

# prepares model for inference when trained with a dropout layer
    logging.info("Baseline testing Starting")    
    model.eval()
    #loss_avg = utils.RunningAverage()
    test_batch_loss = []

    for _X, _y in test_set:

        _X = Variable(_X).float()
        _y = Variable(_y).float()

    #apply modeltest_loss
        test_preds = model(_X)
        test_loss = criterion(test_preds, _y)
        
        test_batch_loss.append(test_loss.item())
        #print("Batch loss on test: {}".format(test_loss.item()))
        #loss_avg.update(test_loss.item())
    meanbatchloss_test = np.sqrt(np.mean(batch_losses)).round(3)
    logging.info("baseline loss on test: {}".format(meanbatchloss_test)) 

