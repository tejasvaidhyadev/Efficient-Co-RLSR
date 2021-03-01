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

from model import linearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='dataset/housing.data', help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments/")
parser.add_argument('--outputDim', type=int, default=1, help="provides outputdimension")

parser.add_argument('--learningRate',type=int,default=0.01, help="define the learningrate")
parser.add_argument("--epochs", type=int,default=100, help="Number of Epoches")

# for now we are not storing pretrained weigths

def train(model,model2, inputs_instance1,inputs_instance2,labels_instance1,labels_instance2,epochs, criterion ,optimizer, restore_dir=None):
    
    for epoch in range(epochs):

        optimizer.zero_grad()
## ToDO 
# make below part parametric
    # get output from the model, given the inputs
        #model.train()
        #odel2.train()
        outputs1 = model(inputs_instance1)
        outputs2 = model2(inputs_instance2)
        
    # get loss for the predicted output

        loss = criterion(outputs1, labels_instance1)+criterion(outputs2, labels_instance2)
        l2_reg = torch.tensor(0.)
        for param in chain(model.parameters(), model2.parameters()):
            l2_reg += torch.norm(param)
        loss += 0.1 * l2_reg + 0.5*(criterion(model(unknow_point[:400,:6]),model2(unknow_point[:400,7:]))) #co-regularsation term
        logging.info("co-RMS-loss: {:05.2f}".format(loss))
        #nn.utils.clip_grad_norm_(parameters=chain(model.parameters(), model2.parameters()), max_norm=params.clip_grad) 
    #adding loss for each functions 
    # get gradients w.r.t to parameters
        loss.backward()
    # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

def load_housing( dataset_path, multiReg = False):
## Reimplementation to get random disjoint sets
    np.random.seed(1)
    D = np.loadtxt(dataset_path)
    np.random.shuffle(D)
    if multiReg:
        X = D[:,:-4] #todo parametric
        Y = D[:,-4:]
    else:
        X = D[:,:-1] #todo parametric
        Y = D[:,-1]

    X = sklearn.preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True) #scaling of X
    #REDO
    X1_train = X[:400,:6]
    Y_train = Y[:400]
    X2_train = X[:400,7:13]
    unseen_point = X[400:,:13]
    X1_test = X[480:,:6]
    X2_test = X[480:,6:13]
    Y_test = Y[480:]
    return X1_train,X2_train, Y_train, X1_test,X2_test, Y_test, unseen_point

def test(model,model2, test_instance1,test_instance2,Y_test, criterion ,optimizer, restore_dir=None):
    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(test_instance1).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(test_instance1.astype(np.float32)))).data.numpy()
    #print(predicted)
    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted2 = model2(Variable(torch.from_numpy(test_instance2).cuda())).cpu().data.numpy()
        else:
            predicted2 = model2(Variable(torch.from_numpy(test_instance2.astype(np.float32)))).data.numpy()
    L1 = criterion(Variable(torch.from_numpy(predicted2)),Variable(torch.from_numpy(Y_test.astype(np.float32))))
    L2 =criterion(Variable(torch.from_numpy(predicted)),Variable(torch.from_numpy(Y_test.astype(np.float32))))
    total_error = (L1 + L2)/2
    print(total_error)
    logging.info("combine-rms: {:05.2f}".format(total_error))
    print("done")

if (__name__ == "__main__"):
    args = parser.parse_args()
    log_model_dir = 'experiments/'

    #TODO: Load the parameters from json file
    
    # Use GPUs if available
    #params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set the logger
    utils.set_logger(os.path.join(log_model_dir, 'train.log'))
    #logging.info("device: {}".format(params.device))

    # Create the input data pipeline
    
    # Initialize the DataLoader
    data_dir = args.dataset

    #TODO: generalised dataloader
    X1_train,X2_train, Y_train, X1_test,X2_test, Y_test, unseen_point = load_housing(data_dir,False) 
    logging.info("Loading the datasets...")

    # Load training data and test data
    ## after writing general dataloader

    
    logging.info("model...")

    # Prepare model
    #model.to(params.device)
    inputDim1 = len(X1_train[1])        # takes variable 'x'
    inputDim2 = len(X2_train[1]) 
    outputDim = args.outputDim      # takes variable 'y'
    learningRate = args.learningRate 
    epochs = args.epochs
    ## TODO
    # need to make parametric to input other kernals
    model = linearRegression(inputDim1, outputDim)
    model2= linearRegression(inputDim2, outputDim)

    if torch.cuda.is_available():
        model.cuda()
    from itertools import chain

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(chain(model.parameters(), model2.parameters()), lr=learningRate)

    if torch.cuda.is_available():
        inputs_instance1 = Variable(torch.from_numpy(X1_train.astype(np.float32)).cuda())
        labels_instance1 = Variable(torch.from_numpy(Y_train.astype(np.float32)).cuda())
        inputs_instance2 = Variable(torch.from_numpy(X2_train.astype(np.float32)).cuda())
        labels_instance2 = Variable(torch.from_numpy(Y_train.astype(np.float32)).cuda())
        unknow_point = Variable(torch.from_numpy(unseen_point.astype(np.float32)).cuda())
    else:
        inputs_instance1 = Variable(torch.from_numpy(X1_train.astype(np.float32)))
        labels_instance1 = Variable(torch.from_numpy(Y_train.astype(np.float32)))
        inputs_instance2 = Variable(torch.from_numpy(X2_train.astype(np.float32)))
        labels_instance2 = Variable(torch.from_numpy(Y_train.astype(np.float32)))
        unknow_point = Variable(torch.from_numpy(unseen_point.astype(np.float32)))

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(args.epochs))
    train(model,model2, inputs_instance1,inputs_instance2,labels_instance1,labels_instance2,epochs, criterion ,optimizer, restore_dir=None)