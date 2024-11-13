################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import matplotlib.pyplot as plt
import pickle

import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    prediction_labels = np.argmax(predictions, axis=1)
    acc = (prediction_labels == targets).sum() / targets.shape[0]

    #######################
    # END OF YOUR CODE    #
    #######################

    return acc


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    accuracies_scaled = []
    total_batch_size = 0
    for batch, targets in data_loader:
        batch_size = targets.shape[0]
        total_batch_size += batch_size
        batch = batch.reshape(batch_size, -1)
        accuracies_scaled.append(batch_size * accuracy(model.forward(batch), targets))
    avg_accuracy = sum(accuracies_scaled) / total_batch_size

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=3072, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation

    best_acc = 0
    best_model = None
    val_accuracies = []

    train_losses = []

    for epoch in range(epochs):
        
        epoch_loss = []
        for batch, targets in cifar10_loader["train"]:
            batch = batch.reshape(batch_size, -1)
            preds = model.forward(batch)
            train_loss = loss_module.forward(preds, targets)
            epoch_loss.append(train_loss)
            model.backward(loss_module.backward(preds, targets))

            # if loss < best_loss:
            #     best_loss = loss
            #     best_model = deepcopy(model)

            #Step
            for layer in model.layers:
                if hasattr(layer, 'params'):
                    layer.params['weight'] -= lr * layer.grads['weight']
                    layer.params['bias'] -= lr * layer.grads['bias']
            
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_acc)

        train_losses.append(epoch_loss)

        print(f"Epoch: {epoch}", "#"*70)
        print(f"Val acc: {val_acc}")
        print( "#"*70)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model)

        

    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    print(f"Learning finished: {epoch}", "#"*70)
    print(f"Test acc: {test_accuracy}")
    print( "#"*70)
    # TODO: Add any information you might want to save for plotting
    logging_dict = {"train_loss": train_losses}
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    
    # train(**kwargs)

    # Code below produces graphs for batch_norm = True and False separatly 
    # if you want to reproduce them - just comment the train(**kwargs) above and uncode everyhting below 
    # and run this file 

    if os.path.isfile('assignment1/train_loss.pkl'):
      with open('assignment1/train_loss.pkl', 'rb') as f:
          val_acc, train_loss = pickle.load(f)
    else: 
      _, val_acc, _, logging_dict = train(**kwargs)
      train_loss = np.array(logging_dict["train_loss"]).reshape(-1)
      with open('assignment1/train_loss.pkl', 'wb') as f:
        pickle.dump((val_acc, train_loss), f)

    plt.figure()

    plt.plot(np.arange(0, train_loss.size) / 351, train_loss, linewidth=0.5, label='train loss')


    moving_average = np.convolve(train_loss, np.ones(31) / 31, mode='valid')

    plt.plot(np.arange(15, train_loss.size - 15) / 351, moving_average, linewidth=0.5, label="moving average 31")

    moving_average = np.convolve(train_loss, np.ones(351) / 351, mode='valid')

    plt.plot(np.arange(175, train_loss.size - 175) / 351, moving_average, linewidth=1.5, label="moving average 351")

    plt.legend()
    plt.ylabel('train loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig('assignment1/train_loss.png', dpi=1000)

    plt.figure()

    plt.plot(val_acc, label='validation accuracy')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.title('validation accuracy')
    plt.savefig('assignment1/val_acc.png', dpi=1000)
    # Feel free to add any additional functions, such as plotting of the loss curve here
