import tensorflow as tf
import numpy as np
from keras import backend as K
from tensorflow.contrib import distributions


# standard categorical cross entropy
# N data points, C classes
# true - true values. Shape: (N, C)
# pred - predicted values. Shape: (N, C)
# returns - loss (N)
def categorical_cross_entropy(true, pred):

    return np.sum(true * np.log(pred), axis=1)


# Bayesian categorical cross entropy.
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N,)
def bayesian_categorical_crossentropy(T, num_classes):
  def bayesian_categorical_crossentropy_internal(true, pred_var):
    # shape: (N,)
    std = K.sqrt(pred_var[:, num_classes:])
    # shape: (N,)
    variance = pred_var[:, num_classes]
    variance_depressor = K.exp(variance) - K.ones_like(variance)
    # shape: (N, C)
    pred = pred_var[:, 0:num_classes]
    # shape: (N,)
    undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
    # shape: (T,)
    iterable = K.variable(np.ones(T))
    dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
    monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable, name='monte_carlo_results')
    
    variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss
    return variance_loss + undistorted_loss + variance_depressor
  return bayesian_categorical_crossentropy_internal


# for a single monte carlo simulation, 
#   calculate categorical_crossentropy of 
#   predicted logit values plus gaussian 
#   noise vs true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# dist - normal distribution to sample from. Shape: (N, C)
# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
# num_classes - the number of classes. C
# returns - total differences for all classes (N,)
def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
  def map_fn(i):
    std_samples = K.transpose(dist.sample(num_classes))
    distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
    diff = undistorted_loss - distorted_loss
    return -K.elu(diff)

  return map_fn


if __name__ == '__main__':
    print('test bayesian deep learning')