import numpy as np
import random
import sys


from .knn_annoy import build_annoy_index, extract_knn
import tensorflow as tf
import tensorflow.keras.backend as K


def input_compute(x,k=16,metric="euclidean",knn_matrix=None): #x représente un dataset 
  if knn_matrix is None:
    build_annoy_index(x,"ind",metric=metric,build_index_on_disk=True)
    knn_matrix = extract_knn(x,"ind",metric=metric)
  positive = np.empty(np.shape(x))
  negative = np.empty(np.shape(x))

  for i, clust in enumerate(knn_matrix):
    positive[i,:] = x[random.choice(clust[1:]),:]
    negative[i,:] = x[random.choice(knn_matrix[:,0]),:]

  inputs = [x,positive,negative]
      
  return inputs 


def euclidean_distance(x, y):
  return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
  
def cosine_distance(x, y):
  def norm(t):
    return K.sqrt(K.maximum(K.sum(K.square(t), axis=1, keepdims=True), K.epsilon()))
  return K.sum(x * y, axis = 1, keepdims = True)/(norm(x)*norm(y))

def pn_loss_g(y_pred,distance=euclidean_distance,margin=1.):    
  anchor, positive, negative = y_pred
  anchor_positive_distance = distance(anchor, positive)
  anchor_negative_distance = distance(anchor, negative)
  positive_negative_distance = distance(positive, negative)

  minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)

  return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))
