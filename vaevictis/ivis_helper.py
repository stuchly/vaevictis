import numpy as np
import random
import sys


from .knn_annoy import build_annoy_index, extract_knn
import tensorflow as tf
import tensorflow.keras.backend as K



#####
#Calcule les arrays positif et négatif à partir de la matrice des KNN, puis forme les batchs
#####

def input_compute(x,k=16,knn_matrix=None): #x représente un dataset 
  if knn_matrix is None:
    build_annoy_index(x,"ind",build_index_on_disk=True)
    knn_matrix = extract_knn(x,"ind")
  positive = np.empty(np.shape(x))
  negative = np.empty(np.shape(x))

  for i, clust in enumerate(knn_matrix):
    positive[i,:] = x[random.choice(clust),:]
    negative[i,:] = x[random.choice(knn_matrix[:,0]),:]

  inputs = [x,positive,negative]
      
  return inputs #renvoie les indices des points choisis



#####
#Fonction de coût/Loss function
#####

def _euclidean_distance(x, y):
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def pn_loss(y_pred):    
  anchor, positive, negative = y_pred
  anchor_positive_distance = _euclidean_distance(anchor, positive)
  anchor_negative_distance = _euclidean_distance(anchor, negative)
  positive_negative_distance = _euclidean_distance(positive, negative)

  minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)

  return K.mean(K.maximum(anchor_positive_distance - minimum_distance + 1., 0))

def euclidean_distance(x, y):
  return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), tf.cast(K.epsilon(),dtype="float64")))
  
def cosine_distance(x, y):
  def norm(t):
    return K.sqrt(K.maximum(K.sum(K.square(t), axis=1, keepdims=True), tf.cast(K.epsilon(),dtype="float64")))
  return K.sum(x * y, axis = 1, keepdims = True)/(norm(x)*norm(y))

def pn_loss_builder(distance="euclidean", margin=1.):  
  
    
  if distance == "euclidean":
    _distance_ = euclidean_distance
  elif distance == "cosine":
    _distance = cosine_distance
      
  def _pn_loss(y_pred):    
    anchor, positive, negative = y_pred
  
    anchor_positive_distance = _distance(anchor, positive)
    anchor_negative_distance = _distance(anchor, negative)
    positive_negative_distance = euclidean_distance(positive, negative)
  
    minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=0, keepdims=True)
  
    return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))
  return _pn_loss

