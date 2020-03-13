import numpy as np
import random
import sys


from .knn_annoy import build_annoy_index, extract_knn
import tensorflow as tf
import tensorflow.keras.backend as K



#####
#Calcule les arrays positif et négatif à partir de la matrice des KNN, puis forme les batchs
#####

def input_compute(x,k=16): #x représente un dataset 

  build_annoy_index(x,"./ind",build_index_on_disk=False)
  knn_matrix = extract_knn(x,"./ind")
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

def euclidean_distance(x, y):
  return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), tf.cast(K.epsilon(),dtype="float64")))
  
def cosine_distance(x, y):
  def norm(t):
    return K.sqrt(K.maximum(K.sum(K.square(t), axis=1, keepdims=True), tf.cast(K.epsilon(),dtype="float64")))
  return K.sum(x * y, axis = 1, keepdims = True)/(norm(x)*norm(y))

def pn_loss_builder(distance="euclidean", margin=1.):    
  def _pn_loss(y_true, y_pred, distance=distance, margin=margin):    
    anchor, positive, negative = tf.unstack(y_pred)
  
    if distance == "euclidean":
      distance = euclidean_distance
    elif distance == "cosine":
      distance = cosine_distance
  
    anchor_positive_distance = distance(anchor, positive)
    anchor_negative_distance = distance(anchor, negative)
    positive_negative_distance = distance(positive, negative)
  
    minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=0, keepdims=True)
  
    return K.mean(K.maximum(anchor_positive_distance - minimum_distance + margin, 0))
  return _pn_loss

