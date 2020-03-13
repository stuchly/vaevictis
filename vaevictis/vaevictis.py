import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np
from .tsne_helper_njit import compute_transition_probability
from .ivis_helper import input_compute, pn_loss_builder
from tensorflow.keras.callbacks import EarlyStopping
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
K.set_floatx('float64')

eps_std=tf.constant(1e-2,dtype=tf.float64)
eps_sq=eps_std**2
eta=tf.constant(1e-4,dtype=tf.float64)
def nll(y_true, y_pred):
    """ loss """
    
    return tf.reduce_mean((y_true-y_pred)**2)
    
def nll_builder(ww):
    
    def nll(y_true, y_pred):
        """ loss """
    
        return ww[2]*tf.reduce_mean((y_true-y_pred)**2)
        
    def nll_null(y_true, y_pred):
        return tf.cast(0.,tf.float64)
        
    return nll_null if ww[2]<=0 else nll

def tsne_reg_builder(ww,perplexity):
        
        def tsne_reg(x,z):
            p=tf.numpy_function(compute_transition_probability,[x,perplexity, 1e-4, 50,False],tf.float64)
            # p=compute_transition_probability(x.numpy(),perplexity, 1e-4, 50,False) ## for eager dubugging
            nu = tf.constant(1.0, dtype=tf.float64)
            n=tf.shape(x)[0]
        
            sum_y = tf.reduce_sum(tf.square(z), 1)
            num = tf.constant(-2.0,tf.float64) * tf.matmul(z,
                               z,
                               transpose_b=True) + tf.reshape(sum_y, [-1, 1]) + sum_y
            num = num / nu

            p = p + tf.constant(0.1,tf.float64) / tf.cast(n,tf.float64)
            p = p / tf.expand_dims(tf.reduce_sum(p, 1), 1)

            num = tf.pow(tf.constant(1.0,tf.float64) + num, -(nu + tf.constant(1.0,tf.float64)) / tf.constant(2.0,tf.float64))
            attraction = tf.multiply(p, tf.math.log(num))
            attraction = -tf.reduce_sum(attraction)

            den = tf.reduce_sum(num, 1) - 1
            repellant = tf.reduce_sum(tf.math.log(den))
            return (repellant + attraction) / tf.cast(n,tf.float64)
            
        def null_reg(x,z):
            return 0.
        
        return null_reg if ww[0]<=0 else tsne_reg

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean  + eps_std*tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):

    def __init__(self,
                 drate=0.1,
                 encoder_shape=[32,32],
                 latent_dim=32,
                 activation="selu",
                 name='encoder',
                 dynamic=True,
                 **kwargs):




        super(Encoder, self).__init__(name=name, **kwargs)
        self.encoder_shape=encoder_shape
        # self.drop0=layers.Dropout(rate=0.2)
        self.alphadrop=layers.AlphaDropout(rate=drate)
        self.dense_proj = [None]*len(encoder_shape)
        for i,v in enumerate(self.encoder_shape):
            self.dense_proj[i]=layers.Dense(v,activation=activation)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)


    def call(self,inputs,training=None):
        x = inputs
        for dl in self.dense_proj: x=dl(x)
        return self.dense_mean(x), self.dense_log_var(x)    

class Decoder(layers.Layer):

    def __init__(self,
                 original_dim,
                # encoder,
                 activation="selu",
                 drate=0.1,
                 decoder_shape=[32,32],
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_shape=decoder_shape
        self.drop=layers.Dropout(rate=drate)
        self.alphadrop=layers.AlphaDropout(rate=drate)
        self.dense_proj = [None]*len(decoder_shape)
        for i,v in enumerate(self.decoder_shape):
            self.dense_proj[i]=layers.Dense(v,activation=activation)
        
        self.dense_output = layers.Dense(original_dim) #,kernel_regularizer=l1_l2(l1=0.001, l2=0.001))
    def call(self, inputs, training=None):
        x = inputs
        for dl in self.dense_proj: x=dl(x)
        return self.dense_output(x)


class Vaevictis(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim,
                 encoder_shape=[32,32],
                 decoder_shape=[32,32],
                 latent_dim=32,
                 perplexity=10.,
                 metric="euclidean",
                 margin=1.,
                 ww=[10.,1.,1.],
                 name='Vaevictis',
                 **kwargs):
        super(Vaevictis, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder_shape = encoder_shape
        self.decoder_shape = decoder_shape
        self.latent_dim = latent_dim
        self.perplexity = perplexity
        self.metric=metric
        self.margin=margin
        self.ww = ww
        self.pn=pn_loss_builder(metric, margin)
        self.encoder = Encoder(latent_dim=latent_dim,
                               encoder_shape=encoder_shape,drate=0.1)
        self.decoder = Decoder(original_dim, decoder_shape = decoder_shape, drate=0.1)
        self.sampling = Sampling()
        self.tsne_reg=tsne_reg_builder(ww,self.perplexity)

    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encoder(inputs[0],training=training)
        pos, _ = self.encoder(inputs[1],training=training)
        neg, _ = self.encoder(inputs[2],training=training)
        pnl=self.pn((z_mean,pos,neg))
        self.add_loss(self.ww[1]*pnl)
        b=self.tsne_reg(inputs[0],z_mean)
        self.add_loss(self.ww[0]*b)
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var + tf.math.log(eps_sq)- tf.square(z_mean) - eps_sq*tf.exp(z_log_var))
        self.add_loss(self.ww[3]*kl_loss)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z)
        return reconstructed
        
    def get_config(self):
        return {'original_dim': self.original_dim, 'encoder_shape': self.encoder_shape, 
            'decoder_shape': self.decoder_shape, 'latent_dim': self.latent_dim, 'perplexity': self.perplexity, 'metric': self.metric,
            'margin': self.margin,'ww': self.ww, 'name': self.name}
    
    def save(self, config_file, weights_file):
        
        json_config=self.get_config()
        json.dump(json_config, open(config_file,'w'))
        self.save_weights(weights_file)
    
    def refit(self,x_train,perplexity=10.,batch_size=512,epochs=100,patience=0,alpha=10,vsplit=0.1):
        es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=patience)
        self.fit(x_train,x_train,batch_size=batch_size,epochs=epochs,callbacks=[es],validation_split=vsplit,
        shuffle=True)
        
    def predict_np(self):
        def predict(data):
           return self.encoder(data)[0].numpy()
        return(predict)




def dimred(x_train,dim=2,vsplit=0.1,enc_shape=[128,128,128],dec_shape=[128,128,128],
perplexity=10.,batch_size=512,epochs=100,patience=0,ivis_pretrain=0,ww=[10.,10.,1.],
metric="euclidean",margin=1.,k=30):

    """Wrapper for model build and training

    Parameters
    ----------
    x_train : array, shape (n_samples, n_dims)
              Data to embedd, training dataset
    dim : integer, embedding_dim
    vsplit : float, proportion of data used at validation step - splitted befor shuffling!
    enc_shape : list of integers, shape of the encoder i.e. [128, 128, 128] means 3 dense layers with 128 neurons
    dec_shape : list of integers, shape of the decoder
    perplexity : float, perplexity parameter for tsne regularisation
    batch_size : integer, batch size
    epochs : integer, maximum number of epochs
    patience : integer, callback patience
    ivis_pretrain : integer, number of epochs to run without tsne regularisation as pretraining; not yet implemented
    ww : list of floats, weights on losses in this order: tsne regularisation, ivis pn loss, reconstruction error, KL divergence
    k : integer, number of nearest neighbors
    """
    
    triplets=input_compute(x_train,k)
    #triplets=(x_train,x_train,x_train)
    optimizer = tf.keras.optimizers.Adam()
    if ivis_pretrain>0:
        ww1=ww.copy()
        ww1[0]=-1.
        vae = Vaevictis(x_train.shape[1], enc_shape,dec_shape, dim, perplexity, metric, margin, ww1)
        nll_f=nll_builder(ww1)
        vae.compile(optimizer,loss=nll_f)
        vae.fit(triplets,triplets[0],batch_size=batch_size,epochs=ivis_pretrain,validation_split=vsplit,shuffle=True)
        pre_weight=vae.get_weights()
        
    vae = Vaevictis(x_train.shape[1], enc_shape,dec_shape, dim, perplexity, metric, margin, ww)
    
   
    
    nll_f=nll_builder(ww)
    vae.compile(optimizer,loss=nll_f)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=patience)
    
    if ivis_pretrain>0:
        aux=vae((x_train[:10,],x_train[:10,],x_train[:10,])) # instantiate model
        vae.set_weights(pre_weight)

    vae.fit(triplets,triplets[0],batch_size=batch_size,epochs=epochs,callbacks=[es],validation_split=vsplit,shuffle=True)

    # train_dataset = tf.data.Dataset.from_tensor_slices(triplets) ## eager debugging
    #@tf.function
    # def train_one_step(m1,optimizer,x):
    #     with tf.GradientTape() as tape:
    #         tape.watch(m1.trainable_weights)
    #         reconstructed = m1(x)
    #         # Compute reconstruction loss
    #         loss = nll(x, reconstructed)
    # 
    #         loss += sum(m1.losses)  # Add KLD regularization loss
    # 
    #     grads = tape.gradient(loss, m1.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, m1.trainable_weights))
    #     return loss
    #
    #
    #
    # # # Iterate over epochs.
    # def train():
    #     loss=0.
    #     for epoch in range(epochs):
    #         print('Start of epoch %d' % (epoch,))
    #         for  x_batch_train in train_dataset.batch(128):
    #             loss=train_one_step(vae,optimizer,x_batch_train)
    #     return loss
    # 
    # loss=train()

    def predict(data):
        return vae.encoder(data)[0].numpy()
    
    z_test = vae.encoder(x_train)[0]
    z_test=z_test.numpy()
    return z_test, predict, vae, vae.get_config(), vae.get_weights()
    
def loadModel(config_file,weights_file):
    config = json.load(open(config_file))
    new_model=Vaevictis(config["original_dim"], config["encoder_shape"],
    config["decoder_shape"], config["latent_dim"], config["perplexity"], 
    config["metric"], config["margin"], config["ww"])
    
    optimizer = tf.keras.optimizers.Adam()
    nll_f=nll_builder(config["ww"])
    new_model.compile(optimizer,loss=nll_f)
    x=np.random.rand(10,config["original_dim"])
    new_model.train_on_batch(x,x)
    new_model.load_weights(weights_file)
    return new_model

    
