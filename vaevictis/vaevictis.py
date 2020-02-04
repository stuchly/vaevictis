import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np
from .tsne_helper_njit import compute_transition_probability
from tensorflow.keras.callbacks import EarlyStopping
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
K.set_floatx('float64')
eps_std=tf.constant(1e-2,dtype=tf.float64)
eps_sq=eps_std**2
eta=tf.constant(1e-4,dtype=tf.float64)

def nll(y_true, y_pred):
    """ loss """
    
    return tf.reduce_mean((y_true-y_pred)**2)

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
                 activation="relu",
                 name='encoder',
                 dynamic=True,
                 perplexity=10.,
                 alpha=10.,
                 **kwargs):




        super(Encoder, self).__init__(name=name, **kwargs)
        self.perplexity=perplexity
        self.alpha=alpha
        self.encoder_shape=encoder_shape
        self.drop0=layers.Dropout(rate=0.2)
        self.drop=layers.Dropout(rate=drate)
        self.dense_proj = [None]*len(encoder_shape)
        for i,v in enumerate(self.encoder_shape):
            self.dense_proj[i]=layers.Dense(v,activation=activation)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()


    def callt(self, inputs,training=None):

        x=inputs
        #x=tf.keras.backend.in_train_phase(layers.GaussianNoise(.1)(x),x,training=training)
        p=tf.numpy_function(compute_transition_probability,[x,self.perplexity, 1e-4, 50,False],tf.float64)
        nu = tf.constant(1.0, dtype=tf.float64)
        n=tf.shape(inputs)[0]
        # with tf.GradientTape() as g:
        #     g.watch(inputs)
        #x=inputs
        for dl in self.dense_proj: x=dl(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        # jacobian=g.jacobian(z,inputs) ##computes jacobian, if frob=0. below, only consumes time
        # frob=eta*(tf.norm(jacobian)**2)
        # frob=0.
        # self.add_loss(frob)
        jacobian=0.
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

        #b=tf.constant(10.0,dtype=tf.float64)*(repellant + attraction) / tf.cast(n,tf.float64)
        b=self.alpha*(repellant + attraction) / tf.cast(n,tf.float64)
        #self.add_loss(b)
        return z_mean, z_log_var, z, jacobian, b

    def callv(self, inputs,training=None):

        x=inputs
        #x=tf.keras.backend.in_train_phase(layers.GaussianNoise(.1)(x),x,training=training)
        p=tf.numpy_function(compute_transition_probability,[x,self.perplexity, 1e-4, 50,False],tf.float64)
        nu = tf.constant(1.0, dtype=tf.float64)
        n=tf.shape(inputs)[0]
        # with tf.GradientTape() as g:
        #     g.watch(inputs)
        #x=inputs
        for dl in self.dense_proj: x=dl(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = z_mean
        # jacobian=g.jacobian(z,inputs) ##computes jacobian, if frob=0. below, only consumes time
        # frob=eta*(tf.norm(jacobian)**2)
        # frob=0.
        # self.add_loss(frob)
        jacobian=0.
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

        #b=tf.constant(10.0,dtype=tf.float64)*(repellant + attraction) / tf.cast(n,tf.float64)
        b=self.alpha*(repellant + attraction) / tf.cast(n,tf.float64)
        #self.add_loss(b)
        return z_mean, z_log_var, z, jacobian, b


    def callp(self,inputs):
        x=inputs
        for dl in self.dense_proj: x=dl(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        #z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z_mean, 0.
        
    def call(self,inputs,training=None):
        z_mean, z_log_var, z, jacobian, b = K.in_train_phase(self.callt(inputs),self.callv(inputs), training=training) 
        self.add_loss(b)
        return z_mean, z_log_var, z, jacobian

class Decoder(layers.Layer):

    def __init__(self,
                 original_dim,
                # encoder,
                 activation="relu",
                 drate=0.1,
                 decoder_shape=[32,32],
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.decoder_shape=decoder_shape
        self.drop=layers.Dropout(rate=drate)
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
                 alpha=10.,
                 name='autoencoder',
                 **kwargs):
        super(Vaevictis, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               encoder_shape=encoder_shape,perplexity=perplexity,alpha=alpha)
        self.decoder = Decoder(original_dim, decoder_shape = decoder_shape)

    def call(self, inputs, training=None):
        z_mean, z_log_var, z, _ = self.encoder(inputs,training=training)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var + tf.math.log(eps_sq)- tf.square(z_mean) - eps_sq*tf.exp(z_log_var))
        self.add_loss(kl_loss)
        
        return reconstructed



def dimred(x_train,dim=2,vsplit=0.1,enc_shape=[128,128,128],dec_shape=[128,128,128],perplexity=10.,batch_size=512,epochs=100,patience=0,alpha=10.,save=None,load=None):


    vae = Vaevictis(x_train.shape[1], enc_shape,dec_shape, dim, perplexity, alpha)

    optimizer = tf.keras.optimizers.Adam()
    mse_loss_fn = nll
    vae.compile(optimizer,loss=nll)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=patience)

    
    vae.fit(x_train,x_train,batch_size=batch_size,epochs=epochs,callbacks=[es],validation_split=vsplit,shuffle=True)

    # train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    # @tf.function
    # def train_one_step(m1,optimizer,x):
    #     with tf.GradientTape() as tape:
    #         tape.watch(m1.trainable_weights)
    #         reconstructed = m1(x)
    #         # Compute reconstruction loss
    #         loss = nll(x, reconstructed)

    #         #loss += sum(m1.losses)  # Add KLD regularization loss

    #     grads = tape.gradient(loss, m1.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, m1.trainable_weights))
    #     return loss



    # # Iterate over epochs.
    # def train():
    #     loss=0.
    #     for epoch in range(10):
    #         print('Start of epoch %d' % (epoch,))
    #         for  x_batch_train in train_dataset.batch(128):
    #             loss=train_one_step(vae,optimizer,x_batch_train)
    #     return loss

    # loss=train()
    def predict(data):
        return vae.encoder.callp(data)[0].numpy()
    
    #vae.save("vae_model")
    z_test = vae.encoder.callp(x_train)[0]
    z_test=z_test.numpy()
    return z_test, predict, vae
