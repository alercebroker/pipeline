"""
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause


9/10  ADD if relu
"""
import os
import sys
import tensorflow as tf

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from modules.lrp_modules.module import Module
import modules.lrp_modules.variables as variables

# import activations
import numpy as np

na = np.newaxis


class Linear(Module):
    """
    Linear Layer
    """

    def __init__(
        self,
        output_dim,
        batch_size=None,
        input_dim=None,
        act="linear",
        keep_prob=tf.constant(1.0),
        weights_init=tf.truncated_normal_initializer(stddev=0.05),
        bias_init=tf.constant_initializer(0.05),
        name="linear",
        param_dir=None,
        use_dropout=False,
        init_DH=False,
    ):
        self.name = name
        # Se crea un objeto Module de module.py, el que lleva el conteo de las capas presentes
        # y un preset a las variables usadas por LRP en cada capa
        Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.act = act
        self.keep_prob = keep_prob
        self.use_dropout = use_dropout

        self.weights_init = weights_init
        self.bias_init = bias_init

        # path to stored weights and biases
        self.param_dir = param_dir
        self.init_DH = init_DH

    # Forward pass, aqui se le pasa el tensor proveniente de la capa anterior o entrada

    ##Aqui no se inicializan los pesos en cada forward que se haga!??!?!?!?!
    def forward(self, input_tensor):
        # input tensor shape=(samples, feturemap_h, featuremap_w, channels)
        # get input sensor and its dimensions
        self.input_tensor = input_tensor
        inp_shape = self.input_tensor.get_shape().as_list()

        # import pdb;pdb.set_trace()
        # if input shape diferente from 2 (a tensor) input tensor its reshaped as [batchsize, inputs]
        # this is meant to flatten incoming tensors!!!!!!!!
        if len(inp_shape) != 2:
            # import numpy as np
            # get number of flat inputs
            self.input_dim = np.prod(inp_shape[1:])
            # 9/10 changed inp_shape[0] to -1
            self.input_tensor = tf.reshape(self.input_tensor, [-1, self.input_dim])

        # if they are already flat define dimension of input
        else:
            self.input_dim = inp_shape[1]

        self.weights_shape = [self.input_dim, self.output_dim]

        # with tf.name_scope(self.name):
        with tf.name_scope(self.name):
            # initialice random weights and biases
            with tf.name_scope(self.name + "_params"):
                if self.param_dir == None:
                    if self.init_DH:
                        self.weights_init = tf.truncated_normal_initializer(
                            stddev=np.sqrt(2 / (self.input_dim))
                        )
                        self.bias_init = tf.constant_initializer(0.0)
                    # call variables.py and set parameters
                    self.weights = variables.weights(
                        self.weights_shape,
                        initializer=self.weights_init,
                        name="weights",
                    )
                    self.biases = variables.biases(
                        self.output_dim, initializer=self.bias_init, name="biases"
                    )

                # CHECK!!!!!!!!!!!!!!1 Correctly
                # CHECK how to correctly pass weights to method
                # 9/10 tf.Variable() is added
                # 9/10 change from txt to np.load, see if a try can be added
                else:
                    # self.biases =  tf.Variable(tf.convert_to_tensor(np.loadtxt(self.param_dir+'-B.txt'), np.float32))
                    # self.weights = tf.Variable(tf.convert_to_tensor(np.loadtxt(self.param_dir+'-W.txt'), np.float32))
                    self.biases = tf.Variable(
                        tf.convert_to_tensor(
                            np.load(self.param_dir + "-B.npy"), np.float32
                        ),
                        name="biases",
                    )
                    self.weights = tf.Variable(
                        tf.convert_to_tensor(
                            np.load(self.param_dir + "-W.npy"), np.float32
                        ),
                        name="weights",
                    )

            # WHAWT this for?
            # create instance in graph model?
            # with tf.name_scope(self.name):
            # SCORES before activations
            linear = tf.nn.bias_add(
                tf.matmul(self.input_tensor, self.weights), self.biases, name=self.name
            )
            # IF attribute act is a string aplly activation function
            # if isinstance(self.act, str):
            # call activation from activations.py
            # self.activations = activations.apply(linear, self.act)
            if self.act == "relu":
                self.activations = tf.nn.relu(linear)
            elif self.act == "lrelu":
                self.activations = tf.maximum(linear, 0.01 * linear)
            else:
                self.activations = linear

                # Dropout
            if self.use_dropout:
                self.activations = tf.nn.dropout(
                    self.activations, keep_prob=self.keep_prob
                )
            # Otherwise call for convolutional
            # elif hasattr(self.act, '__call__'):
            #    self.activations = self.act(conv)

            # def dropout_check_false():
            # print('Dropout adjusted 1.0')
            #    return tf.constant(1.0)

            # def dropout_check_true():
            #    return tf.multiply(self.keep_prob, 1)

            # dropout_check = self.keep_prob<=tf.constant(1.0)

            # extremly rare way to set dropout
            # dropout = tf.cond(dropout_check, dropout_check_true, dropout_check_false)

            # apply dropout to activations
        # 9/10 dropout commented
        # self.activations = tf.nn.dropout(self.activations, keep_prob=self.keep_prob)
        # activations = activation_fn(conv, name='activation')
        # tf.summary.histogram('activations', self.activations)
        # tf.summary.histogram('weights', self.weights)
        # tf.summary.histogram('biases', self.biases)

        return self.activations

    # CAMBIO PARA LRP AVG POOL
    def check_shape(self, R):
        self.R = R
        # R_shape = self.R.get_shape().as_list()
        R_shape = tf.shape(self.R)
        activations_shape = self.activations.get_shape().as_list()
        # if len(R_shape)!=4:
        # if R_shape[0]!=2:
        self.R = tf.reshape(self.R, [-1, activations_shape[1]])
        # N,self.Hout,self.Wout,NF = self.R.get_shape().as_list()
        # N = tf.shape(self.R)[0]

    # def check_input_shape(self):
    #   if len(self.input_shape)!=2:
    #      raise ValueError('Expected dimension of input tensor: 2')

    def check_shapePy(self, R):
        # R_shape = self.R.get_shape().as_list()
        # R_shape = tf.shape(self.R)
        activations_shape = self.activations.get_shape().as_list()
        # if len(R_shape)!=4:
        if len(R.shape) != 2:
            R = R.reshape((-1, activations_shape[1]))

        return R

    # def lrp(self, R):
    #     return self._simple_lrp(R)

    # CLEAN Relaevance an activations
    def clean(self):
        self.activations = None
        self.R = None

    def _simple_lrp(self, R):
        # self.R = R
        # R_shape = self.R.get_shape().as_list()
        # R_shape = tf.shape(self.R)

        # for non 2 dim arrays
        # if len(R_shape)!=2:
        #    activations_shape = self.activations.get_shape().as_list()
        # self.R = tf.reshape(self.R, activations_shape)
        #    self.R = tf.reshape(self.R, [-1, activations_shape[1]])
        self.check_shape(R)
        # scores of forward pass could be replaced by
        # tf.matmul(self.input_tensor, self.weights)
        # and why not add biases?

        # expan dimensions at end and beginning
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
        Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(
            tf.expand_dims(self.biases, 0), 0
        )
        stabilizer = 1e-9 * (
            tf.where(
                tf.greater_equal(Zs, 0),
                tf.ones_like(Zs, dtype=tf.float32),
                tf.ones_like(Zs, dtype=tf.float32) * -1,
            )
        )
        Zs += stabilizer

        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1), 2)

    # Reveice relevance from previous layer
    def _flat_lrp(self, R):
        """
        distribute relevance for each output evenly to the output neurons' receptive fields.
        note that for fully connected layers, this results in a uniform lower layer relevance map.
        """
        # self.R= R
        self.check_shape(R)
        # expand add a dimension to the beginning
        Z = tf.ones_like(tf.expand_dims(self.weights, 0))
        Zs = tf.reduce_sum(Z, 1, keep_dims=True)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1), 2)

    def _ww_lrp(self, R):
        """
        LRR according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        """
        # self.R= R
        self.check_shape(R)
        Z = tf.square(tf.expand_dims(self.weights, 0))
        Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1)
        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1), 2)

    def _epsilon_lrp(self, R, epsilon):
        """
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        """
        # self.R= R
        self.check_shape(R)
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
        Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(
            tf.expand_dims(self.biases, 0), 0
        )
        Zs += epsilon * tf.where(
            tf.greater_equal(Zs, 0), tf.ones_like(Zs) * -1, tf.ones_like(Zs)
        )

        return tf.reduce_sum((Z / Zs) * tf.expand_dims(self.R, 1), 2)

    def _epsilon_lrpPy(self, sess, feed_dict, R, epsilon=1e-12):
        """
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        """
        R = self.check_shapePy(R)
        W = self.weights.eval(sess)
        B = self.biases.eval(sess)
        # X= sess.run(self.input_tensor, feed_dict=feed_dict)
        X = sess.run(self.input_tensor, feed_dict=feed_dict)
        Z = W[na, :, :] * X[:, :, na]  # localized preactivations
        Zs = Z.sum(axis=1)[:, na, :] + B[na, na, :]  # preactivations

        # add slack to denominator. we require sign(0) = 1. since np.sign(0) = 0 would defeat the purpose of the numeric stabilizer we do not use it.
        Zs += epsilon * ((Zs >= 0) * 2 - 1)
        return ((Z / Zs) * R[:, na, :]).sum(axis=2)

    #    def _alphabeta_lrp(self,R,alpha):
    #        '''
    #        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
    #        '''
    #        self.R= R
    #        beta = alpha-1
    #        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)
    #
    #        if alpha != 0:
    #            Zp = tf.where(tf.greater(Z,0),Z, tf.zeros_like(Z))
    #            term2 = tf.expand_dims(tf.expand_dims(tf.where(tf.greater(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
    #            term1 = tf.expand_dims( tf.reduce_sum(Zp, 1), 1)
    #            Zsp = term1 + term2
    #            Ralpha = alpha * tf.reduce_sum((Zp / Zsp) * tf.expand_dims(self.R, 1),2)
    #        else:
    #            Ralpha = 0
    #
    #        if beta != 0:
    #            Zn = tf.where(tf.less(Z,0),Z, tf.zeros_like(Z))
    #            term2 = tf.expand_dims(tf.expand_dims(tf.where(tf.less(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
    #            term1 = tf.expand_dims( tf.reduce_sum(Zn, 1), 1)
    #            Zsp = term1 + term2
    #            Rbeta = beta * tf.reduce_sum((Zn / Zsp) * tf.expand_dims(self.R, 1),2)
    #        else:
    #            Rbeta = 0
    #
    #        return Ralpha - Rbeta
    #   Stab added!!!!!!!!!!!!!!
    def _alphabeta_lrp(self, R, alpha):
        """
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        """
        # self.R= R
        self.check_shape(R)
        beta = 1 - alpha
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims(self.input_tensor, -1)

        if not alpha == 0:
            Zp = tf.where(tf.greater(Z, 0), Z, tf.zeros_like(Z))
            term2 = tf.expand_dims(
                tf.expand_dims(
                    tf.where(
                        tf.greater(self.biases, 0),
                        self.biases,
                        tf.zeros_like(self.biases),
                    ),
                    0,
                ),
                0,
            )
            term1 = tf.expand_dims(tf.reduce_sum(Zp, 1), 1)
            Zsp = term1 + term2
            # stab added
            epsilon = 1e-16
            stabilizer = epsilon * (
                tf.where(
                    tf.greater_equal(Zsp, 0),
                    tf.ones_like(Zsp, dtype=tf.float32),
                    tf.ones_like(Zsp, dtype=tf.float32) * -1,
                )
            )
            Zsp += stabilizer

            Ralpha = alpha * tf.reduce_sum((Zp / Zsp) * tf.expand_dims(self.R, 1), 2)
        else:
            Ralpha = 0

        if not beta == 0:
            Zn = tf.where(tf.less(Z, 0), Z, tf.zeros_like(Z))
            term2 = tf.expand_dims(
                tf.expand_dims(
                    tf.where(
                        tf.less(self.biases, 0), self.biases, tf.zeros_like(self.biases)
                    ),
                    0,
                ),
                0,
            )
            term1 = tf.expand_dims(tf.reduce_sum(Zn, 1), 1)
            Zsp = term1 + term2
            # stab added
            epsilon = 1e-16
            stabilizer = epsilon * (
                tf.where(
                    tf.greater_equal(Zsp, 0),
                    tf.ones_like(Zsp, dtype=tf.float32),
                    tf.ones_like(Zsp, dtype=tf.float32) * -1,
                )
            )
            Zsp += stabilizer
            Rbeta = beta * tf.reduce_sum((Zn / Zsp) * tf.expand_dims(self.R, 1), 2)
        else:
            Rbeta = 0

        return Ralpha + Rbeta
