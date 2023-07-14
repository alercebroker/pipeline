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
"""
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

import numpy as np
import modules.lrp_modules.module as module

# from modules.train import Train
import tensorflow as tf

# no lo usa el 'na'
na = np.newaxis


# -------------------------------
# Sequential layer
# -------------------------------
class Sequential(module.Module):
    """
    Top level access point and incorporation of the neural network implementation.
    Sequential manages a sequence of computational neural network modules and passes
    along in- and outputs.
    """

    def __init__(self, modules):
        """
        Constructor

        Parameters
        ----------
        modules : list, tuple, etc. enumerable.
            an enumerable collection of instances of class Module
        """
        module.Module.__init__(self)
        self.modules = modules
        self.Rel = []  # ADD
        # self.conv = [] #ADD
        # self.act = [] #ADD
        module.layer_count = 0

    def forward(self, X):
        """
        Realizes the forward pass of an input through the net

        Parameters
        ----------
        X : numpy.ndarray
            a network input.

        Returns
        -------
        X : numpy.ndarray
            the output of the network's final layer
        """
        # if 'conv' in self.modules[0].name:
        #    if self.modules[0].batch_size is None or self.modules[0].input_depth is None or self.modules[0].input_dim is None:
        #        raise ValueError('Expects batch_input_shape= AND input_depth= AND input_dim= for the first layer ')
        # elif 'linear' in self.modules[0].name:
        #    if self.modules[0].batch_size is None or self.modules[0].input_dim is None:
        #        raise ValueError('Expects batch_input_shape= AND input_dim= for the first layer ')

        print("Forward Pass ... ")
        print("------------------------------------------------- ")
        print("input::", X.get_shape().as_list())
        for m in self.modules:
            m.batch_size = self.modules[0].batch_size
            X = m.forward(X)
            print(m.name + "::", X.get_shape().as_list())
            try:
                True
                # self.act.append(m.activations)
            except:
                print("Not appended")

        print("softmax::", X.get_shape().as_list())

        print("\n" + "------------------------------------------------- ")

        return X

    def clean(self):
        """
        Removes temporary variables from all network layers.
        """
        for m in self.modules:
            m.clean()

    # set parameter for al LRP in layers
    def set_lrp_parameters(self, lrp_var=None, param=None):
        for m in self.modules:
            m.set_lrp_parameters(lrp_var=lrp_var, param=param)

    def lrp(self, R, lrp_var=None, param=None):
        """
        Performs LRP by calling subroutines, depending on lrp_var and param or
        preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)

        If lrp parameters have been pre-specified (per layer), the corresponding decomposition
        will be applied during a call of lrp().

        Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
        will override the preset values for the current call.

        How to use:

        net.forward(X) #forward feed some data you wish to explain to populat the net.

        then either:

        net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer

        or:

        for m in net.modules:
            m.set_lrp_parameters(...)
        net.lrp() #to preset a lrp configuration to each layer in the net

        or:

        net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.

        Parameters
        ----------
        R : numpy.ndarray
            final layer relevance values. usually the network's prediction of some data points
            for which the output relevance is to be computed
            dimensionality should be equal to the previously computed predictions

        lrp_var : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------

        R : numpy.ndarray
            the first layer relevances as produced by the neural net wrt to the previously forward
            passed input data. dimensionality is equal to the previously into forward entered input data

        Note
        ----

        Requires the net to be populated with temporary variables, i.e. forward needed to be called with the input
        for which the explanation is to be computed. calling clean in between forward and lrp invalidates the
        temporary data
        """
        print("Computing LRP ... ")
        print("------------------------------------------------- ")
        self.Rel = []  # ADD
        self.Rel.append(R)
        for m in self.modules[::-1]:
            R = m.lrp(R, lrp_var, param)
            self.Rel.append(R)  # ADD
            print(m.name + "::", R.get_shape().as_list())

        print("\n" + "------------------------------------------------- ")

        return R

    def lrpPy(self, sess, feed_dict, R, lrp_var=None, param=None):
        print("Computing LRP ... ")
        print("------------------------------------------------- ")
        # self.Rel = [] #ADD
        # self.Rel.append(R)
        for m in self.modules[::-1]:
            R = m.lrpPy(sess, feed_dict, R, lrp_var, param)
            # self.Rel.append(R) #ADD
            print(m.name + "::", R.shape)

        print("\n" + "------------------------------------------------- ")

        return R

    # not used in this class
    # can be calles outside to performe LRP in a specific layer,
    # given relevance of previous layer
    def lrp_layerwise(self, m, R, lrp_var=None, param=None):
        R = m.lrp(R, lrp_var, param)
        m.clean()
        print(m.name + "::", R.get_shape().as_list())
        return R

    # def fit(self,output=None,ground_truth=None, opt_params=[]):
    #     return Train(output,ground_truth,opt_params)

    def getWeights(self):
        weights = []
        biases = []
        for m in self.modules:
            try:
                W = m.weights
                B = m.biases
                weights.append(W)
                biases.append(B)
            except:
                pass
                # print("layer without params")

        return weights, biases

    def getBNparams(self):
        gamma = []
        beta = []
        var = []
        mean = []
        for m in self.modules:
            try:
                g = m.gamma
                b = m.beta
                v = m.var
                me = m.mean
                gamma.append(g)
                beta.append(b)
                var.append(v)
                mean.append(me)
            except:
                print("not a BN layer")

        return beta, gamma, mean, var

    def loadWeights(self, Ncnn, Nfc, path):
        # m = []
        # i = 0
        Ncnn_aux = 0
        Nfc_aux = 0
        for m in self.modules:
            # m.append(n)

            if Ncnn_aux < Ncnn:
                cnn_path = path + "CNN" + str(Ncnn_aux + 1)

                try:
                    # print(cnn_path)
                    m.weights = tf.Variable(
                        tf.convert_to_tensor(np.load(cnn_path + "-W.npy"), np.float32),
                        name="weights",
                    )
                    m.biases = tf.Variable(
                        tf.convert_to_tensor(np.load(cnn_path + "-B.npy"), np.float32),
                        name="biases",
                    )
                    Ncnn_aux += 1
                except Exception as e:
                    print("Conv layer without params")

            if Nfc_aux < Nfc and Ncnn_aux == Ncnn:
                fc_path = path + "FC" + str(Nfc_aux + 1)

                try:
                    # print(fc_path)
                    m.weights = tf.Variable(
                        tf.convert_to_tensor(np.load(fc_path + "-W.npy"), np.float32),
                        name="weights",
                    )
                    m.biases = tf.Variable(
                        tf.convert_to_tensor(np.load(fc_path + "-B.npy"), np.float32),
                        name="biases",
                    )
                    Nfc_aux += 1
                except:
                    print("linear layer without params")
        print("-Params Loaded-")
