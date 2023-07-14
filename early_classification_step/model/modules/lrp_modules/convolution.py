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
import tensorflow as tf

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from modules.lrp_modules.module import Module
import modules.lrp_modules.variables as variables

# import pdb
# import activations
import numpy as np

na = np.newaxis


from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


class Convolution(Module):
    """
    Convolutional Layer
    """

    def __init__(
        self,
        output_depth,
        batch_size=None,
        input_dim=None,
        input_depth=None,
        kernel_size=5,
        stride_size=1,
        act="linear",
        keep_prob=1.0,
        pad="SAME",
        weights_init=tf.truncated_normal_initializer(stddev=0.05),
        bias_init=tf.constant_initializer(0.05),
        name="conv2d",
        param_dir=None,
        init_DH=False,
    ):

        self.name = name
        # self.input_tensor = input_tensor
        Module.__init__(self)

        # comment (it's done in fwd pass)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_depth = input_depth

        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.act = act
        self.keep_prob = keep_prob
        self.pad = pad

        self.weights_init = weights_init
        self.bias_init = bias_init

        # path to stored weights and biases
        self.param_dir = param_dir
        self.init_DH = init_DH

    def check_input_shape(self):
        inp_shape = self.input_tensor.get_shape().as_list()
        try:
            if len(inp_shape) != 4:
                # try change slef.batch_size to -1
                mod_shape = [
                    self.batch_size,
                    self.input_dim,
                    self.input_dim,
                    self.input_depth,
                ]
                self.input_tensor = tf.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError("Expected dimension of input tensor: 4")

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = tf.shape(self.input_tensor)[0]
        self.input_dim = tf.shape(self.input_tensor)[1]
        self.input_depth = tf.shape(self.input_tensor)[3]
        # pdb.set_trace()
        self.check_input_shape()
        # input images, featuremap height, feturemap width, num_input_channels
        # _
        (
            self.in_N,
            self.in_h,
            self.in_w,
            self.in_depth,
        ) = self.input_tensor.get_shape().as_list()
        self.in_N = tf.shape(self.input_tensor)[0]

        # init weights
        self.weights_shape = [
            self.kernel_size,
            self.kernel_size,
            self.in_depth,
            self.output_depth,
        ]
        self.strides = [1, self.stride_size, self.stride_size, 1]

        with tf.name_scope(self.name):
            # initialice random weights and biases
            with tf.name_scope(self.name + "_params"):
                if self.param_dir == None:
                    if self.init_DH:
                        self.weights_init = tf.truncated_normal_initializer(
                            stddev=np.sqrt(
                                2 / (self.input_dim * self.input_dim * self.input_depth)
                            )
                        )
                        self.bias_init = tf.constant_initializer(0.0)
                    # call variables.py and set parameters
                    self.weights = variables.weights(
                        self.weights_shape,
                        initializer=self.weights_init,
                        name="weights",
                    )
                    self.biases = variables.biases(
                        self.output_depth, initializer=self.bias_init, name="biases"
                    )

                # CHECK!!!!!!!!!!!!!!1 Coorectly!
                # CHECK how to correctly pass weights to method
                # 9/10 tf.Variable() is added
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

            # with tf.name_scope(self.name):
            # print("hola")
            conv = tf.nn.conv2d(
                self.input_tensor, self.weights, strides=self.strides, padding=self.pad
            )

            conv += self.biases

            self.conv = conv

            # CHECK this reshape (after all working) simple conv +=biases
            # conv = tf.reshape(tf.nn.bias_add(conv, self.biases), conv.get_shape().as_list())

            #            if isinstance(self.act, str):
            #                self.activations = activations.apply(conv, self.act)
            #            elif hasattr(self.act, '__call__'):
            #                self.activations = self.act(conv)
            if self.act == "relu":
                self.activations = tf.nn.relu(conv)
            elif self.act == "lrelu":
                self.activations = tf.maximum(self.conv, 0.01 * self.conv)
            else:
                self.activations = conv

        # if self.keep_prob<1.0:
        #    self.activations = tf.nn.dropout(self.activations, keep_prob=self.keep_prob)

        # dont know if i want them
        # tf.summary.histogram('activations', self.activations)
        # tf.summary.histogram('weights', self.weights)
        # tf.summary.histogram('biases', self.biases)

        return self.activations

    def check_shapePy(self, R):

        activations_shape = self.activations.get_shape().as_list()
        # if len(R_shape)!=4:
        if len(R.shape) != 4:
            R = R.reshape(
                (-1, activations_shape[1], activations_shape[2], activations_shape[3])
            )

        return R

    def _simple_lrp(self, R):
        """
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        """

        self.check_shape(R)

        image_patches = self.extract_patches()
        Z = self.compute_z(image_patches)
        Zs = self.compute_zs(Z, self.biases)
        result = self.compute_result(Z, Zs)
        return self.restitch_image(result)

    def _epsilon_lrp(self, R, epsilon=1e-12):
        """
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        """
        self.check_shape(R)

        image_patches = self.extract_patches()
        Z = self.compute_z(image_patches)
        Zs = self.compute_zs(Z, self.biases, epsilon=epsilon)
        result = self.compute_result(Z, Zs)
        return self.restitch_image(result)

    def _epsilon_lrpPy(self, sess, feed_dict, R, epsilon=1e-12):
        """
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        """
        R = self.check_shapePy(R)
        W = self.weights.eval(sess)
        B = self.biases.eval(sess)
        X = sess.run(self.input_tensor, feed_dict=feed_dict)

        N, Hout, Wout, NF = R.shape
        hf, wf, df, NF = W.shape
        hstride = self.stride_size
        wstride = self.stride_size

        # print(R.shape)
        # print(W.shape)
        # print(X.shape)
        # print(self.stride_size)

        Rx = np.zeros_like(X, dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                # if(i*hstride+hf>Hout or j*wstride+wf>Wout):
                #    continue
                # print(W[na,...].shape)
                # print(X[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : , na].shape)
                # print("j:",j," wstride:",wstride, " wf:",wf)
                try:
                    Z = (
                        W[na, ...]
                        * X[
                            :,
                            i * hstride : i * hstride + hf,
                            j * wstride : j * wstride + wf,
                            :,
                            na,
                        ]
                    )
                    Zs = Z.sum(axis=(1, 2, 3), keepdims=True) + B[na, na, na, na, ...]
                    Zs += epsilon * ((Zs >= 0) * 2 - 1)
                    Rx[
                        :,
                        i * hstride : i * hstride + hf :,
                        j * wstride : j * wstride + wf :,
                        :,
                    ] += ((Z / Zs) * R[:, i : i + 1, j : j + 1, na, :]).sum(axis=4)
                except:
                    continue
        return Rx

    def _ww_lrp(self, R):
        """
        LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        """
        self.check_shape(R)

        image_patches = tf.ones(
            [
                self.in_N,
                self.Hout,
                self.Wout,
                self.kernel_size,
                self.kernel_size,
                self.in_depth,
            ]
        )
        # pdb.set_trace()
        ww = tf.square(self.weights)
        Z = tf.expand_dims(ww, 0)
        # self.Z = tf.expand_dims(tf.tile(tf.reshape(ww, [1,1,self.kernel_size, self.kernel_size, self.in_depth, self.output_depth]), [self.Hout, self.Wout, 1,1,1,1]), 0)
        # self.Z = tf.expand_dims(tf.square(self.weights), 0) * tf.expand_dims(image_patches, -1)
        # self.Zs = tf.reduce_sum(self.Z, [3,4,5], keep_dims=True)
        Zs = tf.reduce_sum(Z, [1, 2, 3], keepdims=True)
        result = self.compute_result(Z, Zs)
        return self.restitch_image(result)

    def _flat_lrp(self, R):
        """
        distribute relevance for each output evenly to the output neurons' receptive fields.
        """
        self.check_shape(R)

        Z = tf.ones(
            [
                self.in_N,
                self.Hout,
                self.Wout,
                self.kernel_size,
                self.kernel_size,
                self.in_depth,
                self.output_depth,
            ]
        )
        Zs = self.compute_zs(Z, self.biases)
        result = self.compute_result(Z, Zs)
        return self.restitch_image(result)

    # Cambio en funcion compute_zs, nose si es necesairo dejar el stibilizer false #changed to True
    def _alphabeta_lrp(self, R, alpha):
        """
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        """
        beta = 1 - alpha
        self.check_shape(R)

        image_patches = self.extract_patches()
        Z = self.compute_z(image_patches)
        if not alpha == 0:
            Zp = tf.where(tf.greater(Z, 0), Z, tf.zeros_like(Z))
            Bp = tf.where(
                tf.greater(self.biases, 0), self.biases, tf.zeros_like(self.biases)
            )
            Zsp = self.compute_zs(Zp, Bp, stabilizer=True)
            Ralpha = alpha * self.compute_result(Zp, Zsp)
        else:
            Ralpha = 0
        if not beta == 0:
            Zn = tf.where(tf.less(Z, 0), Z, tf.zeros_like(Z))
            Bn = tf.where(
                tf.less(self.biases, 0), self.biases, tf.zeros_like(self.biases)
            )
            Zsn = self.compute_zs(Zn, Bn, stabilizer=True)
            Rbeta = beta * self.compute_result(Zn, Zsn)
        else:
            Rbeta = 0

        result = Ralpha + Rbeta
        return self.restitch_image(result)

    def check_shape(self, R):
        self.R = R
        # R_shape = self.R.get_shape().as_list()
        R_shape = tf.shape(self.R)
        activations_shape = self.activations.get_shape().as_list()
        # if len(R_shape)!=4:
        if R_shape[0] != 4:
            self.R = tf.reshape(
                self.R,
                [-1, activations_shape[1], activations_shape[2], activations_shape[3]],
            )
        N, self.Hout, self.Wout, NF = self.R.get_shape().as_list()
        N = tf.shape(self.R)[0]

    def extract_patches(self):
        image_patches = tf.extract_image_patches(
            self.input_tensor,
            ksizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride_size, self.stride_size, 1],
            rates=[1, 1, 1, 1],
            padding=self.pad,
        )
        return tf.reshape(
            image_patches,
            [
                self.in_N,
                self.Hout,
                self.Wout,
                self.kernel_size,
                self.kernel_size,
                self.in_depth,
            ],
        )

    def compute_z(self, image_patches):
        return tf.multiply(
            tf.expand_dims(self.weights, 0), tf.expand_dims(image_patches, -1)
        )

    # Cambio en funcion compute_zs, se agrega bias
    def compute_zs(self, Z, bias, stabilizer=True, epsilon=1e-12):
        Zs = tf.reduce_sum(Z, [3, 4, 5], keepdims=True) + tf.expand_dims(bias, 0)
        if stabilizer == True:
            stabilizer = epsilon * (
                tf.where(
                    tf.greater_equal(Zs, 0),
                    tf.ones_like(Zs, dtype=tf.float32),
                    tf.ones_like(Zs, dtype=tf.float32) * -1,
                )
            )
            Zs += stabilizer
        return Zs

    def compute_result(self, Z, Zs):
        result = tf.reduce_sum(
            (Z / Zs)
            * tf.reshape(
                self.R, [self.in_N, self.Hout, self.Wout, 1, 1, 1, self.output_depth]
            ),
            6,
        )
        return tf.reshape(
            result,
            [
                self.in_N,
                self.Hout,
                self.Wout,
                self.kernel_size * self.kernel_size * self.in_depth,
            ],
        )

    def restitch_image(self, result):
        return self.patches_to_images(
            result,
            self.in_N,
            self.in_h,
            self.in_w,
            self.in_depth,
            self.Hout,
            self.Wout,
            self.kernel_size,
            self.kernel_size,
            self.stride_size,
            self.stride_size,
        )

    def clean(self):
        self.activations = None
        self.R = None

    def patches_to_images(
        self,
        grad,
        batch_size,
        rows_in,
        cols_in,
        channels,
        rows_out,
        cols_out,
        ksize_r,
        ksize_c,
        stride_h,
        stride_r,
    ):
        rate_r = 1
        rate_c = 1
        padding = self.pad

        ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
        ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

        if padding == "SAME":
            rows_out = int(ceil(rows_in / stride_r))
            cols_out = int(ceil(cols_in / stride_h))
            pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
            pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

        elif padding == "VALID":
            rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
            cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
            pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
            pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

        pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

        grad_expanded = array_ops.transpose(
            array_ops.reshape(
                grad, (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels)
            ),
            (1, 2, 3, 4, 0, 5),
        )
        grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

        row_steps = range(0, rows_out * stride_r, stride_r)
        col_steps = range(0, cols_out * stride_h, stride_h)

        idx = []
        for i in range(rows_out):
            for j in range(cols_out):
                r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
                r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

                idx.extend(
                    [
                        (
                            r * (cols_in) + c,
                            i * (cols_out * ksize_r * ksize_c)
                            + j * (ksize_r * ksize_c)
                            + ri * (ksize_c)
                            + ci,
                        )
                        for (ri, r) in enumerate(range(r_low, r_high, rate_r))
                        for (ci, c) in enumerate(range(c_low, c_high, rate_c))
                        if 0 <= r and r < rows_in and 0 <= c and c < cols_in
                    ]
                )

        sp_shape = (rows_in * cols_in, rows_out * cols_out * ksize_r * ksize_c)

        sp_mat = sparse_tensor.SparseTensor(
            array_ops.constant(idx, dtype=ops.dtypes.int64),
            array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
            sp_shape,
        )

        jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

        grad_out = array_ops.reshape(jac, (rows_in, cols_in, batch_size, channels))
        grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))

        return grad_out

    # OLD METHODS BELOW
    def padding(self, i, j, pad_in_h, pad_in_w, hstride, wstride, hf, wf):
        pad_bottom = (
            pad_in_h - (i * hstride + hf) if (pad_in_h - (i * hstride + hf)) > 0 else 0
        )
        pad_top = i * hstride
        pad_right = (
            pad_in_w - (j * wstride + wf) if (pad_in_w - (j * wstride + wf) > 0) else 0
        )
        pad_left = j * wstride
        return [[pad_top, pad_bottom], [pad_left, pad_right]]

    def __simple_lrp(self, R):
        """
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        """
        import time

        start_time = time.time()

        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape) != 4:
            self.R = tf.reshape(self.R, activations_shape)

        N, Hout, Wout, NF = self.R.get_shape().as_list()
        hf, wf, df, NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        # out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == "SAME":
            pr = (Hout - 1) * hstride + hf - in_h
            pc = (Wout - 1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr / 2
            p_bottom = pr - (pr / 2)
            p_left = pc / 2
            p_right = pc - (pc / 2)
            self.pad_input_tensor = tf.pad(
                self.input_tensor,
                [[0, 0], [p_top, p_bottom], [p_left, p_right], [0, 0]],
                "CONSTANT",
            )
        elif self.pad == "VALID":
            self.pad_input_tensor = self.input_tensor

        (
            pad_in_N,
            pad_in_h,
            pad_in_w,
            pad_in_depth,
        ) = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype=tf.float32)

        # pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0), 0
        )
        for i in range(Hout):
            for j in range(Wout):
                input_slice = self.pad_input_tensor[
                    :, i * hstride : i * hstride + hf, j * wstride : j * wstride + wf, :
                ]
                term2 = tf.expand_dims(input_slice, -1)
                # pdb.set_trace()
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1, 2, 3], keepdims=True)
                # Zs = t1 + t2
                Zs = t1
                stabilizer = 1e-8 * (
                    tf.where(
                        tf.greater_equal(Zs, 0),
                        tf.ones_like(Zs, dtype=tf.float32),
                        tf.ones_like(Zs, dtype=tf.float32) * -1,
                    )
                )
                Zs += stabilizer
                result = tf.reduce_sum(
                    (Z / Zs) * tf.expand_dims(self.R[:, i : i + 1, j : j + 1, :], 3), 4
                )

                # pdb.set_trace()
                # pad each result to the dimension of the out
                pad_bottom = (
                    pad_in_h - (i * hstride + hf)
                    if (pad_in_h - (i * hstride + hf)) > 0
                    else 0
                )
                pad_top = i * hstride
                pad_right = (
                    pad_in_w - (j * wstride + wf)
                    if (pad_in_w - (j * wstride + wf) > 0)
                    else 0
                )
                pad_left = j * wstride
                result = tf.pad(
                    result,
                    [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    "CONSTANT",
                )
                # print(i,j)
                # print(i*hstride, i*hstride+hf , j*wstride, j*wstride+wf)
                # print(pad_top, pad_bottom,pad_left, pad_right)
                Rx += result
        # pdb.set_trace()
        total_time = time.time() - start_time
        print(total_time)
        if self.pad == "SAME":
            return Rx[:, (pc / 2) : in_w + (pc / 2), (pr / 2) : in_h + (pr / 2), :]
        elif self.pad == "VALID":
            return Rx

    def __flat_lrp(self, R):
        """
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        """

        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape) != 4:
            self.R = tf.reshape(self.R, activations_shape)

        N, Hout, Wout, NF = self.R.get_shape().as_list()
        hf, wf, df, NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        # out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == "SAME":
            pr = (Hout - 1) * hstride + hf - in_h
            pc = (Wout - 1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr / 2
            p_bottom = pr - (pr / 2)
            p_left = pc / 2
            p_right = pc - (pc / 2)
            self.pad_input_tensor = tf.pad(
                self.input_tensor,
                [[0, 0], [p_top, p_bottom], [p_left, p_right], [0, 0]],
                "CONSTANT",
            )
        elif self.pad == "VALID":
            self.pad_input_tensor = self.input_tensor

        (
            pad_in_N,
            pad_in_h,
            pad_in_w,
            pad_in_depth,
        ) = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype=tf.float32)

        # pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0), 0
        )
        for i in range(Hout):
            for j in range(Wout):
                Z = tf.ones([N, hf, wf, df, NF], dtype=tf.float32)
                Zs = tf.reduce_sum(Z, [1, 2, 3], keepdims=True)

                result = tf.reduce_sum(
                    (Z / Zs) * tf.expand_dims(self.R[:, i : i + 1, j : j + 1, :], 3), 4
                )

                # pdb.set_trace()
                # pad each result to the dimension of the out
                pad_bottom = (
                    pad_in_h - (i * hstride + hf)
                    if (pad_in_h - (i * hstride + hf)) > 0
                    else 0
                )
                pad_top = i * hstride
                pad_right = (
                    pad_in_w - (j * wstride + wf)
                    if (pad_in_w - (j * wstride + wf) > 0)
                    else 0
                )
                pad_left = j * wstride
                result = tf.pad(
                    result,
                    [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    "CONSTANT",
                )
                # print(i,j)
                # print(i*hstride, i*hstride+hf , j*wstride, j*wstride+wf)
                # print(pad_top, pad_bottom,pad_left, pad_right)
                Rx += result
        # pdb.set_trace()
        if self.pad == "SAME":
            return Rx[:, (pc / 2) : in_w + (pc / 2), (pr / 2) : in_h + (pr / 2), :]
        elif self.pad == "VALID":
            return Rx

    def __ww_lrp(self, R):
        """
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        """

        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape) != 4:
            self.R = tf.reshape(self.R, activations_shape)

        N, Hout, Wout, NF = self.R.get_shape().as_list()
        hf, wf, df, NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        # out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == "SAME":
            pr = (Hout - 1) * hstride + hf - in_h
            pc = (Wout - 1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr / 2
            p_bottom = pr - (pr / 2)
            p_left = pc / 2
            p_right = pc - (pc / 2)
            self.pad_input_tensor = tf.pad(
                self.input_tensor,
                [[0, 0], [p_top, p_bottom], [p_left, p_right], [0, 0]],
                "CONSTANT",
            )
        elif self.pad == "VALID":
            self.pad_input_tensor = self.input_tensor

        (
            pad_in_N,
            pad_in_h,
            pad_in_w,
            pad_in_depth,
        ) = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype=tf.float32)

        # pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0), 0
        )
        for i in range(Hout):
            for j in range(Wout):
                Z = tf.square(tf.expand_dims(self.weights, 0))
                Zs = tf.reduce_sum(Z, [1, 2, 3], keepdims=True)

                result = tf.reduce_sum(
                    (Z / Zs) * tf.expand_dims(self.R[:, i : i + 1, j : j + 1, :], 3), 4
                )

                # pdb.set_trace()
                # pad each result to the dimension of the out
                pad_bottom = (
                    pad_in_h - (i * hstride + hf)
                    if (pad_in_h - (i * hstride + hf)) > 0
                    else 0
                )
                pad_top = i * hstride
                pad_right = (
                    pad_in_w - (j * wstride + wf)
                    if (pad_in_w - (j * wstride + wf) > 0)
                    else 0
                )
                pad_left = j * wstride
                result = tf.pad(
                    result,
                    [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                    "CONSTANT",
                )
                Rx += result
        # pdb.set_trace()
        if self.pad == "SAME":
            return Rx[:, (pc / 2) : in_w + (pc / 2), (pr / 2) : in_h + (pr / 2), :]
        elif self.pad == "VALID":
            return Rx

    def __epsilon_lrp(self, R, epsilon):
        """
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        """
        # pdb.set_trace()
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape) != 4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1] + activations_shape[1:])

        N, Hout, Wout, NF = self.R.get_shape().as_list()
        hf, wf, df, NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == "SAME":
            pr = (Hout - 1) * hstride + hf - in_h
            pc = (Wout - 1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(
                self.input_tensor,
                [[0, 0], [pr / 2, (pr - (pr / 2))], [pc / 2, (pc - (pc / 2))], [0, 0]],
                "CONSTANT",
            )
        elif self.pad == "VALID":
            self.pad_input_tensor = self.input_tensor

        (
            pad_in_N,
            pad_in_h,
            pad_in_w,
            pad_in_depth,
        ) = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype=tf.float32)

        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0), 0
        )
        for i in range(Hout):
            for j in range(Wout):
                input_slice = self.pad_input_tensor[
                    :, i * hstride : i * hstride + hf, j * wstride : j * wstride + wf, :
                ]
                term2 = tf.expand_dims(input_slice, -1)
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1, 2, 3], keepdims=True)
                Zs = t1 + t2
                stabilizer = epsilon * (
                    tf.where(
                        tf.greater_equal(Zs, 0),
                        tf.ones_like(Zs, dtype=tf.float32),
                        tf.ones_like(Zs, dtype=tf.float32) * -1,
                    )
                )
                Zs += stabilizer
                result = tf.reduce_sum(
                    (Z / Zs) * tf.expand_dims(self.R[:, i : i + 1, j : j + 1, :], 3), 4
                )

                # pad each result to the dimension of the out
                pad_right = (
                    pad_in_h - (i * hstride + hf)
                    if (pad_in_h - (i * hstride + hf)) > 0
                    else 0
                )
                pad_left = i * hstride
                pad_bottom = (
                    pad_in_w - (j * wstride + wf)
                    if (pad_in_w - (j * wstride + wf) > 0)
                    else 0
                )
                pad_up = j * wstride
                result = tf.pad(
                    result,
                    [[0, 0], [pad_left, pad_right], [pad_up, pad_bottom], [0, 0]],
                    "CONSTANT",
                )
                Rx += result

        if self.pad == "SAME":
            return Rx[:, (pc / 2) : in_w + (pc / 2), (pr / 2) : in_h + (pr / 2), :]
        elif self.pad == "VALID":
            return Rx

    def __alphabeta_lrp(self, R, alpha):
        """
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        """
        beta = 1 - alpha
        self.R = R
        R_shape = self.R.get_shape().as_list()

        if len(R_shape) != 4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1] + activations_shape[1:])

        N, Hout, Wout, NF = self.R.get_shape().as_list()
        hf, wf, df, NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == "SAME":
            pr = (Hout - 1) * hstride + hf - in_h
            pc = (Wout - 1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(
                self.input_tensor,
                [[0, 0], [pr / 2, (pr - (pr / 2))], [pc / 2, (pc - (pc / 2))], [0, 0]],
                "CONSTANT",
            )
        elif self.pad == "VALID":
            self.pad_input_tensor = self.input_tensor

        (
            pad_in_N,
            pad_in_h,
            pad_in_w,
            pad_in_depth,
        ) = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype=tf.float32)

        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in range(Hout):
            for j in range(Wout):
                input_slice = self.pad_input_tensor[
                    :, i * hstride : i * hstride + hf, j * wstride : j * wstride + wf, :
                ]
                term2 = tf.expand_dims(input_slice, -1)
                # pdb.set_trace()
                Z = term1 * term2

                if not alpha == 0:
                    Zp = tf.where(tf.greater(Z, 0), Z, tf.zeros_like(Z))
                    t2 = tf.expand_dims(
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
                    t1 = tf.expand_dims(tf.reduce_sum(Zp, 1), 1)
                    Zsp = t1 + t2
                    Ralpha = alpha * tf.reduce_sum(
                        (Zp / Zsp)
                        * tf.expand_dims(self.R[:, i : i + 1, j : j + 1, :], 3),
                        4,
                    )
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = tf.where(tf.less(Z, 0), Z, tf.zeros_like(Z))
                    t2 = tf.expand_dims(
                        tf.expand_dims(
                            tf.where(
                                tf.less(self.biases, 0),
                                self.biases,
                                tf.zeros_like(self.biases),
                            ),
                            0,
                        ),
                        0,
                    )
                    t1 = tf.expand_dims(tf.reduce_sum(Zn, 1), 1)
                    Zsp = t1 + t2
                    Rbeta = beta * tf.reduce_sum(
                        (Zn / Zsp)
                        * tf.expand_dims(self.R[:, i : i + 1, j : j + 1, :], 3),
                        4,
                    )
                else:
                    Rbeta = 0

                result = Ralpha + Rbeta

                # pad each result to the dimension of the out
                pad_right = (
                    pad_in_h - (i * hstride + hf)
                    if (pad_in_h - (i * hstride + hf)) > 0
                    else 0
                )
                pad_left = i * hstride
                pad_bottom = (
                    pad_in_w - (j * wstride + wf)
                    if (pad_in_w - (j * wstride + wf) > 0)
                    else 0
                )
                pad_up = j * wstride
                result = tf.pad(
                    result,
                    [[0, 0], [pad_left, pad_right], [pad_up, pad_bottom], [0, 0]],
                    "CONSTANT",
                )
                # print pad_bottom, pad_left, pad_right, pad_up
                # pdb.set_trace()
                Rx += result

        if self.pad == "SAME":
            return Rx[:, (pc / 2) : in_w + (pc / 2), (pr / 2) : in_h + (pr / 2), :]
        elif self.pad == "VALID":
            return Rx
