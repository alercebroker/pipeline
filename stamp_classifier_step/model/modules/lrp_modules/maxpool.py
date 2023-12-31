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
import numpy as np


from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


class MaxPool(Module):
    def __init__(self, pool_size=2, pool_stride=None, pad="SAME", name="maxpool"):
        self.name = name
        Module.__init__(self)
        self.pool_size = pool_size
        self.pool_kernel = [1, self.pool_size, self.pool_size, 1]
        self.pool_stride = pool_stride
        if self.pool_stride is None:
            self.stride_size = self.pool_size
        else:
            self.stride_size = self.pool_stride
        self.pool_stride = [1, self.stride_size, self.stride_size, 1]
        self.pad = pad

    def forward(self, input_tensor, batch_size=10, img_dim=28):
        self.input_tensor = input_tensor
        (
            self.in_N,
            self.in_h,
            self.in_w,
            self.in_depth,
        ) = self.input_tensor.get_shape().as_list()
        self.in_N = tf.shape(self.input_tensor)[0]

        # with tf.variable_scope(self.name):
        with tf.name_scope(self.name):
            self.activations = tf.nn.max_pool(
                self.input_tensor,
                ksize=self.pool_kernel,
                strides=self.pool_stride,
                padding=self.pad,
                name=self.name,
            )
            # tf.summary.histogram('activations', self.activations)

        return self.activations

    def clean(self):
        self.activations = None
        self.R = None

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
        Zs = self.compute_zs(Z)
        result = self.compute_result(Z, Zs)
        return self.restitch_image(result)

    def _epsilon_lrp(self, R, epsilon):
        """
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        """
        return self._simple_lrp(R)

    def _epsilon_lrpPy(self, sess, feed_dict, R, epsilon=1e-12):
        """
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        """
        R = self.check_shapePy(R)

        inp_shape = self.input_tensor.get_shape().as_list()
        # bad practice
        name = next(iter(feed_dict))
        N = feed_dict.get(name).shape[0]
        H = inp_shape[1]
        W = inp_shape[2]
        D = inp_shape[3]
        X = sess.run(self.input_tensor, feed_dict=feed_dict)
        Y = sess.run(self.activations, feed_dict=feed_dict)

        hpool = self.pool_size
        wpool = self.pool_size
        hstride = self.pool_stride[1]
        wstride = self.pool_stride[2]

        # assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        Rx = np.zeros_like(X, dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = (
                    Y[:, i : i + 1, j : j + 1, :]
                    == X[
                        :,
                        i * hstride : i * hstride + hpool,
                        j * wstride : j * wstride + wpool,
                        :,
                    ]
                )
                Zs = Z.sum(
                    axis=(1, 2), keepdims=True, dtype=np.float
                )  # thanks user wodtko for reporting this bug/fix
                Rx[
                    :,
                    i * hstride : i * hstride + hpool,
                    j * wstride : j * wstride + wpool,
                    :,
                ] += (Z / Zs) * R[:, i : i + 1, j : j + 1, :]
        return Rx

    def _ww_lrp(self, R):
        """
        There are no weights to use. default to _flat_lrp(R)
        """
        return self._flat_lrp(R)

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
                self.pool_size,
                self.pool_size,
                self.in_depth,
            ]
        )
        Zs = self.compute_zs(Z)
        result = self.compute_result(Z, Zs)
        return self.restitch_image(result)

    def _alphabeta_lrp(self, R, alpha):
        """
        Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
        """
        return self._simple_lrp(R)

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
            ksizes=[1, self.pool_size, self.pool_size, 1],
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
                self.pool_size,
                self.pool_size,
                self.in_depth,
            ],
        )

    def compute_z(self, image_patches):
        Z = tf.equal(
            tf.reshape(
                self.activations, [self.in_N, self.Hout, self.Wout, 1, 1, self.in_depth]
            ),
            image_patches,
        )
        return tf.where(
            Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z, dtype=tf.float32)
        )

    # [QUESO] stabilizer y reduce
    def compute_zs(self, Z, stabilizer=False, epsilon=1e-12):
        Zs = tf.reduce_sum(Z, [3, 4], keepdims=True)  # + tf.expand_dims(self.biases, 0)
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
        # import pdb; pdb.set_trace()
        # result = (Z/Zs) * tf.reshape(self.R, [self.in_N,self.Hout,self.Wout,1,1,self.in_depth])
        # [QUESO]
        result = Z * tf.reshape(
            self.R, [self.in_N, self.Hout, self.Wout, 1, 1, self.in_depth]
        )

        return tf.reshape(
            result,
            [
                self.in_N,
                self.Hout,
                self.Wout,
                self.pool_size * self.pool_size * self.in_depth,
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
            self.pool_size,
            self.pool_size,
            self.stride_size,
            self.stride_size,
        )

    # def _simple__lrp(self,R):
    #     '''
    #     LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
    #     '''
    #     import time; start_time = time.time()

    #     self.R = R
    #     R_shape = self.R.get_shape().as_list()
    #     activations_shape = self.activations.get_shape().as_list()
    #     if len(R_shape)!=4:
    #         self.R = tf.reshape(self.R, activations_shape)

    #     N,Hout,Wout,NF = self.R.get_shape().as_list()
    #     _, hf,wf,_ = self.pool_size
    #     _, hstride, wstride, _ = self.pool_stride
    #     in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

    #     # op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
    #     # p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
    #     # image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])

    #     # op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
    #     # p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
    #     image_patches = tf.reshape(tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad), [N,Hout,Wout, hf, wf, in_depth])

    #     #import pdb; pdb.set_trace()
    #     Z = tf.equal( tf.reshape(self.activations, [N,Hout,Wout,1,1,NF]), image_patches)
    #     Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
    #     #Z = tf.expand_dims(self.weights, 0) * tf.expand_dims( image_patches, -1)
    #     Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
    #     stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
    #     Zs += stabilizer
    #     #result =   (Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,NF])
    #     total_time = time.time() - start_time
    #     print(total_time)
    #     return self.patches_to_images(tf.reshape( (Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,NF]), [N,Hout,Wout, hf*wf*in_depth]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )

    #     #return Rx

    # def __simple_lrp(self,R):
    #     '''
    #     LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
    #     '''
    #     import time; start_time = time.time()

    #     self.R = R
    #     R_shape = self.R.get_shape().as_list()
    #     if len(R_shape)!=4:
    #         activations_shape = self.activations.get_shape().as_list()
    #         self.R = tf.reshape(self.R, activations_shape)

    #     N,Hout,Wout,NF = self.R.get_shape().as_list()
    #     _,hf,wf,_ = self.pool_size
    #     _,hstride, wstride,_= self.pool_stride

    #     out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
    #     in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()

    #     if self.pad == 'SAME':
    #         pr = (Hout -1) * hstride + hf - in_rows
    #         pc =  (Wout -1) * wstride + wf - in_cols
    #         #similar to TF pad operation
    #         self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
    #     elif self.pad == 'VALID':
    #         self.pad_input_tensor = self.input_tensor

    #     pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()

    #     Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
    #     for i in xrange(Hout):
    #         for j in xrange(Wout):
    #             input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
    #             Z = tf.equal( self.activations[:,i:i+1, j:j+1,:], input_slice)
    #             Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
    #             Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
    #             result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
    #             #pad each result to the dimension of the out
    #             pad_right = pad_in_rows - (i*hstride+hf) if( pad_in_rows - (i*hstride+hf))>0 else 0
    #             pad_left = i*hstride
    #             pad_bottom = pad_in_cols - (j*wstride+wf) if ( pad_in_cols - (j*wstride+wf) > 0) else 0
    #             pad_up = j*wstride
    #             result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")

    #             Rx+= result
    #     total_time = time.time() - start_time
    #     print(total_time)

    #     if self.pad=='SAME':
    #         return Rx[:, (pc/2):in_cols+(pc/2), (pr/2):in_rows+(pr/2), :]
    #     elif self.pad =='VALID':
    #         return Rx

    # def _flat_lrp(self,R):
    #     '''
    #     distribute relevance for each output evenly to the output neurons' receptive fields.
    #     '''

    #     self.R = R
    #     R_shape = self.R.get_shape().as_list()
    #     if len(R_shape)!=4:
    #         activations_shape = self.activations.get_shape().as_list()
    #         self.R = tf.reshape(self.R, activations_shape)

    #     N,Hout,Wout,NF = self.R.get_shape().as_list()
    #     _,hf,wf,_ = self.pool_size
    #     _,hstride, wstride,_= self.pool_stride

    #     out_N, out_rows, out_cols, out_depth = self.activations.get_shape().as_list()
    #     in_N, in_rows, in_cols, in_depth = self.input_tensor.get_shape().as_list()

    #     if self.pad == 'SAME':
    #         pr = (Hout -1) * hstride + hf - in_rows
    #         pc =  (Wout -1) * wstride + wf - in_cols
    #         #similar to TF pad operation
    #         self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
    #     elif self.pad == 'VALID':
    #         self.pad_input_tensor = self.input_tensor

    #     pad_in_N, pad_in_rows, pad_in_cols, pad_in_depth = self.pad_input_tensor.get_shape().as_list()

    #     Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
    #     for i in xrange(Hout):
    #         for j in xrange(Wout):
    #             Z = tf.ones([N, hf,wf,NF], dtype=tf.float32)
    #             Zs = tf.reduce_sum(Z, [1,2], keep_dims=True)
    #             result = (Z/Zs) * self.R[:,i:i+1,j:j+1,:]
    #             #pad each result to the dimension of the out
    #             pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
    #             pad_top = i*hstride
    #             pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
    #             pad_left = j*wstride
    #             result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")

    #             Rx+= result
    #     if self.pad=='SAME':
    #         return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
    #     elif self.pad =='VALID':
    #         return Rx

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
            if rows_out * 2 != rows_in:
                rows_out = int(ceil((rows_in + 1) / stride_r))
                cols_out = int(ceil((cols_in + 1) / stride_h))
            else:
                rows_out = int(ceil(rows_in / stride_r))
                cols_out = int(ceil(cols_in / stride_h))
            pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
            pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

        elif padding == "VALID":
            if rows_out * 2 != rows_in:
                rows_out = int(ceil(((rows_in + 1) - ksize_r_eff + 1) / stride_r))
                cols_out = int(ceil(((cols_in + 1) - ksize_c_eff + 1) / stride_h))
            else:
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
