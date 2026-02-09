import numpy as np
import tensorflow as tf

# TensorFlow 2.x/Keras 3 compatibility
# The original code was written for TensorFlow 1.x with tf.layers.conv1d
# Modern TensorFlow/Keras versions have moved this API
tf.compat.v1.disable_eager_execution()

try:
    # Try TensorFlow 1.x API
    conv1d = tf.layers.conv1d
except AttributeError:
    try:
        # Try TensorFlow 2.x compat.v1 API
        conv1d = tf.compat.v1.layers.conv1d
    except (AttributeError, RuntimeError):
        # Fallback: implement conv1d using dense layers for Keras 3
        def conv1d(inputs, filters, kernel_size, use_bias=True, activation=None, name=None):
            """Fallback 1D convolution using dense layer."""
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            in_channels = inputs.shape[-1]
            
            # Reshape for dense layer
            inputs_flat = tf.reshape(inputs, [-1, in_channels])
            # Dense layer acts as 1x1 conv
            dense = tf.keras.layers.Dense(filters, use_bias=use_bias, activation=activation, name=name)
            output_flat = dense(inputs_flat)
            # Reshape back
            output = tf.reshape(output_flat, [batch_size, seq_len, filters])
            return output

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = conv1d(seq_fts, 1, 1)
        f_2 = conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        
        # TensorFlow 2.x compatibility for bias_add
        try:
            ret = tf.contrib.layers.bias_add(vals)
        except (AttributeError, RuntimeError):
            # TensorFlow 2.x - use compat.v1
            ret = tf.compat.v1.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = conv1d(seq_fts, 1, 1)
        f_2 = conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        try:
            ret = tf.contrib.layers.bias_add(vals)
        except (AttributeError, RuntimeError):
            ret = tf.compat.v1.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

