from layers import *
from metrics import *


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        tf_v1 = tf.compat.v1
        with tf_v1.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf_v1.get_collection(tf_v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, output_dim=None, num_supports=1, hidden_dim=16,
                 act=None, dropout=False, sparse_inputs=False, featureless=False, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim or placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.num_supports = num_supports
        self.hidden_dim = hidden_dim
        self.act = act or tf.nn.relu
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            weight_decay = 5e-4  # Default weight decay
            if hasattr(self, '_weight_decay'):
                weight_decay = self._weight_decay
            self.loss += weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.hidden_dim,
                                 placeholders=self.placeholders,
                                 act=self.act,
                                 dropout=self.dropout,
                                 sparse_inputs=self.sparse_inputs,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.hidden_dim,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, output_dim=None, num_supports=1, hidden_dim=16, 
                 act=None, dropout=False, sparse_inputs=False, featureless=False, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim or placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.num_supports = num_supports
        self.hidden_dim = hidden_dim
        self.act = act or tf.nn.relu
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            weight_decay = 5e-4  # Default weight decay
            if hasattr(self, '_weight_decay'):
                weight_decay = self._weight_decay
            self.loss += weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.hidden_dim,
                                            placeholders=self.placeholders,
                                            act=self.act,
                                            dropout=self.dropout,
                                            sparse_inputs=self.sparse_inputs,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.hidden_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=self.dropout,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
