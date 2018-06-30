from builtins import range
from builtins import object
import numpy as np

from exercise_code.layers import *
from exercise_code.layer_utils import *


class TwoLayerNet(object):
    
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dim])
        self.params['b1'] = np.zeros([hidden_dim])
        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params['b2'] = np.zeros([num_classes])

    def loss(self, X, y=None):
        
        scores = None
        
        batch_size = X.shape[0]

        # Reshape input to vectors.
        flat_X = np.reshape(X, [batch_size, -1])

        # FC1
        fc1_act, fc1_cache = affine_forward(flat_X, self.params['W1'], self.params['b1'])
        # Relu1
        relu1_act, relu1_cache = relu_forward(fc1_act)
        # FC2
        fc2_act, fc2_cache = affine_forward(relu1_act, self.params['W2'], self.params['b2'])

        scores = np.copy(fc2_act)

        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dsoft = softmax_loss(scores, y)

        loss += 0.5*self.reg*(np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))

        dx2, dw2, db2 = affine_backward(dsoft, fc2_cache)
        drelu = relu_backward(dx2, relu1_cache)
        dx1, dw1, db1 = affine_backward(drelu, fc1_cache)

        grads['W2'], grads['b2'] = dw2 + self.reg*self.params['W2'], db2
        grads['W1'], grads['b1'] = dw1 + self.reg*self.params['W1'], db1

        return loss, grads


class FullyConnectedNet(object):
 
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
 
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i in range(self.num_layers - 1):
            self.params['W' + str(i+1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
            self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])

            if self.use_batchnorm:
                self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
                self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])

            input_dim = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.

        # Initialise the weights and biases for final FC layer
        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])


        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        

        fc_cache = {}
        relu_cache = {}
        bn_cache = {}
        dropout_cache = {}
        batch_size = X.shape[0]

        X = np.reshape(X, [batch_size, -1]) 
        for i in range(self.num_layers-1):

            fc_act, fc_cache[str(i+1)] = affine_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            if self.use_batchnorm:
                bn_act, bn_cache[str(i+1)] = batchnorm_forward(fc_act, self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                relu_act, relu_cache[str(i+1)] = relu_forward(bn_act)
            else:
                relu_act, relu_cache[str(i+1)] = relu_forward(fc_act)
            if self.use_dropout:
                relu_act, dropout_cache[str(i+1)] = dropout_forward(relu_act, self.dropout_param)

            X = relu_act.copy()  # Result of one pass through the affine-relu block.

        # Final output layer is FC layer with no relu.
        scores, final_cache = affine_forward(X, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])

        
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
        # Calculate score loss and add reg. loss for last FC layer.
        loss, dsoft = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(self.num_layers)])))

        # Backprop dsoft to the last FC layer to calculate gradients.
        dx_last, dw_last, db_last = affine_backward(dsoft, final_cache)

        # Store gradients of the last FC layer
        grads['W'+str(self.num_layers)] = dw_last + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_last

        # Iteratively backprop through each Relu & FC layer to calculate gradients.
        # Go through batchnorm and dropout layers if needed.
        for i in range(self.num_layers-1, 0, -1):

            if self.use_dropout:
                dx_last = dropout_backward(dx_last, dropout_cache[str(i)])

            drelu = relu_backward(dx_last, relu_cache[str(i)])

            if self.use_batchnorm:
                dbatchnorm, dgamma, dbeta = batchnorm_backward(drelu, bn_cache[str(i)])
                dx_last, dw_last, db_last = affine_backward(dbatchnorm, fc_cache[str(i)])
                grads['beta' + str(i)] = dbeta
                grads['gamma' + str(i)] = dgamma
            else:
                dx_last, dw_last, db_last = affine_backward(drelu, fc_cache[str(i)])

            # Store gradients.
            grads['W' + str(i)] = dw_last + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db_last

            # Add reg. loss for each other FC layer.
            loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(i)])))



        return loss, grads