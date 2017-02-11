import numpy as np
from dnncode.classifiers.layer_utils import *
from dnncode.classifiers.layers import softmax_loss


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    hidden_layer = self.activation(X.dot(W1) + b1)
    scores = hidden_layer.dot(W2) + b2

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # use underflow avoidance trick
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(correct_logprobs) / float(N)
    reg_loss = 0.5*reg*(np.sum(W2 * W2) + np.sum(W1 * W1))
    loss = data_loss + reg_loss

    # Backward pass: compute gradients
    grads = {}

    # compute gradients on scores
    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N

    # backpropagate gradient to the parameters
    # first backpropagate to W2, b2
    dW2 = np.dot(hidden_layer.T, dscores)
    #print "Hidden layer: ", hidden_layer.shape, "\n", hidden_layer
    #print "Dscores: ", dscores.shape, "\n", dscores
    #print "dW2: \n", dW2
    db2 = np.sum(dscores, axis=0, keepdims=True)
    db2 = np.reshape(db2, (db2.shape[1],))

    # next, backpropagate to hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0

    # finally into W,b
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)
    db1 = np.reshape(db1, (db1.shape[1],))
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      inds = np.random.choice(num_train, batch_size)
      X_batch = X[inds, :]
      y_batch = y[inds]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      # add regularization gradient contribution
      grads['W2'] += reg * self.params['W2']
      grads['W1'] += reg * self.params['W1']

      # perform a parameter update
      self.params['W1'] += -learning_rate * grads['W1']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b2'] += -learning_rate * grads['b2']


      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # Compute the forward pass
    hidden_layer = self.activation(X.dot(W1) + b1)
    scores = hidden_layer.dot(W2) + b2
    y_pred = np.argmax(scores, axis=1)

    return y_pred

  def activation(self, x):
    """

      :param x:
      :return:
    """
    return np.maximum(x, 0)


class TwoLayerNet2(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """
    # Step 1: forward pass and caches for two-layer net
    hidden_layer, hidden_cache = affine_relu_forward(x=X, w=self.params['W1'], b=self.params['b1'])
    scores, scores_cache = affine_forward(x=hidden_layer, w=self.params['W2'], b=self.params['b2'])

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    grads = {}

    # Step 2: compute loss
    data_loss, dscores = softmax_loss(x=scores, y=y)  # compute loss and gradient with respect to output
    reg_loss = 0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2']) +\
                                 np.sum(self.params['W1'] * self.params['W1']))
    loss = data_loss + reg_loss

    # Step 3: compute backpropagation
    dhidden, dW2, db2 = affine_backward(dscores, scores_cache)
    dx, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)

    dW1 += self.reg*self.params['W1']
    dW2 += self.reg*self.params['W2']

    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2

    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    curr_layer = 1
    curr_dim = input_dim
    while curr_layer < self.num_layers:
        wp_name = 'W' + str(curr_layer)
        bp_name = 'b' + str(curr_layer)
        next_dim = hidden_dims[curr_layer-1]
        self.params[wp_name] = weight_scale * np.random.randn(curr_dim, next_dim)
        self.params[bp_name] = np.zeros(next_dim)

        if self.use_batchnorm:
            self.params['gamma' + str(curr_layer)] = np.ones((next_dim,))
            self.params['beta' + str(curr_layer)] = np.zeros((next_dim,))

        curr_dim = next_dim
        curr_layer += 1
    # for the last layer, connect to num_classes
    wp_name = 'W' + str(curr_layer)
    bp_name = 'b' + str(curr_layer)
    self.params[wp_name] = weight_scale * np.random.randn(curr_dim, num_classes)
    self.params[bp_name] = np.zeros(num_classes)



    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    caches_data = {}
    caches_drop = {}
    caches_bn = {}

    # Step 1: forward pass and caches for multilayer net
    curr_layer = 1
    curr_data = X
    while curr_layer < self.num_layers:
        wp_name = 'W' + str(curr_layer)
        bp_name = 'b' + str(curr_layer)

        curr_data, curr_cache = affine_relu_forward(x=curr_data, w=self.params[wp_name],
                                                    b=self.params[bp_name])  # propagate forward
        caches_data[str(curr_layer)] = curr_cache

        if self.use_batchnorm:
            curr_data, curr_cache_bn = batchnorm_forward(x=curr_data, gamma=self.params['gamma' + str(curr_layer)],
                                                         beta=self.params['beta' + str(curr_layer)],
                                                         bn_param=self.bn_params[curr_layer-1])
            caches_bn[str(curr_layer)] = curr_cache_bn
        if self.use_dropout:
            curr_data, curr_cache_drop = dropout_forward(curr_data, self.dropout_param)  # dropout after nonlinearity
            caches_drop[str(curr_layer)] = curr_cache_drop
        curr_layer += 1
    scores, scores_cache = affine_forward(x=curr_data, w=self.params['W' + str(curr_layer)],
                                          b=self.params['b' + str(curr_layer)])

    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    # step 2: compute loss
    data_loss, dscores = softmax_loss(x=scores, y=y)  # compute loss and gradient with respect to output
    curr_layer = 1
    reg_loss = 0.0
    while curr_layer < self.num_layers:
        wp_name = 'W' + str(curr_layer)
        reg_loss += np.sum(self.params[wp_name]*self.params[wp_name])
        curr_layer += 1
    reg_loss *= 0.5*self.reg
    loss = data_loss + reg_loss

    # step 3: compute backpropagation
    curr_layer = self.num_layers
    dval = []

    while curr_layer > 0:
        wp_name = 'W' + str(curr_layer)
        bp_name = 'b' + str(curr_layer)
        if curr_layer == self.num_layers:
            dval, dWcurr, dbcurr = affine_backward(dscores, scores_cache)  # get last layer's gradients
        else:
            if self.use_dropout:
                dval = dropout_backward(dval, caches_drop[str(curr_layer)])
            if self.use_batchnorm:
                dval, dgamma, dbeta = batchnorm_backward_alt(dval, caches_bn[str(curr_layer)])
                grads['gamma' + str(curr_layer)] = dgamma
                grads['beta' + str(curr_layer)] = dbeta
            dval, dWcurr, dbcurr = affine_relu_backward(dval, caches_data[str(curr_layer)])  # get previous layer's gradients
        grads[wp_name] = dWcurr + self.reg*self.params[wp_name]
        grads[bp_name] = dbcurr + self.reg*self.params[bp_name]
        curr_layer -= 1

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

