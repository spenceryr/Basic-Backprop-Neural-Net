import numpy as np
import pickle
import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [784, 47, 47, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 500  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 3  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  #assuming x = weighted sum of the inputs from the hidden to output layer
  #print(x)
  return np.divide(np.exp(x).T, np.exp(x).sum(axis=1)).T


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  one_hot = [0,0,0,0,0,0,0,0,0,0]
  images = []
  labels = []
  data = pickle.load(open(fname, "rb"))
  for col in data:
    images.append(list(col[:-1]))
    labels.append(one_hot.copy())
    labels[-1][int(col[-1])] = 1
  return np.array(images), np.array(labels)


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.ReLU(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = 1/(1+np.exp(-x, dtype=np.float64))
    return output

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = np.tanh(self.x)
    return output

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    max_v = np.vectorize(lambda a: max(0,a))
    output = max_v(x)
    return output

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    def _sig(a):
      return 1/(1+np.exp(-a, dtype=np.float64))
    grad = _sig(self.x) * (1-_sig(self.x))
    return grad

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    grad = 1-(np.power(np.tanh(self.x), 2))
    return grad

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    grad_v = np.vectorize(lambda a: 1 if a > 0 else 0)
    return grad_v(self.x)


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.w *= (1/(in_units + out_units))
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    self.a = np.concatenate((np.ones((self.x.shape[0], 1)), self.x), axis=1).dot(np.concatenate((self.b, self.w)))
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    self.d_b = np.ones((1, self.x.shape[0])).dot(delta)
    self.d_w = self.x.T.dot(delta)
    self.d_x = delta.dot(self.w.T)
    return self.d_x


class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    self.targets = targets
    loss = None
    for layer in self.layers:
      self.x = layer.forward_pass(self.x)
    self.y = softmax(self.x)
    if self.targets is not None:
      loss = self.loss_func(self.y, targets)
    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    return -np.sum(targets * np.log(logits + .0001), axis=1)/(10)
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    gradients = []
    bias_gradients = []
    delta = (self.targets - self.y)
    for layer in self.layers[::-1]:
      delta = layer.backward_pass(delta)
      if type(layer) is Layer:
        gradients.append(layer.d_w)
        bias_gradients.append(layer.d_b)
    return gradients[::-1], bias_gradients[::-1]


def is_correct(y, t):
  return np.equal(np.argmax(y, axis=1), np.argmax(t, axis=1))


def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  regularization_func = lambda x: x * config["L2_penalty"]


  #config["learning_rate"] *= (1/150)

  train_accuracies = []
  valid_accuracies = []
  best_weights = []
  early_stop_counter = 0
  training_loss = []
  validation_loss = []
  final_epoch = 0

  for epoch in range(config["epochs"]):
    final_epoch = epoch
    print("---------------")
    print(epoch)
    print("---")
    training_loss.append([])
    train_accuracies.append([])


    if config["momentum"]:
      momentums = [0 for layer in model.layers if type(layer) is Layer]
      momentums_bias = [0 for layer in model.layers if type(layer) is Layer]
    gradients = [0 for layer in model.layers if type(layer) is Layer]
    bias_gradients = [0 for layer in model.layers if type(layer) is Layer]
    for n in range(len(X_train)//config["batch_size"]):
      #print(str(n) + "/" + str(len(X_train)//config["batch_size"]))
      X_batch = X_train[config["batch_size"]*n:config["batch_size"]*(n+1)]
      y_batch = y_train[config["batch_size"]*n:config["batch_size"]*(n+1)]
      loss, y = model.forward_pass(X_batch, targets=y_batch)
      training_loss[-1].append(loss)
      train_accuracies[-1].append(is_correct(y, y_batch))
      gradients, bias_gradients = model.backward_pass()
      for layer_num, layer in enumerate([layer for layer in model.layers if type(layer) is Layer]):
        if config["momentum"]:
          momentums[layer_num] = (-config["momentum_gamma"] * momentums[layer_num]) + gradients[layer_num]
          momentums_bias[layer_num] = (-config["momentum_gamma"] * momentums_bias[layer_num]) + bias_gradients[layer_num]
          layer.w += (config["learning_rate"] * momentums[layer_num]) + regularization_func(layer.w)
          layer.b += (config["learning_rate"] * momentums_bias[layer_num]) + regularization_func(layer.b)
        else:
          layer.w += (config["learning_rate"] * gradients[layer_num]) + regularization_func(layer.w)
          layer.b += (config["learning_rate"] * bias_gradients[layer_num]) + regularization_func(layer.b)
    training_loss[-1] = (np.mean(training_loss[-1]))
    print("train loss: " + str(training_loss[-1]))
    train_accuracies[-1] = (np.mean(train_accuracies[-1])) * 100
    print("train accuracy: " + str(train_accuracies[-1]))
    loss, y = model.forward_pass(X_valid, targets=y_valid)
    validation_loss.append(np.mean(loss))
    print("validation loss: " + str(validation_loss[-1]))
    valid_accuracies.append(np.mean(is_correct(y, y_valid)) * 100)
    print("validation accuracy: " + str(valid_accuracies[-1]))
    if config["early_stop"]:
      if len(validation_loss) > 1 and validation_loss[epoch] > validation_loss[epoch-1]:
        early_stop_counter += 1
      else:
        if len(validation_loss) < 1 or (validation_loss[epoch] < min(validation_loss)):
          best_weights = [layer.w for layer in model.layers if type(layer) is Layer]
        early_stop_counter = 0
      if early_stop_counter >= config["early_stop_epoch"]:
        break
  if config["early_stop"]:
    for layer, weights in zip([l for l in model.layers if type(l) is Layer], best_weights):
      layer.w = weights

  fig, graph = plt.subplots(nrows=1, ncols=1, figsize=(7,6), sharex=True)
  fig.suptitle("Average Validation and Training Accuracies", y=1)
  fig.tight_layout()
  fig.subplots_adjust(top=.85, wspace=.3)
  graph.set_ylabel("Accuracy")
  graph.set_xlabel("# of Epochs")
  graph.plot(list(range(final_epoch + 1)), valid_accuracies, linestyle='-', color='red', marker='o', label="validation")
  graph.plot(list(range(final_epoch + 1)), train_accuracies, linestyle='-', color='blue', marker='o', label="train")
  graph.legend()
  plt.savefig("accuracies_" + config["activation"] + "_" + ("early_" if config["early_stop"] else "notearly_") + ("mom" if config["momentum"] else "notmom") + ".png", bbox_inches='tight')

  fig, graph = plt.subplots(nrows=1, ncols=1, figsize=(7,6), sharex=True)
  fig.suptitle("Average Validation and Training Loss", y=1)
  fig.tight_layout()
  fig.subplots_adjust(top=.85, wspace=.3)
  graph.set_ylabel("Loss")
  graph.set_xlabel("# of Epochs")
  graph.plot(list(range(final_epoch + 1)), validation_loss, linestyle='-', color='red', marker='o', label="validation")
  graph.plot(list(range(final_epoch + 1)), training_loss, linestyle='-', color='blue', marker='o', label="train")
  graph.legend()
  plt.savefig("loss_" + config["activation"] + "_" + ("early_" if config["early_stop"] else "notearly_") + ("mom" if config["momentum"] else "notmom") + ".png", bbox_inches='tight')


def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  totalCorrect = 0
  _, y = model.forward_pass(X_test)
  accuracy = np.mean(is_correct(y, y_test)) * 100
  print(accuracy)
  return accuracy
      

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)
