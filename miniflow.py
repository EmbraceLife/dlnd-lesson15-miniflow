
import numpy as np

class Node:
    """
    Base class for nodes in the network.

    Arguments:

        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        """
        Node's constructor (runs when the object is instantiated). Sets
        properties that all nodes need.
        """
        # A list of nodes (the inbound_nodes=[] above) with edges are taken as inbound_nodes into this node.
        self.inbound_nodes = inbound_nodes
        
        
        # The eventual value of this node. Set by running the forward() method.
        # when initialize this node, or right now, set value to be None
        self.value = None
        
        
        # initialize A list of nodes that this node outputs to
        # set them to be empty list 
        self.outbound_nodes = []
        
        
        # New property! Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        
        
        # Sets this node (we are creating at this moment) as an outbound node for all of
        # this node's inputs.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

            
    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


        
        
class Input(Node):
    """
    A generic input into the network.
    """
    def __init__(self):
        # The base class constructor has to run to set all
        # the properties here.
        #
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        
        # first of all, make sure Input node is a Node class
        Node.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object. Yes, it is right
        self.gradients = {self: 0}
        
        
        # Weights and bias may be inputs, so you need to sum 
        # the gradient from output gradients.                      ######### yes, i see ##########
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self] ######## explained by computational derivation of graph ?

class Linear(Node):
    """
    Represents a node that performs a linear transform.
    """
    def __init__(self, X, W, b):
        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            # for l_2 node, grad_cost is MSE's gradient
            grad_cost = n.gradients[self]
            
            # Set the partial of the loss with respect to this node's inputs: ################################
            # therefore, dL2/dX * grad_cost = sum_(W) * grad_cost = sum_(W*grad_cost) 
            # therefore,                    = dot(MSE's gradient, inbound_weights.T)
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # why grad_cost first, and weight.T second? 
            # this way = we can get dim (1, num_weights_or_inputs)
            
            
            # Set the partial of the loss with respect to this node's weights.
            # therefore, dL2/dW * grad_cost = sum_(X) * grad_cost = sum_(X*grad_cost) 
            # therefore,                    = dot(inbound_X.T, MSE's gradient)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # why X.T first, grad_cost second? 
            # this way = we can get dim (num_weights, 1)
            

            # Set the partial of the loss with respect to this node's bias.
            # dl_2/db_2 * grad_cost = sum(1) * grad_cost = sum(grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

            # 1. take derivative of sigmoid with respect to l1 or first linear combination  ##################
            # 2. add up is like computional graph for adding up to different routes of gradients ################
            # multiple grade_cost = one of sigmoid's outbound_nodes' gradient, it is to inherit gradient from above #
            # https://hyp.is/yPFpUPADEea5ebOmVQr3pw/colah.github.io/posts/2015-08-Backprop/
            # 3. there is one gradient accumulation below, as there is only one input to sigmoid node ##############
            # 4. because there is no summation in the gradient, this is not dot product, just simple multiplicatin
            
            

class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        # The base class constructor.
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        
        
        # Initialize the gradients to 0.  Yes, I agree
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        
        
        # Sum the partial with respect to the input over all the outputs.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            
            # 1. take derivative of sigmoid with respect to l1 or first linear combination  ##################
            # 2. add up is like computional graph for adding up to different routes of gradients ################
            # multiple grade_cost = one of sigmoid's outbound_nodes' gradient, it is to inherit gradient from above #
            # https://hyp.is/yPFpUPADEea5ebOmVQr3pw/colah.github.io/posts/2015-08-Backprop/
            # 3. there is one gradient accumulation below, as there is only one input to sigmoid node ##############
            # 4. because there is no summation in the gradient, this is not dot product, just simple multiplicatin
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        """ ####################################
        The mean squared error cost function.
        Should be used as the last node for a network.
        
        Everything is a node: 
        There are input nodes, linear combination node, activation function node, loss function node, 
        X, y, W, b are inputs nodes too
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.

        # convert dim(3,) to dim(3,1), so 
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1) 
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        # set m to be num of data points
        self.m = self.inbound_nodes[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)

        
    def backward(self):
        """
        Calculates the gradient of the cost with respect to the second linear combination
    
        """
        # take derivative of MSE with respect to y
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff # for y ##########################
        
        # take derivative of MSE with respect to a or y_hat
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff # for a or y_hat ##################


# Now, we have methods or functions, instead of classes        
def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


# forward and backward pass function 
def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # graph contains all nodes (inputs, Linear, sigmoid, MSE) flattened and ordered
    # let each node do forward and backward calculation
    
    # Forward pass
    # start from the first node and do forward() on it
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    # start from the last node to the first node, and do backward() on it
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # Performs SGD
    #
    # Loop over the trainables [W1, b1, W2, b2]
    for t in trainables:
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        
        # calc gradient of weights or biases only
        partial = t.gradients[t]
        
        # update weights or biases, it indicates partial derivative of MSE with respect to W2, W1, b1 and b2
        # in order words, gradient of W1 will inherit gradient cost all the way of the chain from W1 to Cost
        t.value -= learning_rate * partial