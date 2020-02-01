import numpy as np
from collections import defaultdict

class Graph(object):
    """Represents Tensorflow computational graph"""

    def __init__(self):
        self._ops = []
        self._constants = []
        self._placeholders = []
        self._vars = []

    def as_default(self):
        # make global to easily use graph instance when adding to it
        global _default_graph
        _default_graph = self

    def add_op(self, op):
        self._ops.append(op)

    def add_constant(self, constant):
        self._constants.append(constant)

    def add_placeholder(self, placeholder):
        self._placeholders.append(placeholder)

    def add_var(self, var):
        self._vars.append(var)

########################################################################

class Op(object):
    """Represents a tf operation to be added in the computational graph"""

    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.output = None
        self.grads = [] # gradients w.r.t inputs
        _default_graph.add_op(self) # add this operation to the graph

    def forward(self):
        pass

    def backward(self, grad_out):
        pass

class BinaryOp(Op):
    """Represents binary operations"""
    def __init__(self, a, b):
        super().__init__([a, b])

class add(BinaryOp):
    """Computes a + b element-wise"""

    def forward(self, a, b):
        return a + b

    def backward(self, grad_out):
        pass

class multiply(BinaryOp):
    """Computes a * b element-wise"""

    def forward(self, a, b):
        return a * b

    def backward(self, grad_out):
        pass

class divide(BinaryOp):
    """Computes a / b element-wise"""

    def forward(self, a, b):
        return a / b

    def backward(self, grad_out):
        pass

########################################################################

class constant:

    def __init__(self, value):
        assert value is not None, 'Constant value can not be None'
        self.__value = value
        _default_graph.add_constant(self)

    @property
    def value(self):
        return self.__value

class placeholder:
    def __init__(self):
        self.value = None # set later by the Session
        _default_graph.add_placeholder(self)

class Variable:
    def __init__(self, initial_value=None):
        self.value = initial_value
        _default_graph.add_var(self)

########################################################################

class Session(object):

    @staticmethod
    def topological_sort(op):
        """
        Topologically sort the computational graph to make sure that
        dependencies are computed first
        """

        ordering = []
        visited = set()
        def dfs(node):
            if isinstance(node, Op):
                for input_node in node.input_nodes:
                    if input_node not in visited:
                        dfs(input_node)
            visited.add(node)
            ordering.append(node)
        dfs(op)
        return ordering

    def run(self, op, feed_dict={}):
        # op is a chain of the above defined op classes
        sorted_nodes = self.topological_sort(op)
        for node in sorted_nodes:
            if type(node) is placeholder:
                node.output = feed_dict[node]
            elif type(node) is Variable or type(node) is constant:
                node.output = node.value
            else:
                # apply operation
                # collect all the outputs of the input nodes
                inputs = [node.output for node in node.input_nodes]
                # apply forward pass
                node.output = node.forward(*inputs)
        return op.output

########################################################################

if __name__ == '__main__':
    # create default graph
    Graph().as_default()

    # construct computational graph by creating some nodes
    a = Constant(15)
    b = Constant(5)
    prod = multiply(a, b)
    sum = add(a, b)
    res = divide(prod, sum)

    # create a session object
    session = Session()

    # run computational graph to compute the output for 'res'
    out = session.run(res)
    print(out)
