# No additional 3rd party external libraries are allowed
import numpy as np

def linear(num_inputs, num_outputs):
    # Perceptron: y = f(Wx - b)
    # Initialize variables
    W = np.zeros((num_outputs, num_inputs), dtype="f")
    b = np.zeros((num_outputs, 1), dtype = "f")

    # TODO
    raise NotImplementedError("Linear function not implemented")

def linear_grad(num_inputs, num_outputs):
    # Perceptron: y = f(Wx - b)
    # Initialize variables
    W = np.zeros((num_outputs, num_inputs), dtype="f")
    b = np.zeros((num_outputs, 1), dtype = "f")

    # TODO
    raise NotImplementedError("Gradient of Linear function not implemented")

def radial(num_inputs, num_outputs):
    # Perceptron: y = f(Wx - b)
    # Initialize variables
    W = np.zeros((num_outputs, num_inputs), dtype="f")
    b = np.zeros((num_outputs, 1), dtype = "f")

    # TODO
    raise NotImplementedError("Radial Basis function not implemented")

def radial_grad(num_inputs, num_outputs):
    # Perceptron: y = f(Wx - b)
    # Initialize variables
    W = np.zeros((num_outputs, num_inputs), dtype="f")
    b = np.zeros((num_outputs, 1), dtype = "f")

    # TODO
    raise NotImplementedError("Gradient of Radial Basis function not implemented")