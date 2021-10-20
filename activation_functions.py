import math


# Classical sigmoid activation function: https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


# Rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
def relu(x):
    return max(0, x)


# Leaky rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU
# Can be used to avoid the problem of dying neurons
def leaky_relu(x):
    return p_relu(x, 0.01)


# Parametric rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Parametric_ReLU
def p_relu(x, p):
    if x >= 0:
        return x
    else:
        return p * x


# Softplus activation function: https://en.wikipedia.org/wiki/Activation_function
def softplus(x):
    return math.log(1 + math.exp(x))

#Binary step function : https://en.wikipedia.org/wiki/Step_function
def Binary_step(x):
    if x<0:
        return 0
    else:
        return 1

#Swish activation function :https://en.wikipedia.org/wiki/Swish_function
def swish(x,b):
    return x/(1+math.exp(-b*x))