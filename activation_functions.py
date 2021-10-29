import math

def linear(x):
    '''Linear activation function: https://en.wikipedia.org/wiki/Identity_function
    '''
    return x


def tanh(x):
    '''Hyperbolic tangent activation function: https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent
    This returns values between -1 and 1
    '''
    return math.tanh(x)


def sigmoid(x):
    '''Classical sigmoid activation function: https://en.wikipedia.org/wiki/Sigmoid_function
    This 
    '''
    return 1.0 / (1 + math.exp(-x))


def relu(x):
    '''Rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    '''
    return max(0, x)


def leaky_relu(x):
    '''Leaky rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU
    Can be used to avoid the problem of dying neurons
    '''
    return p_relu(x, 0.01)


def p_relu(x, p=0.01):
    '''Parametric rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Parametric_ReLU
    '''
    if x >= 0:
        return x
    else:
        return p * x


def softplus(x):
    '''Softplus activation function: https://en.wikipedia.org/wiki/Activation_function
    '''
    return math.log(1 + math.exp(x))

#Binary step function : https://en.wikipedia.org/wiki/Step_function
def binary_step(x):
    if x<0:
        return 0
    else:
        return 1

#Swish activation function :https://en.wikipedia.org/wiki/Swish_function
def swish(x,b=1.0):
    return x/(1+math.exp(-b*x))


def elu(x, a=1.0):
    '''Exponential linear units(ELU) activation function: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#ELU'''
    if x > 0:
        return x
    else:
        return a * (math.exp(x) - 1)


def silu(x):
    '''Sigmoid Linear Unit(SiLU) activation function: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#SiLU'''
    return x * sigmoid(x)


def mish(x):
    '''Mish Activation Function : https://arxiv.org/abs/1908.08681'''
    return x * tanh(softplus(x))

def bent_identity(x):
    '''Bent-Identity Activation Function: https://www.gabormelli.com/RKB/Bent_Identity_Activation_Function'''
    return (math.sqrt(((x**2)+1)-1)/2)+x


def gelu(x):
    '''GELU Activation Function : https://paperswithcode.com/method/gelu'''
    cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return x * cdf

def arctan(x):
    '''Arc Tan Activation Function - http://theurbanengine.com/blog//arctan'''
    return math.atan(x)

def lecuns_tanh(x):
    '''Le Cun's tanh ACtivation Function - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf'''
    return 1.7159 * math.tanh(2/3 * x)


def bipolar_sigmoid(x):
    '''Bipolar Sigmoid - https://aip.scitation.org/doi/pdf/10.1063/1.4954526#:~:text=Bipolar%20sigmoid%20is%20a%20type,it%20is%20a%20bounded%20function.'''
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


def logit(x):
    '''https://deepai.org/machine-learning-glossary-and-terms/logit#:~:text=The%20Logit%20function%20is%20used,between%20negative%20and%20positive%20infinity.'''
    try:
        return math.log(x / (1 - x))
    except ValueError:
        print('ignoring math domain error..')