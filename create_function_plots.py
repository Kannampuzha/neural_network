# Necessery imports
import os
import matplotlib.pyplot as plt
import numpy as np
from activation_functions import *


def get_act_function(x,acts):
  for act in acts:
    cur_act = []
    for i in x:
      cur_act.append(globals()[act](i))
    plot_acts(x,cur_act,act)
  return cur_act

def plot_acts(x,y,act):
  
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),constrained_layout = True)
  axes[0].set_xlabel('y=x')
  axes[0].plot(x, x)
  axes[1].set_xlabel('y=f(x)')
  axes[1].plot(x, y, color = 'red')
  fig.suptitle(act)
  filename = os.path.join('plots',act+'.png')
  fig.savefig(filename)

def create_dir(path):
    os.makedirs(path,exist_ok=True)

def main():
    acts = [ 'linear', 'tanh', 'sigmoid','relu', 'leaky_relu', 'p_relu', 'softplus', 'binary_step', 'swish', 'elu','silu', 'mish', 'bent_identity','gelu','arctan','lecuns_tanh','bipolar_sigmoid','logit'] 
    path = 'plots'
    create_dir(path)
    x = np.linspace(-10, 10, 100) 
    y = get_act_function(x,acts)

if __name__ == '__main__':
    main()