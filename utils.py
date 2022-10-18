import torch
import tensorflow as tf
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import numpy as np


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def restore_checkpoint_extended(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['optimizer_spectral'].load_state_dict(loaded_state['optimizer_spectral'])
    state['model_spectral'].load_state_dict(loaded_state['model_spectral'], strict=False)
    state['ema_spectral'].load_state_dict(loaded_state['ema_spectral'])
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)

def save_checkpoint_extended(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'optimizer_spectral': state['optimizer_spectral'].state_dict(),
    'model_spectral': state['model_spectral'].state_dict(),
    'ema_spectral': state['ema_spectral'].state_dict(),
  }
  torch.save(saved_state, ckpt_dir)


def rot(U,V):
  # this converts U to V
  W=np.cross(U,V)
  A=np.array([U,W,np.cross(U,W)]).T
  B=np.array([V,W,np.cross(V,W)]).T
  return np.matmul(B,np.linalg.inv(A))

def convert_to_2D(x):
  # we have a vector u in the simplex, x+y+z = 1
  # we first tranfer it to the origin by x= x-1/3, y = y-1/3,.. 
  x = x-np.array(1/3)
  # then that will satisfy x+y+z = 0
  u = np.array([1,1,1])
  v = np.array([0,0,1])
  R = rot(u/np.linalg.norm(u),v) # pass unit vector u
  y = np.matmul(R,x.T)
  y = y.T
  return y

def data_plot_proj_2D(codes, viz_dir, idx=0):
  codes = codes[:,:3]
  codes = convert_to_2D(codes)
  fig =plt.figure()
  ax = fig.add_subplot()
  ax.scatter(codes[:,0], codes[:,1])

  lines = np.eye(3)
  lines = convert_to_2D(lines)
  ax.plot(lines[:,0], lines[:,1], 'r', linestyle='-')

  # plt.xlim([-1,1])
  # plt.ylim([-1,1])
  plt.savefig(viz_dir+'/scatter'+str(idx))
  plt.close()

## numerical integration code adapted from <script src="https://gist.github.com/chausies/c453d561310317e7eda598e229aea537.js"></script>
class Numerical_integration():
  def __init__(self, type = 'Hermite'):
    super().__init__()
  
  
  def h_poly_helper(self, tt):
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
    return [
      sum( A[i, j]*tt[j] for j in range(4) )
      for i in range(4) ]

  def h_poly(self, t):
    tt = [ None for _ in range(4) ]
    tt[0] = 1
    for i in range(1, 4):
      tt[i] = tt[i-1]*t
    return self.h_poly_helper(tt)

  def H_poly(self, t):
    tt = [ None for _ in range(4) ]
    tt[0] = t
    for i in range(1, 4):
      tt[i] = tt[i-1]*t*i/(i+1)
    return self.h_poly_helper(tt)

  def interp_func(self, x, y):
    "Returns integral of interpolating function"
    if len(y)>1:
      m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
      m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    def f(xs):
      if len(y)==1: # in the case of 1 point, treat as constant function
        return y[0] + T.zeros_like(xs)
      I = T.searchsorted(x[1:], xs)
      dx = (x[I+1]-x[I])
      hh = self.h_poly((xs-x[I])/dx)
      return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
    return f

  def interp(self, x, y, xs):
    return self.interp_func(x,y)(xs)

  def integ_func(self, x, y):
    "Returns interpolating function"
    if len(y)>1:
      m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
      m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
      Y = T.zeros_like(y)
      Y[1:] = (x[1:]-x[:-1])*(
          (y[:-1]+y[1:])/2 + (m[:-1] - m[1:])*(x[1:]-x[:-1])/12
          )
      Y = Y.cumsum(0)

    def f(xs):
      if len(y)==1:
        return y[0]*(xs - x[0])
      I = T.searchsorted(x[1:].detach(), xs)
      dx = (x[I+1]-x[I])
      hh = self.H_poly((xs-x[I])/dx)
      return Y[I] + dx*(
          hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
          )
    return f

  def integ(self, x, y, xs):
    return self.integ_func(x,y)(xs)

  def integrate(self, t, y, ts):
    sum = 0
    N = y.shape[0]
    for i in range(y.shape[0]):
      integ= self.integ_func(t[i], y[i])(ts)
      sum = sum + integ[-1]/N
    return sum