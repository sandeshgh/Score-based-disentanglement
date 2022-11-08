# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

from cv2 import mean
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from torchdiffeq import odeint_adjoint as odeint #odeint_adjoint as odeint
import torchsde
import math
from utils import Numerical_integration as NI

def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def score_divergence(score1, score2):
  score1 = score1.view(score1.shape[0], -1)
  norm1 = torch.sqrt(torch.mean(score1**2, dim =-1))
  score2 = score2.view(score2.shape[0], -1)
  norm2 = torch.sqrt(torch.mean(score2**2, dim =-1))
  divergence = (score1/norm1.unsqueeze(1) - score2/norm2.unsqueeze(1))**2
  out = divergence.mean()
  return out

def mse_loss(x, y):
  mse = torch.mean((x-y)**2)
  return mse

class ReverseDrift_ode(nn.Module):
    def __init__(self, sde, model, latent, train = True, continuous = True):
      super().__init__()
      self.model = model
      self.sde = sde
      self.latent =  latent
      self.train = train
      self.continuous = continuous
    
    def drift_fn(self, x, t):
      # # score_fn = get_score_fn(self.sde, self.model, train=False, continuous=True)
      # rsde = self.sde.reverse(self.score_fn, probability_flow=True)
      # return rsde.sde(x, t, latent)[0]
      # drift, diffusion = self.sde.sde(x, t)
      # if variational:
      #   latent, kl_loss = self.model.module.encode(x_in)
      # else:
      #   latent = self.model.module.encode(x_in)

      score_fn = mutils.get_score_fn_neural_ode(self.model, train=self.train, continuous=self.continuous, generate=True)

     
      score = score_fn(x, t, self.latent)
      drift = score
                
      # drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5)
      # Set the diffusion function to zero for ODEs.
      # diffusion = 0. 
      return drift
    
    def forward(self, t, x):
      vec_t = torch.ones(x.shape[0], device=x.device) * t
      drift = self.drift_fn(x, vec_t)
      return drift

class Reverse_sde(nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'scalar'
    def __init__(self, model, latent, beta_min, beta_max, shape, train = True, continuous = True):
      super().__init__()
      self.model = model
      # self.sde = sde
      self.latent =  latent
      self.train = train
      self.continuous = continuous
      self.beta_0 = beta_min
      self.beta_1 = beta_max
      self.shape = shape
    
    def f(self, t, x):
      x = x.view(self.shape)
      vec_t = torch.ones(x.shape[0], device=x.device) * t
      score_fn = mutils.get_score_fn_neural_ode(self.model, train=self.train, continuous=self.continuous, generate=True)     
      score = score_fn(x, vec_t, self.latent)
      drift = score.view(self.shape[0],-1)
      return drift
    
    def g(self, t, x):
      t= 1-t
      beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
      diffusion = torch.sqrt(beta_t)
      diffusion = torch.ones(self.shape, device=diffusion.device)*diffusion
      g_ = diffusion.view(self.shape[0],-1).unsqueeze(-1)
      return g_
    
    # def forward(self, t, x):
    #   vec_t = torch.ones(x.shape[0], device=x.device) * t
    #   drift = self.drift_fn(x, vec_t)
    #   return drift

class Reverse_sde_w_diffusion(nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'scalar'
    def __init__(self, model, latent, beta_min, beta_max, shape, train = True, continuous = True):
      super().__init__()
      self.model = model
      # self.sde = sde
      self.latent =  latent
      self.train = train
      self.continuous = continuous
      self.beta_0 = beta_min
      self.beta_1 = beta_max
      self.shape = shape
    
    def get_drift_diffusion(self, t, x):
      t = 1-t
      beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
      drift = -0.5 * beta_t[None, None, None, None] * x
      diffusion = torch.sqrt(beta_t)
      return drift, diffusion
    
    def get_diffusion(self, t, x):
      t= 1-t
      beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
      diffusion = torch.sqrt(beta_t)
      return diffusion
    
    def f(self, t, x):
      x = x.view(self.shape)
      vec_t = torch.ones(x.shape[0], device=x.device) * t
      score_fn = mutils.get_score_fn_neural_ode(self.model, train=self.train, continuous=self.continuous, generate=True)     
      score = score_fn(x, vec_t, self.latent)
      d, diff = self.get_drift_diffusion(t,x)
      drift = (score*diff[None, None, None, None]**2 - d).view(self.shape[0],-1)
      return drift
    
    def g(self, t, x):
      # t= 1-t
      # beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
      diffusion = self.get_diffusion(t, x)
      diffusion = torch.ones(self.shape, device=diffusion.device)*diffusion
      g_ = diffusion.view(self.shape[0],-1).unsqueeze(-1)
      return g_

def peek_ode(x_in, model, sde, eps=1e-3, variational=False, train = True, continuous = True):
  shape = x_in.shape
  device = x_in.device

  # class ReverseDrift(nn.Module):
  #   def __init__(self, sde, score_fn, latent):
  #     super().__init__()
  #     self.score_fn = score_fn
  #     self.latent = latent
  #     self.sde = sde
    
  #   def drift_fn(self, x, t, latent):
  #     # # score_fn = get_score_fn(self.sde, self.model, train=False, continuous=True)
  #     # rsde = self.sde.reverse(self.score_fn, probability_flow=True)
  #     # return rsde.sde(x, t, latent)[0]
  #     drift, diffusion = self.sde.sde(x, t)
  #     if latent is not None:
  #       score = self.score_fn(x, t, latent)
  #     else:
  #       score = self.score_fn(x, t)
          
  #     drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5)
  #     # Set the diffusion function to zero for ODEs.
  #     # diffusion = 0. 
  #     return drift
    
  #   def forward(self, t, x):
  #     vec_t = torch.ones(shape[0], device=x.device) * t
  #     drift = self.drift_fn(x, vec_t, self.latent)
  #     return drift
  
  class ReverseDrift(nn.Module):
    def __init__(self, sde, model):
      super().__init__()
      self.model = model
      self.sde = sde
    
    def drift_fn(self, x, t):
      # # score_fn = get_score_fn(self.sde, self.model, train=False, continuous=True)
      # rsde = self.sde.reverse(self.score_fn, probability_flow=True)
      # return rsde.sde(x, t, latent)[0]
      drift, diffusion = self.sde.sde(x, t)
      if variational:
        latent, kl_loss = self.model.module.encode(x_in)
      else:
        latent = self.model.module.encode(x_in)

      score_fn = mutils.get_latent_score_fn(self.sde, self.model, train=train, continuous=continuous, generate=True)

     
      score = score_fn(x, t, latent)
                
      drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5)
      # Set the diffusion function to zero for ODEs.
      # diffusion = 0. 
      return drift
    
    def forward(self, t, x):
      vec_t = torch.ones(shape[0], device=x.device) * t
      drift = self.drift_fn(x, vec_t)
      return drift
  
  

    # def ode_func(t, x):
    #   x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
    #   vec_t = torch.ones(shape[0], device=x.device) * t
    #   drift = drift_fn(model, x, vec_t)
    #   return to_flattened_numpy(drift)


  # score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
  ode_func = ReverseDrift(sde, model)

  adj_params = find_parameters(ode_func)
  
  
  # sample the latent code from the prior distibution of the SDE.
  x = sde.prior_sampling(shape).to(device)

  # Black-box ODE solver for the probability flow ODE
  # {Sandesh Comments: note that the time goes in the reverse direction from T to 0(eps). The reason is that ODE equation is the forward equation going from good image to noise
  # Here, since we want to sample from the good distribution p(x), we need to go in the reverse of the forward ODE equation, i.e. from noise to good samples, hence the reverse time 
  # passed to ode solver
  # Don't be confused by the fact that drift_fn is calling rsde. The ODE is still forward. It's just that forward ODE resembles reverse SDE with a factor of 0.5
  # Hence, the authors calls rsde with probability_flow =True which takes care of factor 0.5 and gives a forward ODE equations. :D  Sandesh comment ends}
  sol = odeint(ode_func, x, t=torch.linspace(sde.T, eps, 2).to(x.device), method = 'rk4', rtol=1e-5)
  # nfe = solution.nfev
  # x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
  # sol = inverse_scaler(sol)
  return sol[-1]


def compute_ortho_loss(s1, s2):
  prod = s1*s2
  prod = prod.view(prod.shape[0], -1)
  prod = prod.mean(dim=0)
  loss = (prod**2).mean()
  return loss

def compute_variation_loss(alpha):
  #alpha is B*K 
  var = torch.var(alpha, dim=0, unbiased=False)
  neg_mean_var = -torch.mean(var)
  return neg_mean_var



def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_equal_energy_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=0.5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """

    score_fn = mutils.get_equal_energy_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score, score2 = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      # losses = torch.square((score +score2)* std[:, None, None, None] + 2*z) + gamma*torch.square(score* std[:, None, None, None] + z)
      losses = torch.square(score2* std[:, None, None, None] + z) + torch.square(score* std[:, None, None, None] + z)
      loss_ortho = torch.square((score* std[:, None, None, None] + z)*(score2* std[:, None, None, None] + z))
      # loss_ortho = reduce_op(loss_ortho.reshape(loss_ortho.shape[0], loss_ortho.shape[1], -1), dim=-1)
      losses = losses+gamma*loss_ortho
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      # losses = torch.square(score + score2+ 2*z / std[:, None, None, None]) + gamma*torch.square(score + z / std[:, None, None, None])
      losses = torch.square(score + z / std[:, None, None, None]) + torch.square(score2 + z / std[:, None, None, None])
      loss_ortho = torch.square((score + z / std[:, None, None, None])*(score2 + z / std[:, None, None, None]))
      losses = losses+gamma*loss_ortho
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    # loss = torch.mean(losses) - gamma*loss_div
    loss = torch.mean(losses) 
    return loss

  return loss_fn

def get_latent_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 0.1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    if reconstruction_loss:
      # (x_in, model, sde, eps=1e-3, variational=False, train = True, continuous = True)
      peek_generated = peek_ode(batch, model, sde, train = train, continuous=continuous)
      rec_loss = mse_loss(peek_generated, batch)
      loss = loss_score + gamma*rec_loss
      return loss, rec_loss, loss_score
    else:
      return loss_score
    

  return loss_fn

def get_latent_loss_fn_w_unconditional(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, ortho_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous, variational=ortho_loss)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    if ortho_loss:
      score, orthogonal_loss = score_fn(perturbed_data, t, batch)
    else:
      score = score_fn(perturbed_data, t, batch)
    # if train:
    #   model.train()
    # else:
    #   model.eval()
    # orthogonal_loss = model.module.orthogonal_loss_fn()
    # loss_div = score_divergence(score, score2)
    

    # if not likelihood_weighting:
    losses = torch.square(score * std[:, None, None, None] + z)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss_score = torch.mean(losses) 

    score_fn_u = mutils.get_score_fn_forward_unconditional(sde, model, train=train, continuous=continuous)    
    z_u = torch.randn_like(batch)
    perturbed_data_u = mean + std[:, None, None, None] * z_u
    score_u = score_fn_u(perturbed_data_u, t)

    losses_u = torch.square(score_u * std[:, None, None, None] + z_u)
    losses_u = reduce_op(losses_u.reshape(losses_u.shape[0], -1), dim=-1)
    loss_score_u = torch.mean(losses_u) 
    if ortho_loss:
      orthogonal_loss = orthogonal_loss.mean()
      loss = loss_score + loss_score_u + orthogonal_loss
      return loss, orthogonal_loss, loss_score
    else:
      loss = loss_score + loss_score_u
      return loss, loss_score_u, loss_score


    # else:
    #   g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
    #   losses = torch.square(score + z / std[:, None, None, None])
    #   losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    
    
    
    
    

  return loss_fn

def get_latent_loss_fn_multiscore(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 0.1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, batch)
    score = score.mean(-1)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    if reconstruction_loss:
      # (x_in, model, sde, eps=1e-3, variational=False, train = True, continuous = True)
      peek_generated = peek_ode(batch, model, sde, train = train, continuous=continuous)
      rec_loss = mse_loss(peek_generated, batch)
      loss = loss_score + gamma*rec_loss
      return loss, rec_loss, loss_score
    else:
      return loss_score
    

  return loss_fn


def get_latent_loss_fn_spectral(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 0.1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    zeta = spectral_model(batch)
    u = torch.fft.fft2(batch)
    mean, std = sde.marginal_prob(u, t, zeta )
    perturbed_spectral = mean + std[:, None, None, None] * z
    perturbed_data = torch.fft.ifft2(perturbed_spectral)
    score = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score + torch.fft.ifft2(std[:, None, None, None].pow(-1)*z))
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    # else:
    #   g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
    #   losses = torch.square(score + z / std[:, None, None, None])
    #   losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    
    return loss_score
    

  return loss_fn

def get_latent_loss_fn_predict(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)

    latent_z, latent_x = spectral_model(batch)
    
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    perturbed_data = torch.cat((perturbed_data, latent_x), dim =1)
    
    score = score_fn(perturbed_data, t, latent_z.detach())
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    rec_loss = torch.mean((batch - latent_x)**2)
    # C = compute_cov(latent_z.permute(1,0))
    # epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
    # kl_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)
    loss = loss_score + beta*(rec_loss)
    
    return loss, rec_loss , loss_score
    

  return loss_fn


def get_loss_fn_learned_diffusion(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1

    sde.update_model(spectral_model)

    score_fn = mutils.get_score_fn_learned_diffusion(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)

    # latent_z, latent_x = spectral_model(batch)
    
    mean = sde.generate_trajectory(batch, t)
    perturbed_data = mean + eps * z
    std = sde.get_diffusion(t)
    # perturbed_data = torch.cat((perturbed_data, latent_x), dim =1)
    
    score = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    rec_loss = torch.mean((batch - latent_x)**2)
    # C = compute_cov(latent_z.permute(1,0))
    # epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
    # kl_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)
    loss = loss_score + beta*(rec_loss)
    
    return loss, rec_loss , loss_score
    

  return loss_fn


def get_latent_loss_fn_fkac(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False, t_batch = 16, loss_type = None):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1
    if loss_type == 'simple':
      score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
      t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
      z = torch.randn_like(batch)

      latent_z, latent_x = spectral_model(batch)
      
      mean, std = sde.marginal_prob(batch, t)
      perturbed_data = mean + std[:, None, None, None] * z
      # perturbed_data = torch.cat((perturbed_data, latent_x), dim =1)
      
      score = score_fn(perturbed_data, t, latent_z)
      # loss_div = score_divergence(score, score2)
      

      if not likelihood_weighting:
        losses = torch.square(score * std[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
      else:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

      loss_score = torch.mean(losses) 
      # rec_loss = torch.mean((batch - latent_x)**2)
      # C = compute_cov(latent_z.permute(1,0))
      # epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
      # kl_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)
      loss = loss_score 
    else:

      b_s, c_s, w_s, h_s = batch.shape

      score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
      latent_z, latent_x = spectral_model(batch)
      #b_s , t_batch
      t = torch.rand((batch.shape[0], t_batch), device=batch.device) * (sde.T - eps) + eps
      # t_eps = eps*0.999*torch.ones((batch.shape[0], 1), device = batch.device)
      # t = torch.cat((t_eps,t), dim=1)
      batch = batch.unsqueeze(1).repeat(1,t_batch,1,1,1)
      latent_z = latent_z.unsqueeze(1).repeat(1, t_batch, 1)
      z = torch.randn_like(batch)

      # collapse b_s, t_batch
      t = t.view(b_s*t_batch)
      batch = batch.view(b_s*t_batch, c_s, w_s, h_s)
      z = z.view(b_s*t_batch, c_s, w_s, h_s)
      latent_z = latent_z.view(b_s*t_batch, -1)


      #    
      mean, std = sde.marginal_prob(batch, t)
      perturbed_data = mean + std[:, None, None, None] * z
      # perturbed_data = torch.cat((perturbed_data, latent_x), dim =1)
      
      score = score_fn(perturbed_data, t, latent_z)
      # loss_div = score_divergence(score, score2)
      

      if not likelihood_weighting:
        losses = torch.square(score * std[:, None, None, None] + z)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        
        if loss_type == 'path_integration':
          
          t = t.view(b_s, t_batch)
          losses = losses.view(b_s, t_batch)
          t_sorted, sort_indices = torch.sort(t)
          tmax =(torch.max(t_sorted,-1)[0]).min()
          t_arr = torch.linspace(eps, tmax, 100, device=t.device)
          losses = torch.gather(losses, -1, sort_indices)
          losses = NI().integrate(t_sorted, losses, t_arr)
        else:
          losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
          losses = losses.view(b_s, t_batch).mean(-1)
        # rearrange loss back to batch form
      
      else:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

      
      loss_score = torch.mean(losses) 
      # rec_loss = torch.mean((batch - latent_x)**2)
      # C = compute_cov(latent_z.permute(1,0))
      # epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
      # kl_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)
      loss = loss_score 
    
    return loss, torch.zeros_like(loss) , loss_score
    

  return loss_fn

def get_latent_loss_fn_predict_multi(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1

    score_fn = mutils.get_latent_score_fn_predict_multi(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)

    latent_z, latent_x = spectral_model(batch)
    
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    perturbed_data = torch.cat((perturbed_data[:,:,:,:,None].repeat(1,1,1,1,2), latent_x), dim =1)
    
    
    scores = score_fn(perturbed_data, t, latent_z.detach())
    score = scores.mean(-1)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    latent_x_mean = latent_x.mean(-1)
    rec_loss = torch.mean((batch - latent_x_mean)**2)
    # C = compute_cov(latent_z.permute(1,0))
    # epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
    # kl_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)
    loss = loss_score + beta*(rec_loss)
    
    return loss, rec_loss , loss_score
    

  return loss_fn


def get_latent_loss_fn_neural_ode(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  # reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    # beta = 0.1

    # score_fn = mutils.get_latent_score_fn_neural_ode(sde, model, train=train, continuous=continuous)
    # t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    # z = torch.randn_like(batch)
    x = sde.prior_sampling(batch.shape).to(batch.device)

    latent_z, latent_x = spectral_model(batch)
    ode_func = ReverseDrift_ode(sde, model, latent_z, train = train, continuous = continuous)

    sol = odeint(ode_func, x, t=torch.linspace(sde.T, eps, 2).to(x.device), method = 'rk4', rtol=1e-5)
    pred = sol[-1]
    
    
    rec_loss = torch.mean((batch - pred )**2)
    
    loss = rec_loss
    
    return loss, rec_loss , torch.zeros_like(loss)
    

  return loss_fn

def get_latent_loss_fn_neural_sde(sde, train, continuous=True, eps=1e-5, gamma=1e-2, ):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  # reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
  
    x = sde.prior_sampling(batch.shape).to(batch.device)

    latent_z, latent_x = spectral_model(batch)
    x_shape = x.shape
    sde_func = Reverse_sde_w_diffusion(model, latent_z, beta_min=sde.beta_0, beta_max=sde.beta_1, shape = x_shape )



    pred = torchsde.sdeint_adjoint(sde_func, x.view(x.shape[0],-1), ts=torch.linspace(0, sde.T - eps, 2).to(x.device), method = 'reversible_heun', adjoint_method='adjoint_reversible_heun', dt=0.1)
    pred = pred[-1].view(x_shape)
    
    
    rec_loss = torch.mean((batch - pred )**2)
    
    loss = rec_loss
    
    return loss, rec_loss , torch.zeros_like(loss)
    

  return loss_fn

def compute_cov(x):
  mu = x.mean(1)
  z = x-mu.unsqueeze(-1)
  C = torch.mm(z, z.permute(1,0))
  C = C/x.shape[1]
  return C

def get_latent_loss_fn_anisotropic(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1

    score_fn = mutils.get_latent_score_fn_spectral(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    latent_z, latent_x = spectral_model(batch)
    # u = torch.fft.fft2(batch)
    mean, std = sde.marginal_prob(batch, t, latent_x)
    perturbed_data = mean + std * z
    # perturbed_data = torch.fft.ifft2(perturbed_spectral)
    score = score_fn(perturbed_data, t, latent_z, latent_x)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std.detach() + z)
      # losses = torch.square(score + (std[:, None, None, None].pow(-1)*z))
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    # else:
    #   g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
    #   losses = torch.square(score + z / std[:, None, None, None])
    #   losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    C = compute_cov(latent_z.permute(1,0))
    epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
    kl_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)
    
    return loss_score + beta*kl_loss, kl_loss, loss_score
    

  return loss_fn

def get_latent_loss_fn_highvar(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(losses) 
    latent_z = model.module.encode(batch)
    # var_loss = -torch.var(lat,dim=0).mean()

    C = compute_cov(latent_z.permute(1,0))
    epsl =1e-6*torch.eye(C.shape[0]).to(batch.device)
    var_loss = -torch.trace(C)+1e-3*torch.logdet(C+epsl)

    loss = loss_score + beta * var_loss
   
    return loss, var_loss, loss_score
   
    

  return loss_fn

def get_latent_contrastive_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-5, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 0.1

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)

    m = score.shape[0]
    # assert m%2==0
    m_2 = int(math.ceil(m/2))
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses_1, losses_2 = losses[:m_2], losses[m_2:]
      loss_1 = reduce_op(losses_1.reshape(losses_1.shape[0], -1), dim=-1)
      loss_2 = reduce_op(losses_2.reshape(losses_2.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses_1, losses_2 = losses[:m_2], losses[m_2:]
      loss_1 = reduce_op(losses_1.reshape(losses_1.shape[0], -1), dim=-1) * g2
      loss_2 = reduce_op(losses_2.reshape(losses_2.shape[0], -1), dim=-1) * g2

    loss_score = torch.mean(loss_1) - beta * torch.clamp(torch.mean(loss_2), max = 10)
    
    return loss_score, -torch.mean(loss_2), torch.mean(loss_1)
    

  return loss_fn

def get_variational_latent_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=1e-2, reconstruction_loss=False):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 1e-5

    score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous, variational = True)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score, kl_loss = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses) + beta*kl_loss.mean()
    if reconstruction_loss:
      score_fn_gen = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous, generate=True)
      peek_generated = peek_ode(batch, model, sde, score_fn_gen, variational=True)
      rec_loss = mse_loss(peek_generated, batch)
      loss = loss + gamma*rec_loss
    return loss

  return loss_fn

def get_loss_fn_latent_factor(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=0.5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 0.1

    score_fn = mutils.get_score_fn_latent_factor(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score1, score2 = score_fn(perturbed_data, t, batch)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square((score1+score2) * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
      loss_ortho = compute_ortho_loss(score1, score2)
      # loss_ortho = compute_ortho_loss((score1* std[:, None, None, None] + z), (score2* std[:, None, None, None] + z))
      # loss_ortho = torch.mean((score1* std[:, None, None, None] + z)*(score2* std[:, None, None, None] + z))
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square((score1+score2) + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
      loss_ortho = compute_ortho_loss(score1, score2 )
      # loss_ortho = compute_ortho_loss((score1 + z / std[:, None, None, None]), (score2 + z / std[:, None, None, None]))
      # loss_ortho = torch.mean((score1 + z / std[:, None, None, None])*(score2 + z / std[:, None, None, None]))

    loss = torch.mean(losses) + beta*loss_ortho
    return loss

  return loss_fn

def get_loss_fn_basis(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=0.5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    beta = 0.1

    score_fn = mutils.get_score_fn_basis(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    scores, alpha = score_fn(perturbed_data, t, batch)
    loss_variation = compute_variation_loss(alpha)
    alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    wtd_score = torch.mean(scores*alpha, dim = -1)
    # loss_div = score_divergence(score, score2)
    

    if not likelihood_weighting:
      losses = torch.square(wtd_score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
      # loss_ortho = compute_ortho_loss((score1* std[:, None, None, None] + z), (score2* std[:, None, None, None] + z))
      # loss_ortho = torch.mean((score1* std[:, None, None, None] + z)*(score2* std[:, None, None, None] + z))
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(wtd_score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
      # loss_ortho = compute_ortho_loss((score1 + z / std[:, None, None, None]), (score2 + z / std[:, None, None, None]))
      # loss_ortho = torch.mean((score1 + z / std[:, None, None, None])*(score2 + z / std[:, None, None, None]))

    loss = torch.mean(losses) +beta*loss_variation
    return loss

  return loss_fn

def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_loss_fn_label(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch, class_label):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn_label(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t, class_label)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn



def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn




def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_ddpm_loss_fn_predict_multi(vpsde, train, reduce_mean=True, continuous = False,):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, spectral_model, batch):
    # sde, model, train=False, continuous=False, generate = False, variational=False
    beta = 0.01
    model_fn = mutils.get_model_fn_label(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise

    latent_z, latent_x = spectral_model(batch)
    
    # mean, std = sde.marginal_prob(batch, t)
    # perturbed_data = mean + std[:, None, None, None] * z
    perturbed_data_2 = perturbed_data[:,:,:,:,None].repeat(1,1,1,1,2)
    
    
    scores = model_fn(perturbed_data_2, labels, latent_z)
    score = scores.mean(-1)

    # score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss_score = torch.mean(losses)

    predicted_x_0 = (perturbed_data.detach() - sqrt_1m_alphas_cumprod[labels, None, None, None]*score)/sqrt_alphas_cumprod[labels, None, None, None]
    loss_predicted = torch.square(predicted_x_0 - batch).mean()
    loss = loss_score+beta*loss_predicted
    return loss, loss_predicted, loss_score

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, config = None):
  """Create a one-step training/evaluation function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
  Returns:
    A one-step function for training or evaluation.
  """
  if config.training.conditional_model == 'label':
    loss_fn = get_loss_fn_label(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif config.training.conditional_model == 'basis':
    loss_fn = get_loss_fn_basis(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif config.training.conditional_model == 'equal_energy':
    loss_fn = get_equal_energy_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif config.training.conditional_model == 'latent_contrastive':
    loss_fn = get_latent_contrastive_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  
  elif config.training.conditional_model == 'latent':
    if config.training.regularization == 'highvar':
      loss_fn = get_latent_loss_fn_highvar(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
    elif config.training.reconstruction_loss:
      loss_fn = get_latent_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting, reconstruction_loss=True)
    else:
      loss_fn = get_latent_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif config.training.conditional_model == 'latent_variational':
    assert config.model.name == 'ddpm_latent_variational', "The model must be variational to use variational method"
    if config.training.reconstruction_loss:
      loss_fn = get_variational_latent_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting, reconstruction_loss=True)
    else:
      loss_fn = get_variational_latent_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif config.training.conditional_model == 'latent_factor':
    loss_fn = get_loss_fn_latent_factor(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif config.training.conditional_model == 'latent_multi':
    if config.training.unconditional_loss:
      loss_fn = get_latent_loss_fn_w_unconditional(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting, ortho_loss = config.training.ortho_loss)
    else:
      loss_fn = get_latent_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  elif continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch, class_label=None):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      if class_label is not None:
        loss = loss_fn(model, batch, class_label)
      else:
        loss = loss_fn(model, batch)
      if (config.training.conditional_model == 'latent' and config.training.reconstruction_loss) or (config.training.conditional_model == 'latent_contrastive'):
        loss[0].backward()
      elif (config.training.conditional_model == 'latent' and config.training.regularization == 'highvar'):
        loss[0].backward()
      elif config.training.conditional_model == 'latent_multi' and config.training.unconditional_loss:
        loss[0].backward()
      else:
        loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        if class_label is not None:
          loss = loss_fn(model, batch, class_label)
        else:
          loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss
  
  def step_fn_spectral(state, batch, class_label=None):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    model_spectral = state['model_spectral']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      if class_label is not None:
        loss = loss_fn(model, model_spectral, batch, class_label)
      else:
        loss = loss_fn(model, model_spectral, batch)
      # if (config.training.conditional_model == 'latent' and config.training.reconstruction_loss) or (config.training.conditional_model == 'latent_contrastive'):
      #   loss[0].backward()
      # elif (config.training.conditional_model == 'latent' and config.training.regularization == 'highvar'):
      #   loss[0].backward()
      # else:
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        if class_label is not None:
          loss = loss_fn(model, batch, class_label)
        else:
          loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn



def get_step_fn_spectral(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, config = None):
  """Create a one-step training/evaluation function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
  Returns:
    A one-step function for training or evaluation.
  """
  # if config.training.conditional_model == 'label':
  #   loss_fn = get_loss_fn_label(sde, train, reduce_mean=reduce_mean,
  #                             continuous=True, likelihood_weighting=likelihood_weighting)
  # elif config.training.conditional_model == 'basis':
  #   loss_fn = get_loss_fn_basis(sde, train, reduce_mean=reduce_mean,
  #                             continuous=True, likelihood_weighting=likelihood_weighting)
  # elif config.training.conditional_model == 'equal_energy':
  #   loss_fn = get_equal_energy_loss_fn(sde, train, reduce_mean=reduce_mean,
  #                             continuous=True, likelihood_weighting=likelihood_weighting)
  # elif config.training.conditional_model == 'latent_contrastive':
  #   loss_fn = get_latent_contrastive_loss_fn(sde, train, reduce_mean=reduce_mean,
  #                             continuous=True, likelihood_weighting=likelihood_weighting)
  
  if continuous:
    if config.training.forward_diffusion == 'aniso':
      loss_fn = get_latent_loss_fn_anisotropic(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
    elif config.training.forward_diffusion == 'spectral': 
      loss_fn = get_latent_loss_fn_spectral(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
    elif config.training.forward_diffusion == 'predict': 
      loss_fn = get_latent_loss_fn_predict(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
    elif config.training.forward_diffusion == 'fkac': 
      loss_fn = get_latent_loss_fn_fkac(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting, t_batch=config.training.t_batch, loss_type = config.training.loss_type)
    elif config.training.forward_diffusion == 'predict_multi': 
      loss_fn = get_latent_loss_fn_predict_multi(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
    elif config.training.forward_diffusion == 'neural_ode': 
      loss_fn = get_latent_loss_fn_neural_ode(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
    elif config.training.forward_diffusion == 'neural_sde': 
      loss_fn = get_latent_loss_fn_neural_sde(sde, train, continuous=True)
  # elif config.training.conditional_model == 'latent_variational':
  #   assert config.model.name == 'ddpm_latent_variational', "The model must be variational to use variational method"
  #   if config.training.reconstruction_loss:
  #     loss_fn = get_variational_latent_loss_fn(sde, train, reduce_mean=reduce_mean,
  #                             continuous=True, likelihood_weighting=likelihood_weighting, reconstruction_loss=True)
 
  
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      if config.training.forward_diffusion == 'predict_multi': 
        loss_fn = get_ddpm_loss_fn_predict_multi(sde, train, reduce_mean=reduce_mean, continuous = continuous)
      else:
        loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  
  
  def step_fn_spectral(state, batch, class_label=None):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    model_spectral = state['model_spectral']
    if train:
      optimizer = state['optimizer']
      optimizer_spectral = state['optimizer_spectral']
      optimizer.zero_grad()
      optimizer_spectral.zero_grad()
      if class_label is not None:
        loss = loss_fn(model, model_spectral, batch, class_label)
      else:
        loss = loss_fn(model, model_spectral, batch)
      # if (config.training.conditional_model == 'latent' and config.training.reconstruction_loss) or (config.training.conditional_model == 'latent_contrastive'):
      #   loss[0].backward()
      # elif (config.training.conditional_model == 'latent' and config.training.regularization == 'highvar'):
      #   loss[0].backward()
      # else:
      loss[0].backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      optimize_fn(optimizer_spectral, model_spectral.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
      state['ema_spectral'].update(model_spectral.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

        ema_spectral = state['ema_spectral']
        ema_spectral.store(model_spectral.parameters())
        ema_spectral.copy_to(model_spectral.parameters())
        if class_label is not None:
          loss = loss_fn(model, model_spectral,batch, class_label)
        else:
          loss = loss_fn(model,model_spectral,batch)
        ema_spectral.restore(model_spectral.parameters())

    return loss

  return step_fn_spectral