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
from torchdiffeq import odeint_adjoint as odeint


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

def peek_ode(x_in, model, sde, score_fn, eps=1e-3, variational=False):
  shape = x_in.shape
  device = x_in.device

  class ReverseDrift(nn.Module):
    def __init__(self, sde, score_fn, latent):
      super().__init__()
      self.score_fn = score_fn
      self.latent = latent
      self.sde = sde
    
    def drift_fn(self, x, t, latent):
      # # score_fn = get_score_fn(self.sde, self.model, train=False, continuous=True)
      # rsde = self.sde.reverse(self.score_fn, probability_flow=True)
      # return rsde.sde(x, t, latent)[0]
      drift, diffusion = self.sde.sde(x, t)
      if latent is not None:
        score = self.score_fn(x, t, latent)
      else:
        score = self.score_fn(x, t)
          
      drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5)
      # Set the diffusion function to zero for ODEs.
      # diffusion = 0. 
      return drift
    
    def forward(self, t, x):
      vec_t = torch.ones(shape[0], device=x.device) * t
      drift = self.drift_fn(x, vec_t, self.latent)
      return drift

    # def ode_func(t, x):
    #   x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
    #   vec_t = torch.ones(shape[0], device=x.device) * t
    #   drift = drift_fn(model, x, vec_t)
    #   return to_flattened_numpy(drift)


  if variational:
    latent, kl_loss = model.module.encode(x_in)
  else:
    latent = model.module.encode(x_in)

  # score_fn = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous)
  ode_func = ReverseDrift(sde, score_fn, latent)
  
  
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
  return sol


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

def get_latent_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5, gamma=5e-2, reconstruction_loss=False):
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
      score_fn_gen = mutils.get_latent_score_fn(sde, model, train=train, continuous=continuous, generate=True)
      peek_generated = peek_ode(batch, model, sde, score_fn_gen)
      rec_loss = mse_loss(peek_generated, batch)
      loss = loss_score + gamma*rec_loss
      return loss, rec_loss, loss_score
    else:
      return loss_score
    

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
  
  elif config.training.conditional_model == 'latent':
    if config.training.reconstruction_loss:
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
      if config.training.conditional_model == 'latent' and config.training.reconstruction_loss:
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

  return step_fn