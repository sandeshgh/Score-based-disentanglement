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

"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np
import os
# os.environ["CUDA_DEVIC_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3"



_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  # parameters need to be adapted if number of time steps differs from 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config, model_name = None, device = None):
  """Create the score model."""
  if not model_name:
    model_name = config.model.name
  score_model = get_model(model_name)(config)
  if not device:
    device = config.device
  score_model = score_model.to(device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model

def create_model_spectral(config):
  """Create the score model."""
  model_name = config.model.spectral_model_name
  model = get_model(model_name)(config)
  model = model.to(config.device)
  model = torch.nn.DataParallel(model)
  return model


def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.
  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the score-based model.
    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn

def get_model_fn_label(model, train=False):
  """Create a function to give the output of the score-based model.
  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """

  def model_fn(x, labels, class_label):
    """Compute the output of the score-based model.
    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.
    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels, class_label)
    else:
      model.train()
      return model(x, labels, class_label)

  return model_fn

def get_model_equal_energy_fn(model, train=False, generate=False):
  """Create a function to give the output of the score-based model.
  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """
  if generate:
    def model_fn(x, labels, latent):
      """Compute the output of the score-based model.
      Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
          for different models.
      Returns:
        A tuple of (model output, new mutable states)
      """
      if not train:
        model.eval()
        return model.module.forward_generate(x, labels, latent)
      else:
        model.train()
        return model.module.forward_generate(x, labels, latent)
  else:
    def model_fn(x, labels, x_clean):
      """Compute the output of the score-based model.
      Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
          for different models.
      Returns:
        A tuple of (model output, new mutable states)
      """
      if not train:
        model.eval()
        return model(x, labels, x_clean)
      else:
        model.train()
        return model(x, labels, x_clean)

  return model_fn

def get_model_fn_latent_factor(model, train=False, generate=False):
  """Create a function to give the output of the score-based model.
  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.
  Returns:
    A model function.
  """
  if generate:
    def model_fn(x, labels, latent):
      """Compute the output of the score-based model.
      Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
          for different models.
      Returns:
        A tuple of (model output, new mutable states)
      """
      if not train:
        model.eval()
        return model.module.forward_generate(x, labels, latent)
      else:
        model.train()
        return model.module.forward_generate(x, labels, latent)
  else:
    def model_fn(x, labels, x_clean):
      """Compute the output of the score-based model.
      Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
          for different models.
      Returns:
        A tuple of (model output, new mutable states)
      """
      if not train:
        model.eval()
        return model(x, labels, x_clean)
      else:
        model.train()
        return model(x, labels, x_clean)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        # std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        std = sde.marginal_std(t)
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_score_fn_forward_unconditional(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  # model_fn = get_model_fn(model, train=train)
  model_fn = model
  if not train:
    model_fn.eval()
  else:
    model_fn.train()

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn.module.forward_uncond(x, labels)
        # std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        std = sde.marginal_std(t)
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  # elif isinstance(sde, sde_lib.VESDE):
  #   def score_fn(x, t):
  #     if continuous:
  #       labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
  #     else:
  #       # For VE-trained models, t=0 corresponds to the highest noise level
  #       labels = sde.T - t
  #       labels *= sde.N - 1
  #       labels = torch.round(labels).long()

  #     score = model_fn(x, labels)
  #     return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_score_fn_label(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_fn_label(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, class_label):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels, class_label)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, class_label):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels, class_label)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_latent_score_fn(sde, model, train=False, continuous=False, generate = False, variational=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  if generate:
    variational = False
    
  get_model_fn_latent = get_model_equal_energy_fn
  model_fn = get_model_fn_latent(model, train=train, generate = generate)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, x_clean):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        if variational:
          score, kl_loss = model_fn(x, labels, x_clean)
        else:
          score = model_fn(x, labels, x_clean)
        std = sde.marginal_std(t)
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        if variational:
          score, kl_loss = model_fn(x, labels, x_clean)
        else:
          score = model_fn(x, labels, x_clean)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / (std[:, None, None, None].detach())
      if variational:
        return score, kl_loss
      else:
        return score


  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, x_clean):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()
      if variational:
        score, kl_loss = model_fn(x, labels, x_clean)
        return score, kl_loss
      else:
        score = model_fn(x, labels, x_clean)
        return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_latent_score_fn_predict_multi(sde, model, train=False, continuous=False, generate = False, variational=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  if generate:
    variational = False
    
  # get_model_fn_latent = 
  model_fn = get_model_equal_energy_fn(model, train=train, generate = generate)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, x_clean):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        if variational:
          score, kl_loss = model_fn(x, labels, x_clean)
        else:
          score = model_fn(x, labels, x_clean)
        std = sde.marginal_std(t)
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        if variational:
          score, kl_loss = model_fn(x, labels, x_clean)
        else:
          score = model_fn(x, labels, x_clean)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / (std[:, None, None, None, None].detach())
      # score[1] = -score[1] / (std[:, None, None, None].detach())
      
      return score


  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, x_clean):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()
      
      score = model_fn(x, labels, x_clean)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_latent_score_fn_spectral(sde, model, train=False, continuous=False, generate = False, variational=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
    
  model_fn = get_model_fn_label(model, train=train)

  if isinstance(sde, sde_lib.Spectral_VPSDE):
    def score_fn(x, t, latent, z_x):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999        
        score = model_fn(x, labels, latent)
        std = sde.marginal_prob(torch.zeros_like(x), t, z_x.detach())[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        
        score = model_fn(x, labels, latent)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score * (std.detach().pow(-1)) #score = -score * (std.detach())  #
  
      return score


  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, x_clean):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()
      if variational:
        score, kl_loss = model_fn(x, labels, x_clean)
        return score, kl_loss
      else:
        score = model_fn(x, labels, x_clean)
        return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def get_score_fn_neural_ode(model, train=False, continuous=False, generate = False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
    
  if not train:
    model.eval()
  else:
    model.train()
    

  def score_fn(x, t, latent):

    score = model(x, t, latent)

    return score


  return score_fn


def get_score_fn_latent_factor(sde, model, train=False, continuous=False, generate=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_fn_latent_factor(model, train=train, generate=generate)

  # if not generate:

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, x_clean):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score, score2 = model_fn(x, labels, x_clean)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score, score2 = model_fn(x, labels, x_clean)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      score2 = -score2 / std[:, None, None, None]
      return score, score2

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, x_clean):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score, score2 = model_fn(x, labels, x_clean)
      return score, score2

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  
  return score_fn

def get_equal_energy_score_fn(sde, model, train=False, continuous=False, generate=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_equal_energy_fn(model, train=train, generate=generate)

  if not generate:

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      def score_fn(x, t, x_clean):
        # Scale neural network output by standard deviation and flip sign
        if continuous or isinstance(sde, sde_lib.subVPSDE):
          # For VP-trained models, t=0 corresponds to the lowest noise level
          # The maximum value of time embedding is assumed to 999 for
          # continuously-trained models.
          labels = t * 999
          score, score2 = model_fn(x, labels, x_clean)
          std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VP-trained models, t=0 corresponds to the lowest noise level
          labels = t * (sde.N - 1)
          score, score2 = model_fn(x, labels, x_clean)
          std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None, None]
        score2 = -score2 / std[:, None, None, None]
        return score, score2

    elif isinstance(sde, sde_lib.VESDE):
      def score_fn(x, t, x_clean):
        if continuous:
          labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VE-trained models, t=0 corresponds to the highest noise level
          labels = sde.T - t
          labels *= sde.N - 1
          labels = torch.round(labels).long()

        score, score2 = model_fn(x, labels, x_clean)
        return score, score2

    else:
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  else:
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      def score_fn(x, t, latent):
        # Scale neural network output by standard deviation and flip sign
        if continuous or isinstance(sde, sde_lib.subVPSDE):
          # For VP-trained models, t=0 corresponds to the lowest noise level
          # The maximum value of time embedding is assumed to 999 for
          # continuously-trained models.
          labels = t * 999
          score = model_fn(x, labels, latent)
          std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VP-trained models, t=0 corresponds to the lowest noise level
          labels = t * (sde.N - 1)
          score = model_fn(x, labels, latent)
          std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None, None]
        #score2 = -score2 / std[:, None, None, None]
        return score

    elif isinstance(sde, sde_lib.VESDE):
      def score_fn(x, t, latent):
        if continuous:
          labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VE-trained models, t=0 corresponds to the highest noise level
          labels = sde.T - t
          labels *= sde.N - 1
          labels = torch.round(labels).long()

        score = model_fn(x, labels, latent)
        return score

    else:
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")


  return score_fn

def get_score_fn_basis(sde, model, train=False, continuous=False, generate=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
  Returns:
    A score function.
  """
  model_fn = get_model_equal_energy_fn(model, train=train, generate=generate)

  if not generate:

    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      def score_fn(x, t, x_clean):
        # Scale neural network output by standard deviation and flip sign
        if continuous or isinstance(sde, sde_lib.subVPSDE):
          # For VP-trained models, t=0 corresponds to the lowest noise level
          # The maximum value of time embedding is assumed to 999 for
          # continuously-trained models.
          labels = t * 999
          scores, wt = model_fn(x, labels, x_clean)
          std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VP-trained models, t=0 corresponds to the lowest noise level
          labels = t * (sde.N - 1)
          scores, wt = model_fn(x, labels, x_clean)
          std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -scores / std[:, None, None, None, None]
        return score, wt

    elif isinstance(sde, sde_lib.VESDE):
      def score_fn(x, t, x_clean):
        if continuous:
          labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VE-trained models, t=0 corresponds to the highest noise level
          labels = sde.T - t
          labels *= sde.N - 1
          labels = torch.round(labels).long()

        scores, wt = model_fn(x, labels, x_clean)
        return scores, wt

    else:
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  else:
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      def score_fn(x, t, latent):
        # Scale neural network output by standard deviation and flip sign
        if continuous or isinstance(sde, sde_lib.subVPSDE):
          # For VP-trained models, t=0 corresponds to the lowest noise level
          # The maximum value of time embedding is assumed to 999 for
          # continuously-trained models.
          labels = t * 999
          score = model_fn(x, labels, latent)
          std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VP-trained models, t=0 corresponds to the lowest noise level
          labels = t * (sde.N - 1)
          score = model_fn(x, labels, latent)
          std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None, None]
        #score2 = -score2 / std[:, None, None, None]
        return score

    elif isinstance(sde, sde_lib.VESDE):
      def score_fn(x, t, latent):
        if continuous:
          labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
          # For VE-trained models, t=0 corresponds to the highest noise level
          labels = sde.T - t
          labels *= sde.N - 1
          labels = torch.round(labels).long()

        score = model_fn(x, labels, latent)
        return score

    else:
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")


  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))