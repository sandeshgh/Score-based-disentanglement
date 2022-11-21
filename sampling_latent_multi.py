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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import torch.nn as nn
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from utils import data_plot_proj_2D
import sde_lib
from models import utils as mutils
import copy
from torchdiffeq import odeint_adjoint as odeint

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device,
                                  compositional=config.training.compositional)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 compositional=config.training.compositional,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn
def max_normalize_by_batch(x):
  y = x.view(x.shape[0], -1)
  y_max, _ = torch.max(y, dim =-1)
  out = x/y_max[:, None, None, None]
  return out

def get_visual_fn(config, sde, shape, inverse_scaler, eps):
  # predictor = get_predictor(config.sampling.predictor.lower())
  # corrector = get_corrector(config.sampling.corrector.lower())
  # snr=config.sampling.snr
  # n_steps=config.sampling.n_steps_each
  # probability_flow=config.sampling.probability_flow
  continuous=config.training.continuous
  # compositional=config.training.compositional
  # denoise=config.sampling.noise_removal
  device=config.device

  def visual_fn(model, x_in, viz_dir = None, alpha =0.1):
    with torch.no_grad():
      # Initial sample
      # x = sde.prior_sampling(shape).to(device)
      n_im = 2
      eps = 1e-3
      timesteps = torch.linspace(eps, 2*eps, n_im).to(device)
      
      score_fn = mutils.get_latent_score_fn(sde, model, train=False, continuous=continuous, generate=True)
      
      model.eval()
      for i in range(n_im):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        mean, std = sde.marginal_prob(x_in, vec_t)
        z = torch.randn_like(x_in)
        x = x_in # mean + std[:, None, None, None] * z #
        
        latent_1 = model.module.encoder_1(x_in)
        score_1 = score_fn(x, vec_t, latent_1)
        im_1 = score_1 #std[:, None, None, None] * score_1#alpha*score_1+(1-alpha)*x_in
        im_1 = torch.sqrt(torch.sum(im_1**2, dim = 1, keepdim=True))
        im_1 = max_normalize_by_batch(im_1)

        latent_2 = model.module.encoder_2(x_in)
        score_2 = score_fn(x, vec_t, latent_2)
        im_2 = score_2
        im_2 = torch.sqrt(torch.sum(im_2**2, dim = 1, keepdim=True))
        im_2 = max_normalize_by_batch(im_2)

        latent_3 = model.module.encoder_3(x_in)
        score_3 = score_fn(x, vec_t, latent_3)
        im_3 = score_3
        im_3 = torch.sqrt(torch.sum(im_3**2, dim = 1, keepdim=True))
        im_3 = max_normalize_by_batch(im_3)

        im = torch.cat((im_1.unsqueeze(-1), im_2.unsqueeze(-1), im_3.unsqueeze(-1)), dim=-1).unsqueeze(-1)
        # im = inverse_scaler(im)

        if i >0:
          out_stack = torch.cat((out_stack,im), dim=-1)
        else:
          out_stack = im
      
      return out_stack      

      # latent = torch.cat((latent_1.unsqueeze(-1), latent_2.unsqueeze(-1), latent_3.unsqueeze(-1)), dim =-1)

  return visual_fn


def get_analysis_fn(config, sde, shape, inverse_scaler, eps, type=None):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    Latent analysis function.
  """

  
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())
  if type =='close':
    analysis_fn = get_latent_analyzer_close(sde=sde,
                                  shape=shape,
                                  predictor=predictor,
                                  corrector=corrector,
                                  inverse_scaler=inverse_scaler,
                                  snr=config.sampling.snr,
                                  n_steps=config.sampling.n_steps_each,
                                  probability_flow=config.sampling.probability_flow,
                                  continuous=config.training.continuous,
                                  compositional=config.training.compositional,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device,
                                  config = config)
  elif type =='uncond':
    analysis_fn = get_latent_analyzer_w_uncond(sde=sde,
                                  shape=shape,
                                  predictor=predictor,
                                  corrector=corrector,
                                  inverse_scaler=inverse_scaler,
                                  snr=config.sampling.snr,
                                  n_steps=config.sampling.n_steps_each,
                                  probability_flow=config.sampling.probability_flow,
                                  continuous=config.training.continuous,
                                  compositional=config.training.compositional,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device,
                                  config = config)
  elif type is not None:
    analysis_fn = get_latent_analyzer_mode(sde=sde,
                                  shape=shape,
                                  predictor=predictor,
                                  corrector=corrector,
                                  inverse_scaler=inverse_scaler,
                                  snr=config.sampling.snr,
                                  n_steps=config.sampling.n_steps_each,
                                  probability_flow=config.sampling.probability_flow,
                                  continuous=config.training.continuous,
                                  compositional=config.training.compositional,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device,
                                  config = config,
                                  mode = type)
  else:
    analysis_fn = get_latent_analyzer(sde=sde,
                                  shape=shape,
                                  predictor=predictor,
                                  corrector=corrector,
                                  inverse_scaler=inverse_scaler,
                                  snr=config.sampling.snr,
                                  n_steps=config.sampling.n_steps_each,
                                  probability_flow=config.sampling.probability_flow,
                                  continuous=config.training.continuous,
                                  compositional=config.training.compositional,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device,
                                  config = config)
  # else:
  #   raise ValueError(f"Sampler name {sampler_name} unknown.")

  return analysis_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, mode=None):
    super().__init__(sde, score_fn, probability_flow)
    self.rsde = sde.reverse_multilatent(score_fn, probability_flow)
    self.mode = mode
    if mode =='uncond':
      self.rsde = sde.reverse_multilatent_uncond(score_fn, probability_flow)
    elif mode =='mix':
      self.rsde = sde.reverse_multilatent_mix(score_fn, probability_flow)
    elif mode == 'tune':
      self.rsde = sde.reverse_multilatent_tune(score_fn, probability_flow)
    elif mode == 'switch_1_u':
      self.rsde = sde.reverse_switch_1_u(score_fn, probability_flow)
    elif mode == 'switch_2_u':
      self.rsde = sde.reverse_switch_2_u(score_fn, probability_flow)


  def update_fn(self, x, t, latent, alpha= None):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    if self.mode is not None:
      drift, diffusion = self.rsde.sde(x, t, latent, alpha )
    else:
      drift, diffusion = self.rsde.sde(x, t, latent)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous, compositional, mode = None, latent=None, alpha=None):
  """A wrapper that configures and returns the update function of predictors."""
  
  score_fn = mutils.get_latent_score_fn(sde, model, train=False, continuous=continuous, generate=True)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  elif mode is not None:
    predictor_obj = predictor(sde, score_fn, probability_flow, mode = mode)
  elif alpha is not None:
    predictor_obj = predictor(sde, score_fn, probability_flow, mode = 'uncond')
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  if alpha is not None:
    return predictor_obj.update_fn(x, t, latent, alpha)
  else:
    return predictor_obj.update_fn(x, t, latent)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps, compositional):
  """A wrapper tha configures and returns the update function of correctors."""
  
  score_fn = mutils.get_latent_score_fn(sde, model, train=False, continuous=continuous, generate=True)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   compositional= False, denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          compositional=compositional)

  def pc_sampler(model, latent):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      latent = latent.unsqueeze(0)
      latent = torch.cat(x.shape[0]*[latent], dim = 0).to(x.device)

      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_update_fn(x, vec_t, model=model, latent=latent)

      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler

def get_latent_analyzer(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   compositional= False, denoise=True, eps=1e-3, device='cuda', config = None):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional)
  predictor_update_fn_mix = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional,
                                          mode = 'mix')
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          compositional=compositional)
  def generate_samples(model, x, timesteps, latent, mix = False):
    if mix:
        predictor_fn = predictor_update_fn_mix
    else:
        predictor_fn = predictor_update_fn
    for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_fn(x, vec_t, model=model, latent=latent)

    return inverse_scaler(x_mean if denoise else x)
  
  def modify_latent(z, pos):
    # z1= z+0.2 if z <=0.8 else z - 0.8
    # z2= z+0.4 if z <=0.6 else z - 0.6
    # z3= z+0.6 if z <=0.4 else z - 0.4

    # z1= torch.clamp(z-0.2, min=0)
    z1=copy.deepcopy(z)
    z2= copy.deepcopy(z)
    z3= copy.deepcopy(z)
    k = 5

    z1[:, :pos] =0 
    z1[:, pos+1:] =0 

    z2[:, :pos+k] =0 
    z2[:, pos+k+1:] =0 

    z3[:, :pos] =0 
    z3[:, pos+k:] =0 

    # z3[:, :pos] =0 
    # z3[:, pos+1:] =0

    return z1, z2, z3
  
  def generate_latent(z, pos):
    
    z1=torch.zeros_like(z)
    z2= torch.zeros_like(z)
    z3= torch.zeros_like(z)
    k = 1

    z1[:, pos] =1
    

    z2[:, pos+k] =1
    

    z3[:, pos:pos+k] =1
    

    return z1, z2, z3



  def latent_analyzer(model, x_in, num_latent=3, latent_position=1):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """

    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      model.eval()
      if config.training.conditional_model == 'latent_variational':
        latent, _ = model.module.encode(x_in)
      else:
        latent = model.module.encode(x_in)
      # latent = model.module.encode(x_in)
      # latent1, latent2, latent3 = modify_latent(latent, latent_position)
      latent1, latent2, latent3 = generate_latent(latent, latent_position)
      latent1, latent2, latent3 = latent1.to(x.device), latent2.to(x.device), latent3.to(x.device)
      latent3 = latent
      print('latent1', latent1[:3])
      print('latent2', latent2[:3])
      print('latent3', latent3[:10])
      # latent = latent.unsqueeze(0)
      # latent = torch.cat(x.shape[0]*[latent], dim = 0).to(x.device)
      out1 = generate_samples(model, x, timesteps, latent1)
      out2 = generate_samples(model, x, timesteps, latent2)
      # out3 = generate_samples(model, x, timesteps, latent3)

      out3 = generate_samples(model, x, timesteps, torch.cat((latent1.unsqueeze(-1), latent2.unsqueeze(-1)), dim=-1), mix = True)
    
    
    time = sde.N * (n_steps + 1)
    return out1, out2, out3

  return latent_analyzer

class ReverseDrift(nn.Module):
    def __init__(self, sde, model, score_fn = None, x_in=None, latent=None):
      super().__init__()
      self.model = model
      self.sde = sde
      self.x_in = x_in
      self.score_fn = score_fn
      self.latent = latent
      
    def update_latent(self, lat):
      self.latent = lat
    
    def drift_fn(self, x, t):
      # # score_fn = get_score_fn(self.sde, self.model, train=False, continuous=True)
      # rsde = self.sde.reverse(self.score_fn, probability_flow=True)
      # return rsde.sde(x, t, latent)[0]
      drift, diffusion = self.sde.sde(x, t)
      # if latent is None:
      #   latent = self.model.module.encode(self.x_in)
      # score_fn = mutils.get_latent_score_fn(self.sde, self.model, train=train, continuous=continuous, generate=True)
    
      score = self.score_fn(x, t, self.latent)
                
      drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5)
      # Set the diffusion function to zero for ODEs.
      # diffusion = 0. 
      return drift
    
    def forward(self, t, x):
      shape = x.shape
      vec_t = torch.ones(shape[0], device=x.device) * t
      drift = self.drift_fn(x, vec_t)
      return drift

def get_latent_analyzer_close(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   compositional= False, denoise=True, eps=1e-3, device='cuda', config=None):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional)
  predictor_update_fn_mix = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional,
                                          mode = 'mix')
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          compositional=compositional)
  def generate_samples(model, x, timesteps, latent, mix = False, alpha = None):
    if mix:
        predictor_fn = predictor_update_fn_mix
    else:
        predictor_fn = predictor_update_fn
    for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        # x, x_mean = predictor_fn(x, vec_t, model=model, latent=latent)
        x, x_mean = predictor_fn(x, vec_t, model=model, latent=latent, alpha=alpha)

    return inverse_scaler(x_mean if denoise else x)
  
  def generate_samples_ode(ode_func, x, mix = False):
    sol = odeint(ode_func, x, t=torch.linspace(sde.T, 1e-5, 2).to(x.device), method = 'rk4', rtol=1e-5)
    sol = sol[-1]
    
    return inverse_scaler(sol)
  
  def generate_samples_sde(ode_func, x, timesteps, latent, mix = False):
    sol = odeint(ode_func, x, t=torch.linspace(sde.T, eps, 2).to(x.device), method = 'rk4', rtol=1e-5)
    
    return inverse_scaler(sol)
  
  def modify_latent(z, pos):
    # z1= z+0.2 if z <=0.8 else z - 0.8
    # z2= z+0.4 if z <=0.6 else z - 0.6
    # z3= z+0.6 if z <=0.4 else z - 0.4

    # z1= torch.clamp(z-0.2, min=0)
    z1=copy.deepcopy(z)
    z2= copy.deepcopy(z)
    z3= copy.deepcopy(z)
    k = 5

    z1[:, :pos] =0 
    z1[:, pos+1:] =0 

    z2[:, :pos+k] =0 
    z2[:, pos+k+1:] =0 

    z3[:, :pos] =0 
    z3[:, pos+k:] =0 

    # z3[:, :pos] =0 
    # z3[:, pos+1:] =0

    return z1, z2, z3
  
  def get_dim_interp(latent, n_interp =5, dim = 0):
    
    z_0=torch.zeros_like(latent)
    z2= torch.zeros_like(latent)
    z_0[:,dim]=1
    dim1= dim+1
    z2[:,dim1%3]=1


    alpha = torch.linspace(0,1,n_interp).unsqueeze(0).unsqueeze(0).to(latent.device)
    out = z_0.unsqueeze(-1)*(1-alpha) + z2.unsqueeze(-1)*alpha
  
    return out
  
#   def get_one_hot_interp(latent, n_interp=5):
#     z_0 = latent[0]
#     z_one_hot = torch.zeros_like(z_0)
#     ind_max = torch.argmax(z_0)
#     z_one_hot[ind_max] = 1
#     z_one_hot = z_one_hot.repeat(latent.shape[0],1)
#     alpha = torch.linspace(0,1,n_interp).unsqueeze(0).unsqueeze(0).to(latent.device)
#     out = latent.unsqueeze(-1)*(1-alpha) + z_one_hot.unsqueeze(-1)*alpha
    
#     return out

  def get_one_hot_interp(latent, n_interp=5):
    z_0 = latent[0, :, 0]
    z_one_hot = torch.zeros_like(z_0)
    ind_max = torch.argmax(z_0)
    z_one_hot[ind_max] = 1
    z_one_hot = z_one_hot.unsqueeze(0).unsqueeze(-1)
    z_one_hot = z_one_hot.repeat(latent.shape[0],1, latent.shape[2])
    alpha = torch.linspace(0,1,n_interp).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(latent.device)
    out = latent.unsqueeze(-1)*(1-alpha) + z_one_hot.unsqueeze(-1)*alpha
    
    return out



  def latent_analyzer(model, x_in, n_interp = 10, mode = 'project_one_hot', dim_idx = 1, sampler = 'pc', viz_dir = None, batch_idx =0):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """

    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      model.eval()
      
      latent_1 = model.module.encoder_1(x_in)
      latent_2 = model.module.encoder_2(x_in)
      latent_3 = model.module.encoder_3(x_in)

      latent = torch.cat((latent_1.unsqueeze(-1), latent_2.unsqueeze(-1), latent_3.unsqueeze(-1)), dim =-1)
      # latent1, latent2, latent3 = modify_latent(latent, latent_position)
      if mode == 'mix':
        x = x.unsqueeze(-1).repeat(1,1,1,1,4)

        for i in range(3):
          print(i, 'out of 3')
          
          
          out_i = generate_samples(model, x, timesteps, latent, mix=True, alpha = i)
          if i >0:
            out_stack = torch.cat((out_stack,out_i.unsqueeze(-1)), dim=-1)
          else:
            out_stack = out_i.unsqueeze(-1)
      else:
        if mode == 'project_one_hot':
          latents_interp_stack = get_one_hot_interp(latent, n_interp =n_interp)
        elif mode == 'test_dim':
          latents_interp_stack = get_dim_interp(latent, n_interp =n_interp, dim = dim_idx)
        
      #   data_plot_proj_2D(latents_interp_stack[0].permute(1,0).cpu().numpy(),viz_dir=viz_dir,idx='interpolation_'+str(dim_idx))
        data_plot_proj_2D(latents_interp_stack[0,:,0].permute(1,0).cpu().numpy(),viz_dir=viz_dir,idx=mode+'_dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_a')
        data_plot_proj_2D(latents_interp_stack[0,:,1].permute(1,0).cpu().numpy(),viz_dir=viz_dir,idx=mode+'_dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_b')
        x = x.unsqueeze(-1).repeat(1,1,1,1,4)

        for i in range(latents_interp_stack.shape[-1]):
          latent_i = latents_interp_stack[:,:,:,i].to(x.device)
          print('latent:', i)
          if sampler == 'ode':
            score_fn = mutils.get_latent_score_fn(sde, model, train=False, continuous=continuous, generate=True)
            ode_func = ReverseDrift(sde, model, score_fn, latent= latent_i)
            ode_func.update_latent(latent_i)

            out_i = generate_samples_ode(ode_func, x)
          elif sampler == 'sde':
            out_i = generate_samples_sde(ode_func, x, timesteps, latent_i)
          else:
            out_i = generate_samples(model, x, timesteps, latent_i)
          if i >0:
            out_stack = torch.cat((out_stack,out_i.unsqueeze(-1)), dim=-1)
          else:
            out_stack = out_i.unsqueeze(-1)

    return out_stack

  return latent_analyzer

def get_latent_analyzer_w_uncond(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   compositional= False, denoise=True, eps=1e-3, device='cuda', config=None):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional)
  # predictor_update_fn_mix = functools.partial(shared_predictor_update_fn,
  #                                         sde=sde,
  #                                         predictor=predictor,
  #                                         probability_flow=probability_flow,
  #                                         continuous=continuous,
  #                                         compositional=compositional,
  #                                         mode = 'mix')
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          compositional=compositional)
  def generate_samples(model, x, timesteps, latent, alpha = None):
    # if mix:
    #     predictor_fn = predictor_update_fn_mix
    # else:
    predictor_fn = predictor_update_fn
    for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_fn(x, vec_t, model=model, latent=latent, alpha=alpha)

    return inverse_scaler(x_mean if denoise else x)
  
  
  
  # def get_dim_interp(latent, n_interp =5, dim = 0):
    
  #   z_0=torch.zeros_like(latent)
  #   z2= torch.zeros_like(latent)
  #   z_0[:,dim]=1
  #   dim1= dim+1
  #   z2[:,dim1%3]=1


  #   alpha = torch.linspace(0,1,n_interp).unsqueeze(0).unsqueeze(0).to(latent.device)
  #   out = z_0.unsqueeze(-1)*(1-alpha) + z2.unsqueeze(-1)*alpha
  
  #   return out
  


  # def get_one_hot_interp(latent, n_interp=5):
  #   z_0 = latent[0, :, 0]
  #   z_one_hot = torch.zeros_like(z_0)
  #   ind_max = torch.argmax(z_0)
  #   z_one_hot[ind_max] = 1
  #   z_one_hot = z_one_hot.unsqueeze(0).unsqueeze(-1)
  #   z_one_hot = z_one_hot.repeat(latent.shape[0],1, latent.shape[2])
  #   alpha = torch.linspace(0,1,n_interp).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(latent.device)
  #   out = latent.unsqueeze(-1)*(1-alpha) + z_one_hot.unsqueeze(-1)*alpha
    
  #   return out



  def latent_analyzer(model, x_in, n_interp = 11, mode = 'project_one_hot', dim_idx = 1, sampler = 'pc', viz_dir = None, batch_idx =0):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """

    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      model.eval()
      
      latent_1 = model.module.encoder_1(x_in)
      latent_2 = model.module.encoder_2(x_in)
      latent_3 = model.module.encoder_3(x_in)

      latent = torch.cat((latent_1.unsqueeze(-1), latent_2.unsqueeze(-1), latent_3.unsqueeze(-1)), dim =-1)
      latent = latent.to(x.device)
      # latent1, latent2, latent3 = modify_latent(latent, latent_position)
      # if mode == 'project_one_hot':
      #   latents_interp_stack = get_one_hot_interp(latent, n_interp =n_interp)
      # elif mode == 'test_dim':
      #   latents_interp_stack = get_dim_interp(latent, n_interp =n_interp, dim = dim_idx)
      
    #   data_plot_proj_2D(latents_interp_stack[0].permute(1,0).cpu().numpy(),viz_dir=viz_dir,idx='interpolation_'+str(dim_idx))
      # data_plot_proj_2D(latents_interp_stack[0,:,0].permute(1,0).cpu().numpy(),viz_dir=viz_dir,idx=mode+'_dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_a')
      # data_plot_proj_2D(latents_interp_stack[0,:,1].permute(1,0).cpu().numpy(),viz_dir=viz_dir,idx=mode+'_dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_b')
      x = x.unsqueeze(-1).repeat(1,1,1,1,4)
      alpha_ar = torch.linspace(0,1,n_interp).to(latent.device)

      for i in range(alpha_ar.shape[0]):
        alpha_i = alpha_ar[i]
        
        print('latent:', i)
        
        out_i = generate_samples(model, x, timesteps, latent, alpha_i)
        if i >0:
          out_stack = torch.cat((out_stack,out_i.unsqueeze(-1)), dim=-1)
        else:
          out_stack = out_i.unsqueeze(-1)

    return out_stack

  return latent_analyzer

def get_latent_analyzer_mode(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   compositional= False, denoise=True, eps=1e-3, device='cuda', config=None, mode=None):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  # predictor_update_fn = functools.partial(shared_predictor_update_fn,
  #                                         sde=sde,
  #                                         predictor=predictor,
  #                                         probability_flow=probability_flow,
  #                                         continuous=continuous,
  #                                         compositional=compositional)
  predictor_update_fn_mode = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          compositional=compositional,
                                          mode = mode)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          compositional=compositional)
  def generate_samples(model, x, timesteps, latent, alpha = None):
    # if mix:
    #     predictor_fn = predictor_update_fn_mix
    # else:
    predictor_fn = predictor_update_fn_mode
    for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_fn(x, vec_t, model=model, latent=latent, alpha=alpha)

    return inverse_scaler(x_mean if denoise else x)
  
  

  def latent_analyzer(model, x_in, n_interp = 11, mode = 'project_one_hot', dim_idx = 1, sampler = 'pc', viz_dir = None, batch_idx =0):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """

    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      model.eval()
      
      latent_1 = model.module.encoder_1(x_in)
      latent_2 = model.module.encoder_2(x_in)
      latent_3 = model.module.encoder_3(x_in)

      latent = torch.cat((latent_1.unsqueeze(-1), latent_2.unsqueeze(-1), latent_3.unsqueeze(-1)), dim =-1)
      latent = latent.to(x.device)
     
      x = x.unsqueeze(-1).repeat(1,1,1,1,4)
      alpha_ar = torch.linspace(0,1,n_interp).to(latent.device)

      for i in range(alpha_ar.shape[0]):
        alpha_i = alpha_ar[i]
        
        print('latent:', i)
        
        out_i = generate_samples(model, x, timesteps, latent, alpha_i)
        if i >0:
          out_stack = torch.cat((out_stack,out_i.unsqueeze(-1)), dim=-1)
        else:
          out_stack = out_i.unsqueeze(-1)

    return out_stack

  return latent_analyzer



def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda', compositional=False):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    if compositional:
      score_fn = mutils.get_compositional_score_fn(sde, model, train=False, continuous=True)
    else:
      score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      # {Sandesh Comments: note that the time goes in the reverse direction from T to 0(eps). The reason is that ODE equation is the forward equation going from good image to noise
      # Here, since we want to sample from the good distribution p(x), we need to go in the reverse of the forward ODE equation, i.e. from noise to good samples, hence the reverse time 
      # passed to ode solver
      # Don't be confused by the fact that drift_fn is calling rsde. The ODE is still forward. It's just that forward ODE resembles reverse SDE with a factor of 0.5
      # Hence, the authors calls rsde with probability_flow =True which takes care of factor 0.5 and gives a forward ODE equations. :D  Sandesh comment ends}
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
