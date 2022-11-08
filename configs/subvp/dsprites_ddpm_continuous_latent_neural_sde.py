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

# Lint as: python3
"""Training DDPM with sub-VP SDE."""

from configs.default_dsprites_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'subvpsde'
  training.continuous = True
  training.reduce_mean = True
  training.compositional = False
  training.conditional_model = 'latent'
  training.reconstruction_loss = False
  training.snapshot_freq = 2000
  training.forward_diffusion = 'neural_sde'
  

  # sampling
  sampling = config.sampling
  sampling.method = 'neural_sde'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.type = 'conditional'
  sampling.plot_score = True

  # data
  data = config.data
  data.centered = True
  data.num_channels = 1
  # model
  model = config.model
  model.name = 'ddpm_latent_aniso'
  model.spectral_model_name = 'encode_resnet'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'elu'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.latent_dim = 8
  model.rotate_basis = True

  config.eval.begin_ckpt = 1
  config.eval.end_ckpt = 2

  return config