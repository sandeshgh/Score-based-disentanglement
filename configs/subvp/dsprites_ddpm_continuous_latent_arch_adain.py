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
import ml_collections


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
  training.regularization = None
  training.batch_size = 128
  training.snapshot_freq = 5000
  

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'
  sampling.type = 'conditional'
  sampling.plot_score = True

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ddpm_latent_Adain'
  
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
  model.latent_dim = 128
  model.rotate_basis = True

  config.eval.begin_ckpt = 32
  config.eval.end_ckpt = 32

  conf = config.autoenc_conf = ml_collections.ConfigDict()
  conf.batch_size = 32
  conf.beatgans_gen_type = 'ddpm'  #'ddim'
#   conf.beta_scheduler = 'linear'
#   conf.data_name = 'ffhq'
#   conf.diffusion_type = 'beatgans'
#   conf.eval_ema_every_samples = 200_000
#   conf.eval_every_samples = 200_000
#   conf.fp16 = True
#   conf.lr = 1e-4
# #   conf.model_name = model.name.beatgans_autoenc
#   conf.net_attn = (16, )
#   conf.net_beatgans_attn_head = 1
#   conf.net_beatgans_embed_channels = 512
#   conf.net_beatgans_resnet_two_cond = True
#   conf.net_ch_mult = (1, 2, 4, 8)
#   conf.net_ch = 64
#   conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
#   conf.net_enc_pool = 'adaptivenonzero'
#   conf.sample_size = 32
#   conf.T_eval = 20
#   conf.T = 1000




  return config
