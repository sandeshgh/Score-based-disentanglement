
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
# from losses import get_optimizer
# from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_gan as tfgan
# import tqdm
import io
# import likelihood
# import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
# from models import ncsnv2
from models import ncsnpp
# from models import ddpm as ddpm_model
# from models import layerspp
# from models import layers
# from models import normalization
# import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      EulerMaruyamaPredictor,
                      AncestralSamplingPredictor,
                      NoneCorrector,
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets



sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import cifar10_ncsnpp_continuous as configs
  ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
  config = configs.get_config()
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  sampling_eps = 1e-5
elif sde.lower() == 'vpsde':
  from configs.vp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/vp/cifar10_ddpmpp_continuous/checkpoint_8.pth"
  config = configs.get_config()
  sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3
elif sde.lower() == 'subvpsde':
  from configs.subvp import cifar10_ddpmpp_continuous as configs
  ckpt_filename = "exp/subvp/cifar10_ddpmpp_continuous/checkpoint_26.pth"
  config = configs.get_config()
  sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  sampling_eps = 1e-3

batch_size =   64#@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

#@title Likelihood computation
train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=True, evaluation=True)
eval_iter = iter(eval_ds)
bpds = []
likelihood_fn = get_likelihood_fn(sde, inverse_scaler, eps=1e-5)
for i in range(5):
    batch = next(iter(train_ds))
    img = batch['image']._numpy()
    img = torch.tensor(img).permute(0, 3, 1, 2).to(config.device)
    img = scaler(img)
    bpd, z, nfe = likelihood_fn(score_model, img)
    bpds.extend(bpd)
    print(f"average bpd: {torch.tensor(bpds).mean().item()}, NFE: {nfe}")

for i in range(5):
    batch = next(iter(eval_ds))
    img = batch['image']._numpy()
    img = torch.tensor(img).permute(0, 3, 1, 2).to(config.device)
    img = scaler(img)
    bpd, z, nfe = likelihood_fn(score_model, img)
    bpds.extend(bpd)
    print(f"average bpd cifar eval: {torch.tensor(bpds).mean().item()}, NFE: {nfe}")
#Previous model was cifar10 and tested on cifar10 also
#Now, we load celeba dataset and test the model trained on cifar10 to celeba

from configs.ve import celeba_ncsnpp as configs_celeba
config_celeba = configs_celeba.get_config()
train_celeba, eval_celeba, _ = datasets.get_dataset(config_celeba, uniform_dequantization=True, evaluation=True)

batch = next(iter(train_celeba))
img = batch['image']._numpy()
img = torch.tensor(img).permute(0, 3, 1, 2).to(config.device)
img = scaler(img)
bpd, z, nfe = likelihood_fn(score_model, img)
bpds.extend(bpd)
print(f"average bpd celeba: {torch.tensor(bpds).mean().item()}, NFE: {nfe}")


# from configs.ve import bedroom_ncsnpp_continuous as configs_bedr
# config_bedr = configs_bedr.get_config()
# train_bedr, eval_bedr, _ = datasets.get_dataset(config_bedr, uniform_dequantization=True, evaluation=True)
#
# for r in range(3):
#     batch = next(iter(train_bedr))
#     img = batch['image']._numpy()
#     img = torch.tensor(img).permute(0, 3, 1, 2).to(config.device)
#     img = scaler(img)
#     bpd, z, nfe = likelihood_fn(score_model, img)
#     bpds.extend(bpd)
#     print(f"average bpd bedr: {torch.tensor(bpds).mean().item()}, NFE: {nfe}")