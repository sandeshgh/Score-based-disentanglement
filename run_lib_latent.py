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
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
# import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, unet_autoenc #ncsnv2, ncsnpp
import losses
import sampling, sampling_latent, sampling_equal_energy, sampling_latent_factor, sampling_latent_multi, sampling_latent_multi_lat5
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from models.unet import BeatGANsUNetConfig
# class BeatGANsAutoencConfig(BeatGANsUNetConfig):
from models.unet_autoenc import BeatGANsAutoencConfig
import datasets
# import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def resize_image(batch, size):
  batch = batch.permute(0,3,1,2)
  batch = transforms.Resize(size)(batch)
  out = batch.permute(0,2,3,1)
  return out

def plot_samples(sample_dir, sample, step, class_n=None):
  this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
  tf.io.gfile.makedirs(this_sample_dir)
  nrow = int(np.sqrt(sample.shape[0]))
  image_grid = make_grid(sample, nrow, padding=2)
  sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
  # with tf.io.gfile.GFile(
  #     os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
  #   np.save(fout, sample)
  if class_n:
    image_name = "sample_"+str(class_n)
  else:
    image_name = "sample"

  with tf.io.gfile.GFile(
      os.path.join(this_sample_dir, image_name+".png"), "wb") as fout:
    save_image(image_grid, fout)

def plot_samples_conditional(sample_dir, sample_stack, step, class_n):
  this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
  tf.io.gfile.makedirs(this_sample_dir)
  for k in range(sample_stack.shape[-1]):
    sample = sample_stack[:,:,:,:,k]
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = make_grid(sample, nrow, padding=2)
    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    # with tf.io.gfile.GFile(
    #     os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    #   np.save(fout, sample)
    if len(class_n)>1:
        image_name = "sample_"+str(class_n[k])+".png"
    else:
      if k ==0:
        image_name = "sample.png"
      else:
        image_name = "sample_"+str(class_n)+".png"

    with tf.io.gfile.GFile(
        os.path.join(this_sample_dir, image_name), "wb") as fout:
      save_image(image_grid, fout)

def plot_samples_analysis(sample_dir, sample_stack, step, pos_index):
  this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step), str(pos_index))
  tf.io.gfile.makedirs(this_sample_dir)
  for k in range(sample_stack.shape[-1]):
    sample = sample_stack[:,:,:,:,k]
    nrow = int(np.sqrt(sample.shape[0]))
    image_grid = make_grid(sample, nrow, padding=2)
    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    # with tf.io.gfile.GFile(
    #     os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    #   np.save(fout, sample)
   
    image_name = "sample_"+str(k)+".png"
    
    with tf.io.gfile.GFile(
        os.path.join(this_sample_dir, image_name), "wb") as fout:
      save_image(image_grid, fout)


def plot_scores(sample_dir, sample, step):
  this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
  tf.io.gfile.makedirs(this_sample_dir)
  nrow = sample.shape[-1]
  sample = sample.permute(0,4,1,2,3)
  sample = torch.abs(sample.contiguous().view(sample.shape[0]*sample.shape[1], sample.shape[2], sample.shape[3], sample.shape[4]))
  sample = sample/torch.max(sample)
  image_grid = make_grid(sample, nrow, padding=2)
  sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
  
  with tf.io.gfile.GFile(
      os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    np.save(fout, sample)

  with tf.io.gfile.GFile(
      os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
    save_image(image_grid, fout)
 
def plot_samples_multivariate(sample_dir, sample, sample_stack, step):
  k = sample_stack.shape[-1]
  this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
  tf.io.gfile.makedirs(this_sample_dir)

  nrow = int(np.sqrt(sample.shape[0]))
  image_grid = make_grid(sample, nrow, padding=2)
  sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
  with tf.io.gfile.GFile(
      os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
    np.save(fout, sample)

  with tf.io.gfile.GFile(
      os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
    save_image(image_grid, fout)

  for i in range(k):
    sample_i = sample_stack[:,:,:,:,i]
    nrow = int(np.sqrt(sample_i.shape[0]))
    image_grid = make_grid(sample_i, nrow, padding=2)
    sample = np.clip(sample_i.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    img_name = "sample_"+str(i)
    with tf.io.gfile.GFile(
        os.path.join(this_sample_dir, img_name+".np"), "wb") as fout:
      np.save(fout, sample)

    with tf.io.gfile.GFile(
        os.path.join(this_sample_dir, img_name+".png"), "wb") as fout:
      save_image(image_grid, fout)

def scatter_plot_data(score_model, train_ds, config, scaler, viz_dir, epoch, n_ep=1000 ):
  score_model.eval()
  train_iter = iter(train_ds)
  codes = []

  
  for idx in range(n_ep):
    batch = next(train_iter)
  # batch_idx = 5
  # for dim_idx in range(8):
    # if idx >1000:
    #   break
    # print('batch index:', idx)
    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    if eval_batch.shape[2]!=config.data.image_size:
      eval_batch = resize_image(eval_batch, config.data.image_size )
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    # eval_batch = eval_batch[batch_idx].repeat(eval_batch.shape[0],1,1,1)
    eval_batch_model = scaler(eval_batch)
    with torch.no_grad():
      c = score_model.module.encode(eval_batch_model)
    codes.append(c.data.cpu().numpy())
  
  codes = np.concatenate(codes, axis=0)
  codes = convert_to_2D(codes)
  fig =plt.figure()
  ax = fig.add_subplot()
  ax.scatter(codes[:,0], codes[:,1])
  plt.xlim([-1,1])
  plt.ylim([-1,1])
  plt.savefig(viz_dir+'/scatter_'+str(epoch))
  plt.close()


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  if config.model.name == 'ddpm_latent_Adain' or config.model.name == 'ddpm_latent_Adain_multilatent' or config.model.name == 'ddpm_latent_Adain_multilatent_new' :
    
    # unet_config = BeatGANsUNetConfig(
    #   image_size = config.data.image_size,
    #   in_channels= config.data.num_channels,
    #   # model_channels=
    #   out_channels=config.data.num_channels
    # )
    conf = BeatGANsAutoencConfig()
    conf.image_size = config.data.image_size
    conf.in_channels = config.data.num_channels
    conf.out_channels = config.data.num_channels
    conf.embed_channels = config.model.latent_dim
    conf.enc_out_channels = config.model.latent_dim
    conf.resnet_two_cond = True
    # conf.model.name = 'ddpm_latent_Adain'
    score_model = mutils.create_model(conf, model_name = config.model.name, device= config.device)
  else:
    score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, config = config)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, config = config)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    if config.training.conditional_model == 'latent' or config.training.conditional_model == 'latent_variational' or config.training.conditional_model == 'latent_multi':
      sampling_fn = sampling_latent.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps) 
    elif config.training.conditional_model == 'latent_factor':
      sampling_fn = sampling_latent_factor.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps) 
    elif config.training.conditional_model == 'equal_energy':
      sampling_fn = sampling_equal_energy.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
    else:
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    if batch.shape[2]!=config.data.image_size:
      batch = resize_image(batch, config.data.image_size )
    batch_orig = batch.permute(0, 3, 1, 2)
    # plot_samples(sample_dir, batch_orig, 10000)
    batch = scaler(batch_orig)
    # plot_samples(sample_dir, batch, 10000)
    # Execute one training step
    loss = train_step_fn(state, batch)
    if (config.training.conditional_model == 'latent' and config.training.reconstruction_loss) or (config.training.conditional_model == 'latent_contrastive'):
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e, recons_loss: %.5e, score_loss: %.5e," % (step, loss[0].item(), loss[1].item(), loss[2].item() ))
        writer.add_scalar("training_loss", loss[0], step)
        writer.add_scalar("recon_loss", loss[1], step)
        writer.add_scalar("score_loss", loss[2], step)
    elif (config.training.conditional_model == 'latent' and config.training.regularization=='highvar') :
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e, var_loss: %.5e, score_loss: %.5e," % (step, loss[0].item(), loss[1].item(), loss[2].item() ))
        writer.add_scalar("training_loss", loss[0], step)
        writer.add_scalar("recon_loss", loss[1], step)
        writer.add_scalar("score_loss", loss[2], step)
    elif (config.training.conditional_model == 'latent_multi' and config.training.unconditional_loss) :
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e, uncond_score_loss: %.5e, score_loss: %.5e," % (step, loss[0].item(), loss[1].item(), loss[2].item() ))
        writer.add_scalar("training_loss", loss[0], step)
        writer.add_scalar("uncond_score_loss", loss[1], step)
        writer.add_scalar("score_loss", loss[2], step)
    else:
      if step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
        writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      if eval_batch.shape[2]!=config.data.image_size:
        eval_batch = resize_image(eval_batch, config.data.image_size)
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      if (config.training.conditional_model == 'latent' and config.training.reconstruction_loss) or (config.training.conditional_model == 'latent_contrastive'):
        logging.info("step: %d, eval_loss: %.5e, eval_rec_loss: %.5e, eval_score_loss: %.5e" % (step, eval_loss[0].item(), eval_loss[1].item(), eval_loss[2].item()))
        writer.add_scalar("eval_loss", eval_loss[0].item(), step)
        writer.add_scalar("eval_loss", eval_loss[1].item(), step)
        writer.add_scalar("eval_loss", eval_loss[2].item(), step)
      elif (config.training.conditional_model == 'latent' and config.training.regularization=='highvar'):
        logging.info("step: %d, eval_loss: %.5e, eval_var_loss: %.5e, eval_score_loss: %.5e" % (step, eval_loss[0].item(), eval_loss[1].item(), eval_loss[2].item()))
        writer.add_scalar("eval_loss0", eval_loss[0].item(), step)
        writer.add_scalar("eval_loss1", eval_loss[1].item(), step)
        writer.add_scalar("eval_loss2", eval_loss[2].item(), step)
      elif (config.training.conditional_model == 'latent_multi' and config.training.unconditional_loss):
        logging.info("step: %d, eval_loss: %.5e, uncond_score_loss: %.5e, eval_score_loss: %.5e" % (step, eval_loss[0].item(), eval_loss[1].item(), eval_loss[2].item()))
        writer.add_scalar("eval_loss0", eval_loss[0].item(), step)
        writer.add_scalar("eval_loss1", eval_loss[1].item(), step)
        writer.add_scalar("eval_loss2", eval_loss[2].item(), step)
      else:
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
        writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        if config.training.conditional_model == 'latent' or config.training.conditional_model == 'latent_variational' or config.training.conditional_model == 'latent_multi':
          latent_dim = config.model.latent_dim
          class_n = save_step%latent_dim
          latent = torch.zeros(latent_dim)
          latent[class_n] = 1
          
          sample, n = sampling_fn(score_model, latent)
        elif config.training.conditional_model == 'latent_factor':
          sample, n = sampling_fn(score_model, batch)
        else:
          sample, n = sampling_fn(score_model)
        

        plot_samples_conditional(sample_dir, torch.cat((batch_orig.unsqueeze(-1),sample.unsqueeze(-1)), dim=-1), step, [0,1])
        # scatter_plot_data(score_model, train_ds, config, scaler, viz_dir = sample_dir, epoch=step, n_ep=1000 )

        ema.restore(score_model.parameters())
        
        # plot_samples(sample_dir, sample, step, class_n) #either single or multivariate

        
def visualize(config, workdir, visualization_folder="viz"):
  # Create directory to eval_folder
  viz_dir = os.path.join(workdir, visualization_folder)
  tf.io.gfile.makedirs(viz_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled

  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  if config.training.conditional_model == 'latent':
      sampling_fn = sampling_latent.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  else:
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)


  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  ckpt = begin_ckpt
  # for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
  # Wait if the target checkpoint doesn't exist yet
  # waiting_message_printed = False
  # ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
  # while not tf.io.gfile.exists(ckpt_filename):
  #   if not waiting_message_printed:
  #     logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
  #     waiting_message_printed = True
  #   time.sleep(60)

  # Wait for 2 additional mins in case the file exists but is not ready for reading
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  try:
    state = restore_checkpoint(ckpt_path, state, device=config.device)
  except:
    time.sleep(60)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(120)
      state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  sample_dir = os.path.join(viz_dir, "sample")
  # score_dir = os.path.join(viz_dir, "score_plot")

  for class_i in range(3,16):
    for value_i in [0.1,0.5,0.7,0.9,1]:
      if config.training.conditional_model == 'latent':
        class_n = class_i
        val = value_i
        latent = torch.zeros(16)
        latent[class_n] = val

        # class_n_2 = 0
        # latent_2 = torch.zeros(3)
        # latent_2[class_n_2] = 0.7

        # latent_stack = torch.cat((latent.unsqueeze(-1), latent_2.unsqueeze(-1)), dim=-1)
        step = (torch.FloatTensor([ckpt]) + 0.1*torch.FloatTensor([class_n])).item()
        sample, n = sampling_fn(score_model, latent)
        plot_samples(sample_dir, sample, step, val)
        # plot_samples_conditional(sample_dir, sample, 12.0, [0.5,0.7])
      else:
        sample, n = sampling_fn(score_model)
        plot_samples(sample_dir, sample, ckpt)
  ema.restore(score_model.parameters())

def analyze(config, workdir, visualization_folder="viz"):
  # Create directory to eval_folder
  viz_dir = os.path.join(workdir, visualization_folder)
  tf.io.gfile.makedirs(viz_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled

  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  if config.training.conditional_model == 'latent' or config.training.conditional_model == 'latent_variational':
    analysis_fn = sampling_latent.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  if config.training.conditional_model == 'latent_factor':
    analysis_fn = sampling_latent_factor.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, mode = 'interpolate')
  
  # else:
  #   sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)


  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  ckpt = begin_ckpt
  # for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
  # Wait if the target checkpoint doesn't exist yet
  # waiting_message_printed = False
  # ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
  # while not tf.io.gfile.exists(ckpt_filename):
  #   if not waiting_message_printed:
  #     logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
  #     waiting_message_printed = True
  #   time.sleep(60)

  # Wait for 2 additional mins in case the file exists but is not ready for reading
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  try:
    state = restore_checkpoint(ckpt_path, state, device=config.device)
  except:
    time.sleep(60)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(120)
      state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  sample_dir = os.path.join(viz_dir, "analyze")
  # score_dir = os.path.join(viz_dir, "score_plot")
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  train_iter = iter(train_ds)
  for pos_index in range(7):
    print('Analysis .... of idx: ', pos_index)
  
  # step = (torch.FloatTensor([ckpt]) + 0.1*torch.FloatTensor([class_n])).item()
    step = ckpt
    
    # for i, batch in enumerate(train_iter):
    batch = next(train_iter)
    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    if eval_batch.shape[2]!=config.data.image_size:
      eval_batch = resize_image(eval_batch, config.data.image_size )
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    eval_batch_model = scaler(eval_batch)

    if config.training.conditional_model == 'latent_factor':
      pos_index = [5,6]
      samples = analysis_fn(score_model, eval_batch_model, pos_index)
      # plot_samples_analysis(sample_dir, samples, step, pos_index)

    else:
      sample1, sample2, sample3 = analysis_fn(score_model, eval_batch_model, latent_position=pos_index)
      samples = torch.cat((sample1.unsqueeze(-1), sample2.unsqueeze(-1), sample3.unsqueeze(-1)), dim =-1)
    samples=torch.cat((eval_batch.unsqueeze(-1),samples), dim=-1)
    plot_samples_analysis(sample_dir, samples, step, 'mix_'+str(pos_index))
        # plot_samples_conditional(sample_dir, sample, 12.0, [0.5,0.7])
  print("Done!")

def analyze_close(config, workdir, visualization_folder="viz"):
  # Create directory to eval_folder
  viz_dir = os.path.join(workdir, visualization_folder)
  tf.io.gfile.makedirs(viz_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  if config.model.name == 'ddpm_latent_Adain' or config.model.name == 'ddpm_latent_Adain_multilatent' or config.model.name == 'ddpm_latent_Adain_multilatent_new':
    
    # unet_config = BeatGANsUNetConfig(
    #   image_size = config.data.image_size,
    #   in_channels= config.data.num_channels,
    #   # model_channels=
    #   out_channels=config.data.num_channels
    # )
    conf = BeatGANsAutoencConfig()
    conf.image_size = config.data.image_size
    conf.in_channels = config.data.num_channels
    conf.out_channels = config.data.num_channels
    conf.embed_channels = config.model.latent_dim
    conf.enc_out_channels = config.model.latent_dim
    conf.resnet_two_cond = True
    # conf.model.name = 'ddpm_latent_Adain'
    score_model = mutils.create_model(conf, model_name = config.model.name, device= config.device)
  else:
    score_model = mutils.create_model(config)
  
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  # analysis_type 

  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  analysis_type = config.eval.analysis_type
  if config.training.conditional_model == 'latent' or config.training.conditional_model == 'latent_variational':
    analysis_fn = sampling_latent.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, type=analysis_type)
  elif config.training.conditional_model == 'latent_multi':
    if config.model.name == 'ddpm_latent_Adain_multilatent_new':
      analysis_fn = sampling_latent_multi_lat5.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, type=analysis_type)
    else:
      analysis_fn = sampling_latent_multi.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, type=analysis_type)
  if config.training.conditional_model == 'latent_factor':
    analysis_fn = sampling_latent_factor.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, mode = 'interpolate')
  
  # else:
  #   sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  
  # train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                             uniform_dequantization=config.data.uniform_dequantization,
  #                                             evaluation=True)


  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  ckpt = begin_ckpt
  

  # Wait for 2 additional mins in case the file exists but is not ready for reading
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  try:
    state = restore_checkpoint(ckpt_path, state, device=config.device)
  except:
    time.sleep(60)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(120)
      state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  sample_dir = viz_dir
  print('Sample dir', sample_dir)
  # score_dir = os.path.join(viz_dir, "score_plot")
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  train_iter = iter(train_ds)

  # for pos_index in range(7):
    # print('Analysis .... of idx: ', pos_index)
  
  # step = (torch.FloatTensor([ckpt]) + 0.1*torch.FloatTensor([class_n])).item()
  step = ckpt
  
  # for i, batch in enumerate(train_iter):
  batch = next(train_iter)
  batch_idx = 5
  dim_idx = 0
  mode = analysis_type#'mix'#'project_one_hot'#'test_dim'#'project_one_hot'
  for batch_idx in range(6,9):
    print('dim_idx:', dim_idx)
    print('batch_idx:', batch_idx)
    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    if eval_batch.shape[2]!=config.data.image_size:
      eval_batch = resize_image(eval_batch, config.data.image_size )
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    if mode == 'mix':
      # if it's a mix, then use two images and repeat half half to make up the full batch size
      eval_batch_1 = eval_batch[batch_idx].repeat(eval_batch.shape[0]//2,1,1,1)
      eval_batch_2 = eval_batch[batch_idx+1].repeat(eval_batch.shape[0]//2,1,1,1)
      eval_batch = torch.cat((eval_batch_1, eval_batch_2), dim = 0)
    else:
      eval_batch = eval_batch[batch_idx].repeat(eval_batch.shape[0],1,1,1)
    eval_batch_model = scaler(eval_batch)

    if config.training.conditional_model == 'latent_factor':
      pos_index = [5,6]
      samples = analysis_fn(score_model, eval_batch_model, pos_index)
      # plot_samples_analysis(sample_dir, samples, step, pos_index)

    else:
      close_mode = 'project_one_hot'#'test_dim'#'project_one_hot'
      sampler = 'pc'
      print("type:", mode)
      samples = analysis_fn(score_model, eval_batch_model, mode=close_mode, dim_idx=dim_idx, sampler=sampler, viz_dir=sample_dir, batch_idx = batch_idx)
      # samples = torch.cat((sample1.unsqueeze(-1), sample2.unsqueeze(-1), sample3.unsqueeze(-1)), dim =-1)
    if config.training.conditional_model == 'latent_multi':
      if config.model.name == 'ddpm_latent_Adain_multilatent_new':
        samples_a=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,0,:]), dim=-1)
        samples_b=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,1,:]), dim=-1)
        samples_c=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,2,:]), dim=-1)
        samples_d=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,3,:]), dim=-1)
        samples_e=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,4,:]), dim=-1)
        samples_sum = torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,5,:]), dim=-1)
        plot_samples_analysis(sample_dir, samples_a, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_a')
        plot_samples_analysis(sample_dir, samples_b, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_b')
        plot_samples_analysis(sample_dir, samples_c, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_c')
        plot_samples_analysis(sample_dir, samples_d, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_d')
        plot_samples_analysis(sample_dir, samples_e, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_e')
        plot_samples_analysis(sample_dir, samples_sum, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_sum')
      else:
        samples_a=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,0,:]), dim=-1)
        samples_b=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,1,:]), dim=-1)
        samples_c=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,2,:]), dim=-1)
        samples_sum = torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,3,:]), dim=-1)
        plot_samples_analysis(sample_dir, samples_a, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_a')
        plot_samples_analysis(sample_dir, samples_b, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_b')
        plot_samples_analysis(sample_dir, samples_c, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_c')
        plot_samples_analysis(sample_dir, samples_sum, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_sum')
    else:
      samples=torch.cat((eval_batch.unsqueeze(-1),samples), dim=-1)
      plot_samples_analysis(sample_dir, samples, step, analysis_type+'_'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx))
        # plot_samples_conditional(sample_dir, sample, 12.0, [0.5,0.7])
  # print("Done!")

def visual(config, workdir, visualization_folder="viz"):
  # Create directory to eval_folder
  viz_dir = os.path.join(workdir, visualization_folder)
  tf.io.gfile.makedirs(viz_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  if config.model.name == 'ddpm_latent_Adain' or config.model.name == 'ddpm_latent_Adain_multilatent' or config.model.name == 'ddpm_latent_Adain_multilatent_new':
    
    # unet_config = BeatGANsUNetConfig(
    #   image_size = config.data.image_size,
    #   in_channels= config.data.num_channels,
    #   # model_channels=
    #   out_channels=config.data.num_channels
    # )
    conf = BeatGANsAutoencConfig()
    conf.image_size = config.data.image_size
    conf.in_channels = config.data.num_channels
    conf.out_channels = config.data.num_channels
    conf.embed_channels = config.model.latent_dim
    conf.enc_out_channels = config.model.latent_dim
    conf.resnet_two_cond = True
    # conf.model.name = 'ddpm_latent_Adain'
    score_model = mutils.create_model(conf, model_name = config.model.name, device= config.device)
  else:
    score_model = mutils.create_model(config)
  
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  # analysis_type 

  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  analysis_type = config.eval.analysis_type
  if config.training.conditional_model == 'latent' or config.training.conditional_model == 'latent_variational':
    raise NotImplementedError
    # analysis_fn = sampling_latent.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, type=analysis_type)
  elif config.training.conditional_model == 'latent_multi':
    analysis_fn = sampling_latent_multi.get_visual_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  if config.training.conditional_model == 'latent_factor':
    raise NotImplementedError
    # analysis_fn = sampling_latent_factor.get_analysis_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, mode = 'interpolate')


  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  ckpt = begin_ckpt
  

  # Wait for 2 additional mins in case the file exists but is not ready for reading
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  try:
    state = restore_checkpoint(ckpt_path, state, device=config.device)
  except:
    time.sleep(60)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(120)
      state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  sample_dir = viz_dir
  print('Sample dir', sample_dir)
  # score_dir = os.path.join(viz_dir, "score_plot")
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  train_iter = iter(train_ds)

  step = ckpt
  
  # for i, batch in enumerate(train_iter):
  
  batch_idx = 5
  dim_idx = 0
  mode = 'mix'#'project_one_hot'#'test_dim'#'project_one_hot'
  for batch_idx in range(5,10):
    batch = next(train_iter)
    print('dim_idx:', dim_idx)
    print('batch_idx:', batch_idx)
    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    if eval_batch.shape[2]!=config.data.image_size:
      eval_batch = resize_image(eval_batch, config.data.image_size )
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    
    eval_batch_model = scaler(eval_batch)

    if config.training.conditional_model == 'latent_factor':
      pos_index = [5,6]
      samples = analysis_fn(score_model, eval_batch_model, pos_index)
      # plot_samples_analysis(sample_dir, samples, step, pos_index)

    else:
      sampler = 'pc'
      print(mode)
      samples = analysis_fn(score_model, eval_batch_model, viz_dir=sample_dir)
      # samples = torch.cat((sample1.unsqueeze(-1), sample2.unsqueeze(-1), sample3.unsqueeze(-1)), dim =-1)
    if config.training.conditional_model == 'latent_multi':
      # samples_a=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,0,:]), dim=-1)
      samples_a=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,0,:].repeat(1,3,1,1,1)), dim=-1) #samples[:,:,:,:,0,:]#
      samples_b=torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,1,:].repeat(1,3,1,1,1)), dim=-1) #samples[:,:,:,:,1,:]#
      samples_c=samples[:,:,:,:,2,:]#torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,2,:]), dim=-1) #samples[:,:,:,:,2,:]#
      # samples_sum = torch.cat((eval_batch.unsqueeze(-1),samples[:,:,:,:,3,:]), dim=-1)
      plot_samples_analysis(sample_dir, samples_a, step, 'gradplot_'+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_a')
      plot_samples_analysis(sample_dir, samples_b, step, 'gradplot_'+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_b')
      plot_samples_analysis(sample_dir, samples_c, step, 'gradplot_'+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_c')
      # plot_samples_analysis(sample_dir, samples_sum, step, analysis_type+'__'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx)+'_sum')
    else:
      samples=torch.cat((eval_batch.unsqueeze(-1),samples), dim=-1)
      plot_samples_analysis(sample_dir, samples, step, analysis_type+'_'+sampler+mode+'dim_'+str(dim_idx)+'_batch_'+str(batch_idx))

  print("Done!")

def rot(U,V):
  # this converts U to V
  W=np.cross(U,V)
  A=np.array([U,W,np.cross(U,W)]).T
  B=np.array([V,W,np.cross(V,W)]).T
  return np.matmul(B,np.linalg.inv(A))

def convert_to_2D(x):
  x = x[:,:3]
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
  


      
  
def plot_data(config, workdir, visualization_folder="viz"):
  # Create directory to eval_folder
  viz_dir = os.path.join(workdir, visualization_folder)
  tf.io.gfile.makedirs(viz_dir)
  print('Plot dir:', viz_dir)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  ckpt = begin_ckpt
  

  # Wait for 2 additional mins in case the file exists but is not ready for reading
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  try:
    state = restore_checkpoint(ckpt_path, state, device=config.device)
  except:
    time.sleep(60)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(120)
      state = restore_checkpoint(ckpt_path, state, device=config.device)
  # ema.copy_to(score_model.parameters())
  score_model = state['model']
  score_model.eval()

  

  # sample_dir = os.path.join(viz_dir, "analyze")
  # print('Sample dir', sample_dir)
  # score_dir = os.path.join(viz_dir, "score_plot")
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  train_iter = iter(train_ds)

  # for pos_index in range(7):
    # print('Analysis .... of idx: ', pos_index)
  
  # step = (torch.FloatTensor([ckpt]) + 0.1*torch.FloatTensor([class_n])).item()
  # step = ckpt
  codes = []
  # print('dataloader length', train_iter)

  
  for idx in range(10000):
    batch = next(train_iter)
  # batch_idx = 5
  # for dim_idx in range(8):
    # if idx >500:
    #   break
    # print('batch index:', idx)
    eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    if eval_batch.shape[2]!=config.data.image_size:
      eval_batch = resize_image(eval_batch, config.data.image_size )
    eval_batch = eval_batch.permute(0, 3, 1, 2)
    # eval_batch = eval_batch[batch_idx].repeat(eval_batch.shape[0],1,1,1)
    eval_batch_model = scaler(eval_batch)
    c = score_model.module.encode(eval_batch_model)
    codes.append(c.data.cpu().numpy())
  
  codes = np.concatenate(codes, axis=0)
  codes = convert_to_2D(codes)
  fig =plt.figure()
  ax = fig.add_subplot()
  ax.scatter(codes[:,0], codes[:,1])



  lines = np.eye(3)
  lines = convert_to_2D(lines)
  ax.plot(lines[:,0], lines[:,1], 'r', linestyle='-')

  plt.xlim([np.amin(codes[:,0]),np.amax(codes[:,0])])
  plt.ylim([np.amin(codes[:,1]),np.amax(codes[:,1])])
  plt.savefig(viz_dir+'/scatter')
  plt.close()

def data_plot_proj_2D(codes, viz_dir, idx=0):
  codes = convert_to_2D(codes)
  fig =plt.figure()
  ax = fig.add_subplot()
  ax.scatter(codes[:,0], codes[:,1])

  lines = np.eye(3)
  lines = convert_to_2D(lines)
  ax.plot(lines[:,0], lines[:,1], 'r', linestyle='-')

  plt.xlim([-1,1])
  plt.ylim([-1,1])
  plt.savefig(viz_dir+'/scatter'+str(idx))
  plt.close()
       

# def evaluate(config,
#              workdir,
#              eval_folder="eval"):
#   """Evaluate trained models.

#   Args:
#     config: Configuration to use.
#     workdir: Working directory for checkpoints.
#     eval_folder: The subfolder for storing evaluation results. Default to
#       "eval".
#   """
#   # Create directory to eval_folder
#   eval_dir = os.path.join(workdir, eval_folder)
#   tf.io.gfile.makedirs(eval_dir)

#   # Build data pipeline
#   train_ds, eval_ds, _ = datasets.get_dataset(config,
#                                               uniform_dequantization=config.data.uniform_dequantization,
#                                               evaluation=True)

#   # Create data normalizer and its inverse
#   scaler = datasets.get_data_scaler(config)
#   inverse_scaler = datasets.get_data_inverse_scaler(config)

#   # Initialize model
#   score_model = mutils.create_model(config)
#   optimizer = losses.get_optimizer(config, score_model.parameters())
#   ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
#   state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

#   checkpoint_dir = os.path.join(workdir, "checkpoints")

#   # Setup SDEs
#   if config.training.sde.lower() == 'vpsde':
#     sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
#     sampling_eps = 1e-3
#   elif config.training.sde.lower() == 'subvpsde':
#     sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
#     sampling_eps = 1e-3
#   elif config.training.sde.lower() == 'vesde':
#     sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
#     sampling_eps = 1e-5
#   else:
#     raise NotImplementedError(f"SDE {config.training.sde} unknown.")

#   # Create the one-step evaluation function when loss computation is enabled
#   if config.eval.enable_loss:
#     optimize_fn = losses.optimization_manager(config)
#     continuous = config.training.continuous
#     likelihood_weighting = config.training.likelihood_weighting

#     reduce_mean = config.training.reduce_mean
#     eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
#                                    reduce_mean=reduce_mean,
#                                    continuous=continuous,
#                                    likelihood_weighting=likelihood_weighting)


#   # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
#   train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
#                                                       uniform_dequantization=True, evaluation=True)
#   if config.eval.bpd_dataset.lower() == 'train':
#     ds_bpd = train_ds_bpd
#     bpd_num_repeats = 1
#   elif config.eval.bpd_dataset.lower() == 'test':
#     # Go over the dataset 5 times when computing likelihood on the test dataset
#     ds_bpd = eval_ds_bpd
#     bpd_num_repeats = 5
#   else:
#     raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

#   # Build the likelihood computation function when likelihood is enabled
#   if config.eval.enable_bpd:
#     likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

#   # Build the sampling function when sampling is enabled
#   if config.eval.enable_sampling:
#     sampling_shape = (config.eval.batch_size,
#                       config.data.num_channels,
#                       config.data.image_size, config.data.image_size)
#     sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

#   # Use inceptionV3 for images with resolution higher than 256.
#   inceptionv3 = config.data.image_size >= 256
#   inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

#   begin_ckpt = config.eval.begin_ckpt
#   logging.info("begin checkpoint: %d" % (begin_ckpt,))
#   for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
#     # Wait if the target checkpoint doesn't exist yet
#     waiting_message_printed = False
#     ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
#     while not tf.io.gfile.exists(ckpt_filename):
#       if not waiting_message_printed:
#         logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
#         waiting_message_printed = True
#       time.sleep(60)

#     # Wait for 2 additional mins in case the file exists but is not ready for reading
#     ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
#     try:
#       state = restore_checkpoint(ckpt_path, state, device=config.device)
#     except:
#       time.sleep(60)
#       try:
#         state = restore_checkpoint(ckpt_path, state, device=config.device)
#       except:
#         time.sleep(120)
#         state = restore_checkpoint(ckpt_path, state, device=config.device)
#     ema.copy_to(score_model.parameters())
#     # Compute the loss function on the full evaluation dataset if loss computation is enabled
#     if config.eval.enable_loss:
#       all_losses = []
#       eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
#       for i, batch in enumerate(eval_iter):
#         eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
#         eval_batch = eval_batch.permute(0, 3, 1, 2)
#         eval_batch = scaler(eval_batch)
#         eval_loss = eval_step(state, eval_batch)
#         all_losses.append(eval_loss.item())
#         if (i + 1) % 1000 == 0:
#           logging.info("Finished %dth step loss evaluation" % (i + 1))

#       # Save loss values to disk or Google Cloud Storage
#       all_losses = np.asarray(all_losses)
#       with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
#         io_buffer = io.BytesIO()
#         np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
#         fout.write(io_buffer.getvalue())

#     # Compute log-likelihoods (bits/dim) if enabled
#     if config.eval.enable_bpd:
#       bpds = []
#       for repeat in range(bpd_num_repeats):
#         bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
#         for batch_id in range(len(ds_bpd)):
#           batch = next(bpd_iter)
#           eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
#           eval_batch = eval_batch.permute(0, 3, 1, 2)
#           eval_batch = scaler(eval_batch)
#           bpd = likelihood_fn(score_model, eval_batch)[0]
#           bpd = bpd.detach().cpu().numpy().reshape(-1)
#           bpds.extend(bpd)
#           logging.info(
#             "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
#           bpd_round_id = batch_id + len(ds_bpd) * repeat
#           # Save bits/dim to disk or Google Cloud Storage
#           with tf.io.gfile.GFile(os.path.join(eval_dir,
#                                               f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
#                                  "wb") as fout:
#             io_buffer = io.BytesIO()
#             np.savez_compressed(io_buffer, bpd)
#             fout.write(io_buffer.getvalue())

#     # Generate samples and compute IS/FID/KID when enabled
#     if config.eval.enable_sampling:
#       num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
#       for r in range(num_sampling_rounds):
#         logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

#         # Directory to save samples. Different for each host to avoid writing conflicts
#         this_sample_dir = os.path.join(
#           eval_dir, f"ckpt_{ckpt}")
#         tf.io.gfile.makedirs(this_sample_dir)
#         samples, n = sampling_fn(score_model)
#         samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
#         samples = samples.reshape(
#           (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
#         # Write samples to disk or Google Cloud Storage
#         with tf.io.gfile.GFile(
#             os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
#           io_buffer = io.BytesIO()
#           np.savez_compressed(io_buffer, samples=samples)
#           fout.write(io_buffer.getvalue())

#         # Force garbage collection before calling TensorFlow code for Inception network
#         gc.collect()
#         latents = evaluation.run_inception_distributed(samples, inception_model,
#                                                        inceptionv3=inceptionv3)
#         # Force garbage collection again before returning to JAX code
#         gc.collect()
#         # Save latent represents of the Inception network to disk or Google Cloud Storage
#         with tf.io.gfile.GFile(
#             os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
#           io_buffer = io.BytesIO()
#           np.savez_compressed(
#             io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
#           fout.write(io_buffer.getvalue())

#       # Compute inception scores, FIDs and KIDs.
#       # Load all statistics that have been previously computed and saved for each host
#       all_logits = []
#       all_pools = []
#       this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
#       stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
#       for stat_file in stats:
#         with tf.io.gfile.GFile(stat_file, "rb") as fin:
#           stat = np.load(fin)
#           if not inceptionv3:
#             all_logits.append(stat["logits"])
#           all_pools.append(stat["pool_3"])

#       if not inceptionv3:
#         all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
#       all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

#       # Load pre-computed dataset statistics.
#       data_stats = evaluation.load_dataset_stats(config)
#       data_pools = data_stats["pool_3"]

#       # Compute FID/KID/IS on all samples together.
#       if not inceptionv3:
#         inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
#       else:
#         inception_score = -1

#       fid = tfgan.eval.frechet_classifier_distance_from_activations(
#         data_pools, all_pools)
#       # Hack to get tfgan KID work for eager execution.
#       tf_data_pools = tf.convert_to_tensor(data_pools)
#       tf_all_pools = tf.convert_to_tensor(all_pools)
#       kid = tfgan.eval.kernel_classifier_distance_from_activations(
#         tf_data_pools, tf_all_pools).numpy()
#       del tf_data_pools, tf_all_pools

#       logging.info(
#         "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
#           ckpt, inception_score, fid, kid))

#       with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
#                              "wb") as f:
#         io_buffer = io.BytesIO()
#         np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
#         f.write(io_buffer.getvalue())