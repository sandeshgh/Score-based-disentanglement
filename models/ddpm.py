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
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
from sklearn.metrics import SCORERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torchvision
from scipy.fft import dct
import numpy as np
import math

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

def swish(x):
  return x * torch.sigmoid(x)


class up_conv_block(nn.Module):
  def __init__(self, in_ch , out_ch, kernel_size = 3):
    super(up_conv_block, self).__init__()
    self.act = nn.ELU()
    self.upconv = nn.Sequential(
      nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3),
      nn.BatchNorm2d(16),
      nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3),
      self.act,

    )
  def forward(self, x):
    return self.upconv(x)


class LinearConv(nn.Module):
  def __init__(self, in_dim, img_size, out_channel = 1, type = 'sigmoid'):
    super().__init__()
    self.act = nn.ELU()
    self.linear = nn.Sequential(
      nn.Linear(in_dim, 16),
      self.act,
      nn.Linear(16, 64),
      self.act,
      nn.Linear(64, 256),

    )
    
    self.upconv1 = up_conv_block(1, 16)
    self.upconv2 = up_conv_block(16, 16)
    self.upconv3 = up_conv_block(16, 16)
    self.upconv4 = up_conv_block(16, 16)
    self.upconv5 = up_conv_block(16, 16)
    self.upconv6 = up_conv_block(16, 16)
    self.conv = nn.Conv2d(16, out_channel, kernel_size=3, padding=1)
    self.softmax = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()
    self.type = type


  def forward(self, x):
    out = self.linear(x)
    b, hw = out.shape
    h = np.sqrt(hw).astype(int)
    out = out.unsqueeze(1).view(b,1,h,h)
    
    out = self.upconv1(out)
    out = self.upconv2(out)
    out = self.upconv3(out)
    out = self.upconv4(out)
    # out = self.upconv5(out)
    # out = self.upconv6(out)
    out = self.conv(out)
    if self.type == 'softmax':
      B, C, H, W = out.shape
      out = self.softmax(out.view(B, C, H*W))
      out = out.view(B, C, H, W)
    # else:

    #   out = self.sigmoid(out)
    return out



class CondResBlock(nn.Module):
  def __init__(self, downsample=True, rescale=True, filters=8, latent_dim=8, im_size=64, latent_grid=False):
    super(CondResBlock, self).__init__()

    self.filters = filters
    self.latent_dim = latent_dim
    self.im_size = im_size
    self.downsample = downsample
    self.latent_grid = latent_grid

    if filters <= 128:
      self.bn1 = nn.InstanceNorm2d(filters, affine=False)
    else:
      self.bn1 = nn.GroupNorm(32, filters, affine=False)

    self.conv1 = nn.Conv2d(32, filters, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
    self.expand_conv = nn.Conv2d(filters, 32, kernel_size=3, stride=1, padding=1)

    if filters <= 128:
      self.bn2 = nn.InstanceNorm2d(filters, affine=False)
    else:
      self.bn2 = nn.GroupNorm(32, filters, affine=False)


    torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

    # Upscale to an mask of image
    self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
    self.latent_fc2 = nn.Linear(latent_dim, 2*filters)

    # Upscale to mask of image
    if downsample:
      if rescale:
        self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
      else:
        self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

      self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

  def forward(self, x, latent):
    x_orig = x

    latent_1 = self.latent_fc1(latent)
    latent_2 = self.latent_fc2(latent)

    gain = latent_1[:, :self.filters, None, None]
    bias = latent_1[:, self.filters:, None, None]

    gain2 = latent_2[:, :self.filters, None, None]
    bias2 = latent_2[:, self.filters:, None, None]

    x = self.conv1(x)
    x = gain * x + bias
    x = swish(x)


    x = self.conv2(x)
    x = gain2 * x + bias2
    x = swish(x)
    x = self.expand_conv(x)

    x_out = x_orig + x

    if self.downsample:
      x_out = swish(self.conv_downsample(x_out))
      x_out = self.avg_pool(x_out)

    return x_out




class CondResBlockNoLatent(nn.Module):
  def __init__(self, downsample=True, rescale=True, filters=64, upsample=False):
    super(CondResBlockNoLatent, self).__init__()

    self.filters = filters
    self.downsample = downsample

    if filters <= 128:
      self.bn1 = nn.GroupNorm(int(32  * filters / 128), filters, affine=True)
    else:
      self.bn1 = nn.GroupNorm(32, filters, affine=False)

    self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

    if filters <= 128:
      self.bn2 = nn.GroupNorm(int(32 * filters / 128), filters, affine=True)
    else:
      self.bn2 = nn.GroupNorm(32, filters, affine=True)

    self.upsample = upsample
    self.upsample_module = nn.Upsample(scale_factor=2)
    # Upscale to mask of image
    if downsample:
      if rescale:
        self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
      else:
        self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

      self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    if upsample:
      self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    x_orig = x


    x = self.conv1(x)
    x = swish(x)

    x = self.conv2(x)
    x = swish(x)

    x_out = x_orig + x

    if self.upsample:
      x_out = self.upsample_module(x_out)
      x_out = swish(self.conv_downsample(x_out))

    if self.downsample:
      x_out = swish(self.conv_downsample(x_out))
      x_out = self.avg_pool(x_out)

    return x_out



@utils.register_model(name='ddpm')
class DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    channels = config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

  def forward(self, x, labels):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

@utils.register_model(name='ddpm_compositional')
class DDPM_compositional(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.score_model1 = DDPM(config)
    self.score_model2 = DDPM(config)
  def forward(self, x, labels):
    score1 = self.score_model1(x, labels)
    score2 = self.score_model2(x, labels)
    return score1, score2

@utils.register_model(name='ddpm_compositional_latent')
class DDPM_compositional_latent(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    # this is input channel to score model. It has been changed to 32 to incorporate the conditional
    # resnet block's output dim
    channels = 32 #config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, 3, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

    ## for the encoding part####------
    filter_dim = 64
    latent_dim = 64
    latent_dim_expand = 16
    self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
    self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_fc1 = nn.Linear(filter_dim, int(filter_dim/2))
    self.embed_fc2 = nn.Linear(int(filter_dim/2), latent_dim_expand)


    self.layer_encode = CondResBlock(rescale=False, downsample=False)
    self.layer1 = CondResBlock(rescale=False, downsample=False)
    self.layer2 = CondResBlock(downsample=False)
    self.begin_conv = nn.Sequential(nn.Conv2d(3, filter_dim, kernel_size = 3, stride=1, padding=1),
                                        nn.Conv2d(filter_dim, int(filter_dim/2), 3, stride=1, padding=1))
    self.upconv = nn.Conv2d(in_channels=64, out_channels=filter_dim, kernel_size=3, padding=1)

  def encode(self, x):
    x = self.embed_conv1(x)
    x = F.relu(x)
    x = self.embed_layer1(x)
    x = self.embed_layer2(x)
    x = self.embed_layer3(x)

        # if self.recurrent_model:

        #     #if self.dataset != "clevr":
        #     x = self.embed_layer4(x)

        #     s = x.size()
        #     x = x.view(s[0], s[1], -1)
        #     x = x.permute(0, 2, 1).contiguous()
        #     pos_embed = self.pos_embedding

        #     # x = x + pos_embed[None, :, :]
        #     h = torch.zeros(1, im.size(0), self.filter_dim).to(x.device), torch.zeros(1, im.size(0), self.filter_dim).to(x.device)
        #     outputs = []

        #     for i in range(self.components):
        #         (sx, cx) = h

        #         cx = cx.permute(1, 0, 2).contiguous()
        #         context = torch.cat([cx.expand(-1, x.size(1), -1), x], dim=-1)
        #         at_wt = self.at_fc2(F.relu(self.at_fc1(context)))
        #         at_wt = F.softmax(at_wt, dim=1)
        #         inp = (at_wt * context).sum(dim=1, keepdim=True)
        #         inp = self.map_embed(inp)
        #         inp = inp.permute(1, 0, 2).contiguous()

        #         output, h = self.lstm(inp, h)
        #         outputs.append(output)

        #     output = torch.cat(outputs, dim=0)
        #     output = output.permute(1, 0, 2).contiguous()
        #     output = self.embed_fc2(output)
        #     s = output.size()
        #     output = output.view(s[0], -1)
        # else:
    x = x.mean(dim=2).mean(dim=2)

    x = x.view(x.size(0), -1)
    # output = self.embed_fc1(x)
    x = F.relu(self.embed_fc1(x))
    output = self.embed_fc2(x)
    output = output.view(output.shape[0], 2, -1)
    return output

  
  def decode(self, x, labels, latent):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    h = self.begin_conv(h)

    h = self.layer_encode(h, latent) # 64, 64, 64

    h = self.layer1(h, latent)  # 64,64,64
    h = self.layer2(h, latent)  # 64,64,64
    # h = self.upconv(h)  # 128,64,64


    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  def forward(self, x_tilde, labels, x):
    latent = self.encode(x)
    score1 = self.decode(x_tilde,labels, latent[:,0])
    score2 = self.decode(x_tilde,labels, latent[:,1])
    return score1, score2, latent

@utils.register_model(name='ddpm_compositional_conditional')
class DDPM_compositional_conditional(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    # this is input channel to score model. It has been changed to 32 to incorporate the conditional
    # resnet block's output dim
    channels = 32 #config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, 3, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

    ## for the encoding part####------
    filter_dim = 64
    latent_dim = 64
    latent_dim_expand = 16
    self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
    self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_fc1 = nn.Linear(filter_dim, int(filter_dim/2))
    self.embed_fc2 = nn.Linear(int(filter_dim/2), latent_dim_expand)


    self.layer_encode = CondResBlock(rescale=False, downsample=False, latent_dim=10, filters=10)
    self.layer1 = CondResBlock(rescale=False, downsample=False, latent_dim=10, filters=10)
    self.layer2 = CondResBlock(downsample=False, latent_dim=10, filters=10)
    self.begin_conv = nn.Sequential(nn.Conv2d(3, filter_dim, kernel_size = 3, stride=1, padding=1),
                                        nn.Conv2d(filter_dim, int(filter_dim/2), 3, stride=1, padding=1))
    self.upconv = nn.Conv2d(in_channels=64, out_channels=filter_dim, kernel_size=3, padding=1)
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000, 10)

    self.ddpm = DDPM(config=config)


  def encode(self, x):
    out = self.encoder(x)
    out = self.encoder_linear(out)
    out = nn.Softmax(dim=-1)(out)
    return out

  def decode(self, x, labels):
    return self.ddpm(x,labels)

  
  def decode_conditional(self, x, labels, latent):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    h = self.begin_conv(h)

    h = self.layer_encode(h, latent) # 64, 64, 64

    h = self.layer1(h, latent)  # 64,64,64
    h = self.layer2(h, latent)  # b,32,32,32
    # h = self.upconv(h)  # 128,64,64


    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  def get_latent(self, prob):
    latent = torch.zeros_like(prob)
    m = prob.shape[0]
    i = torch.arange(m)
    if torch.rand(1) > 0.5:
      val, idx = torch.max(prob, dim=-1)
    else:
      idx = torch.randint(10,(m,)).to(prob.device)
      val = torch.gather(prob, dim = 1, index = idx.unsqueeze(-1))
    latent[i,idx] = 1.0
    return latent, val
  
  def forward_generate(self, x_tilde, labels, latent_factor):
    score1 = self.decode(x_tilde,labels)
    score2 = self.decode_conditional(x_tilde,labels, latent_factor)
    return score1, score2

  def forward(self, x_tilde, labels, x):
    latent_prob = self.encode(x)
    latent, prob_max = self.get_latent(latent_prob)
    score1 = self.decode(x_tilde,labels)
    score2 = self.decode_conditional(x_tilde,labels, latent)
    out = (score1, score2, prob_max.view(prob_max.shape[0],1,1,1), latent_prob.view(latent_prob.shape[0],latent_prob.shape[1],1,1))
    return out

@utils.register_model(name='ddpm_compositional_equal_energy')
class DDPM_compositional_equal_energy(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    # this is input channel to score model. It has been changed to 32 to incorporate the conditional
    # resnet block's output dim
    channels = 32 #config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, 3, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

    ## for the encoding part####------
    filter_dim = 64
    latent_dim = 64
    encode_vector_length = 3
    latent_dim_expand = 16
    self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
    self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_fc1 = nn.Linear(filter_dim, int(filter_dim/2))
    self.embed_fc2 = nn.Linear(int(filter_dim/2), latent_dim_expand)


    self.layer_encode = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer1 = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer2 = CondResBlock(downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.begin_conv = nn.Sequential(nn.Conv2d(3, filter_dim, kernel_size = 3, stride=1, padding=1),
                                        nn.Conv2d(filter_dim, int(filter_dim/2), 3, stride=1, padding=1))
    self.upconv = nn.Conv2d(in_channels=64, out_channels=filter_dim, kernel_size=3, padding=1)
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000, encode_vector_length)

    self.ddpm = DDPM(config=config)


  def encode(self, x):
    out = self.encoder(x)
    out = self.encoder_linear(out)
    out = nn.Softmax(dim=-1)(out)
    return out

  def decode(self, x, labels):
    return self.ddpm(x,labels)

  
  def decode_conditional(self, x, labels, latent):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    h = self.begin_conv(h)

    h = self.layer_encode(h, latent) # 64, 64, 64

    h = self.layer1(h, latent)  # 64,64,64
    h = self.layer2(h, latent)  # b,32,32,32
    # h = self.upconv(h)  # 128,64,64


    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  def get_latent(self, prob):
    m = prob.shape[0]
    I = torch.diag_embed(prob)
    # latent = torch.mv(I, prob)

    indx = torch.randint(low=0, high=3, size=(m,2))
    # index = [arange(m), indx, :]
    out = I[torch.arange(m).unsqueeze(1), indx]
    
    return out[:,0], out[:,1]
  
  def forward_generate(self, x_tilde, labels, latent_factor):
    # score1 = self.decode(x_tilde,labels)
    score = self.decode_conditional(x_tilde,labels, latent_factor)
    return score

  def forward(self, x_tilde, labels, x):
    latent_prob = self.encode(x)
    latent1, latent2 = self.get_latent(latent_prob)

    # score1 = self.decode(x_tilde,labels)

    score1 = self.decode_conditional(x_tilde,labels, latent1)
    score2 = self.decode_conditional(x_tilde,labels, latent2)
    out = (score1, score2)
    return out


@utils.register_model(name='ddpm_conditional')
class DDPM_conditional(nn.Module):
  def __init__(self, config, latent_cond_dim =16):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    # this is input channel to score model. It has been changed to 32 to incorporate the conditional
    # resnet block's output dim
    channels = 32 #config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    if (config.training.forward_diffusion == 'predict' or config.training.forward_diffusion == 'predict_multi') and config.training.continuous:
      out_channel_num = config.data.num_channels//2
    else:
      out_channel_num = config.data.num_channels
    modules.append(conv3x3(in_ch, out_channel_num, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

    ## for the encoding part####------
    filter_dim = 64
    latent_dim = 64
    encode_vector_length = latent_cond_dim
    latent_dim_expand = 16
    self.embed_conv1 = nn.Conv2d(config.data.num_channels, filter_dim, kernel_size=3, stride=1, padding=1)
    self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_fc1 = nn.Linear(filter_dim, int(filter_dim/2))
    self.embed_fc2 = nn.Linear(int(filter_dim/2), latent_dim_expand)


    self.layer_encode = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer1 = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer2 = CondResBlock(downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.begin_conv = nn.Sequential(nn.Conv2d(config.data.num_channels, filter_dim, kernel_size = 3, stride=1, padding=1),
                                        nn.Conv2d(filter_dim, int(filter_dim/2), 3, stride=1, padding=1))
    self.upconv = nn.Conv2d(in_channels=64, out_channels=filter_dim, kernel_size=3, padding=1)
    # self.encoder = torchvision.models.resnet18(pretrained = False)
    # self.encoder_linear = nn.Linear(1000, encode_vector_length)

    # self.ddpm = DDPM(config=config)

  
  

  
  
  # def forward_generate(self, x_tilde, labels, latent_factor):
  #   # score1 = self.decode(x_tilde,labels)
  #   score = self.decode_conditional(x_tilde,labels, latent_factor)
  #   return score

  def forward(self, x, labels, latent):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    h = self.begin_conv(h)

    h = self.layer_encode(h, latent) # 64, 64, 64

    h = self.layer1(h, latent)  # 64,64,64
    h = self.layer2(h, latent)  # b,32,32,32
    # h = self.upconv(h)  # 128,64,64


    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  


@utils.register_model(name='ddpm_latent')
class DDPM_latent(nn.Module):
  def __init__(self, config, latent_cond_dim =8):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    latent_cond_dim = config.model.latent_dim
    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    self.rotate_basis = config.model.rotate_basis
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    # this is input channel to score model. It has been changed to 32 to incorporate the conditional
    # resnet block's output dim
    channels = 32 #config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, config.data.num_channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

    ## for the encoding part####------
    filter_dim = 64
    latent_dim = 64
    encode_vector_length = latent_cond_dim
    latent_dim_expand = 16
    self.embed_conv1 = nn.Conv2d(config.data.num_channels, filter_dim, kernel_size=3, stride=1, padding=1)
    self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_fc1 = nn.Linear(filter_dim, int(filter_dim/2))
    self.embed_fc2 = nn.Linear(int(filter_dim/2), latent_dim_expand)


    self.layer_encode = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer1 = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer2 = CondResBlock(downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.begin_conv = nn.Sequential(nn.Conv2d(config.data.num_channels, filter_dim, kernel_size = 3, stride=1, padding=1),
                                        nn.Conv2d(filter_dim, int(filter_dim/2), 3, stride=1, padding=1))
    self.upconv = nn.Conv2d(in_channels=64, out_channels=filter_dim, kernel_size=3, padding=1)
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000, encode_vector_length)

    self.ddpm = DDPM(config=config)
    self.W = self.compute_dct(latent_cond_dim)
  

  def compute_dct(self, n):
  #computes a matrix of size n*n such that it can be used to perform discrete cosine transform
  # note that axis =1 means that the columns are the basis, i.e fist col has all the same values

    D = dct(np.eye(n), axis =1)
    W = torch.from_numpy(D).float()
    return W


  def encode(self, x):
    if x.shape[1]==1:
      x = x.repeat(1, 3, 1, 1)
    out = self.encoder(x)
    out = self.encoder_linear(out)
    out = nn.Softmax(dim=-1)(out)
    return out

  def decode(self, x, labels):
    return self.ddpm(x,labels)
  
  def decode_conditional(self, x, labels, latent):
    '''
    A wrapper function to check if we need to rotate basis before applying the conditional latent generation
    '''
    if self.rotate_basis:
      latent = torch.mm(latent, self.W.to(latent.device))
      return self.decode_conditional_ori(x, labels, latent)
    else:
      return self.decode_conditional_ori(x, labels, latent)

  
  def decode_conditional_ori(self, x, labels, latent):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    h = self.begin_conv(h)

    h = self.layer_encode(h, latent) # 64, 64, 64

    h = self.layer1(h, latent)  # 64,64,64
    h = self.layer2(h, latent)  # b,32,32,32
    # h = self.upconv(h)  # 128,64,64


    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  def get_latent(self, prob):
    m = prob.shape[0]
    I = torch.diag_embed(prob)
    # latent = torch.mv(I, prob)

    indx = torch.randint(low=0, high=3, size=(m,2))
    # index = [arange(m), indx, :]
    out = I[torch.arange(m).unsqueeze(1), indx]
    
    return out[:,0], out[:,1]
  
  def forward_generate(self, x_tilde, labels, latent_factor):
    # score1 = self.decode(x_tilde,labels)
    score = self.decode_conditional(x_tilde,labels, latent_factor)
    return score
 

  def forward(self, x_tilde, labels, x):
    latent = self.encode(x)
    if 'detach' in self.config.training.keys() and self.config.training.detach:
      latent = latent.detach()
    score = self.decode_conditional(x_tilde,labels, latent)
    # score2 = self.decode_conditional(x_tilde,labels, latent2)
    # out = score
    return score

@utils.register_model(name='ddpm_latent_variational')
class DDPM_latent_variational(nn.Module):
  def __init__(self, config, latent_cond_dim =8):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
    latent_cond_dim = config.model.latent_dim
    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    # this is input channel to score model. It has been changed to 32 to incorporate the conditional
    # resnet block's output dim
    channels = 32 #config.data.num_channels

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, config.data.num_channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

    ## for the encoding part####------
    filter_dim = 64
    latent_dim = 64
    self.encode_vector_length = encode_vector_length = latent_cond_dim
    latent_dim_expand = 16
    self.embed_conv1 = nn.Conv2d(config.data.num_channels, filter_dim, kernel_size=3, stride=1, padding=1)
    self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
    self.embed_fc1 = nn.Linear(filter_dim, int(filter_dim/2))
    self.embed_fc2 = nn.Linear(int(filter_dim/2), latent_dim_expand)


    self.layer_encode = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer1 = CondResBlock(rescale=False, downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.layer2 = CondResBlock(downsample=False, latent_dim=encode_vector_length, filters=encode_vector_length)
    self.begin_conv = nn.Sequential(nn.Conv2d(config.data.num_channels, filter_dim, kernel_size = 3, stride=1, padding=1),
                                        nn.Conv2d(filter_dim, int(filter_dim/2), 3, stride=1, padding=1))
    self.upconv = nn.Conv2d(in_channels=64, out_channels=filter_dim, kernel_size=3, padding=1)
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000, 2*encode_vector_length)

    self.ddpm = DDPM(config=config)
  
  def compute_kl(self, mu, log_var):
    sum = torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    kl = torch.mean(-0.5 * sum, dim = 0)
    return kl


  def encode(self, x):
    if x.shape[1]==1:
      x = x.repeat(1, 3, 1, 1)
    out = self.encoder(x)
    out = self.encoder_linear(out)
    mu, logvar = out[:, :self.encode_vector_length], out[:, self.encode_vector_length:]
    if self.training:
      std =torch.exp(0.5 * logvar)
      z = mu + std*torch.randn_like(std)
    else:
      z = mu
    kl_loss = self.compute_kl(mu, logvar)
    # out = nn.Softmax(dim=-1)(out)
    return z, kl_loss

  def decode(self, x, labels):
    return self.ddpm(x,labels)

  
  def decode_conditional(self, x, labels, latent):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    h = self.begin_conv(h)

    h = self.layer_encode(h, latent) # 64, 64, 64

    h = self.layer1(h, latent)  # 64,64,64
    h = self.layer2(h, latent)  # b,32,32,32
    # h = self.upconv(h)  # 128,64,64


    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h

  def get_latent(self, prob):
    m = prob.shape[0]
    I = torch.diag_embed(prob)
    # latent = torch.mv(I, prob)

    indx = torch.randint(low=0, high=3, size=(m,2))
    # index = [arange(m), indx, :]
    out = I[torch.arange(m).unsqueeze(1), indx]
    
    return out[:,0], out[:,1]
  
  def forward_generate(self, x_tilde, labels, latent_factor):
    # score1 = self.decode(x_tilde,labels)
    score = self.decode_conditional(x_tilde,labels, latent_factor)
    return score

  # def compute_regularization_loss(self, z):


  # def get_regularization(self):
  #   return self.regularization_loss

  def forward(self, x_tilde, labels, x):
    latent, kl_loss = self.encode(x)
    # self.compute_regularization_loss(latent)
    # latent1, latent2 = self.get_latent(latent_prob)

    # score1 = self.decode(x_tilde,labels)

    score = self.decode_conditional(x_tilde,labels, latent)
    # score2 = self.decode_conditional(x_tilde,labels, latent2)
    # out = score
    return score, kl_loss

@utils.register_model(name='ddpm_latent_factor')
class DDPM_latent_factor(nn.Module):
  def __init__(self, config, latent_cond_dim =16):
    super().__init__()
    self.cond_decoder_1 = DDPM_conditional(config=config, latent_cond_dim=latent_cond_dim)
    self.cond_decoder_2 = DDPM_conditional(config=config, latent_cond_dim=latent_cond_dim)
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000,latent_cond_dim)
    self.latent_cond_dim = latent_cond_dim
    self.ddpm = DDPM(config=config)


  def encode(self, x):
    out = self.encoder(x)
    out = self.encoder_linear(out)
    # out = nn.Softmax(dim=-1)(out)
    return out

  def decode(self, x, labels):
    return self.ddpm(x,labels)

  
  # def decode_conditional(self, x, labels, latent):
    

  # def get_latent(self, prob):
  #   m = prob.shape[0]
  #   I = torch.diag_embed(prob)
  #   # latent = torch.mv(I, prob)

  #   indx = torch.randint(low=0, high=3, size=(m,2))
  #   # index = [arange(m), indx, :]
  #   out = I[torch.arange(m).unsqueeze(1), indx]
    
  #   return out[:,0], out[:,1]
  
  def forward_generate(self, x_tilde, labels, latent_factor):
    # score1 = self.decode(x_tilde,labels)
    score1 = self.cond_decoder_1(x_tilde,labels, latent_factor)
    score2 = self.cond_decoder_2(x_tilde,labels, latent_factor)
    return score1, score2


  def forward(self, x_tilde, labels, x):
    if x.shape[1]==1:
      x = x.repeat(1, 3, 1, 1)
    latent = self.encode(x)

    # score1 = self.cond_decoder_1(x_tilde,labels, latent[:, :self.latent_cond_dim])
    score1 = self.cond_decoder_1(x_tilde,labels, latent)
    score2 = self.cond_decoder_2(x_tilde,labels, latent)
    # score2 = self.decode_conditional(x_tilde,labels, latent2)
    # out = score
    return score1, score2

@utils.register_model(name='ddpm_basis')
class DDPM_basis(nn.Module):
  def __init__(self, config, latent_cond_dim =3):
    super().__init__()
    self.cond_decoder_1 = DDPM_conditional(config=config, latent_cond_dim=latent_cond_dim)
    # self.cond_decoder_2 = DDPM_conditional(config=config, latent_cond_dim=latent_cond_dim)
    # self.cond_decoder_3 = DDPM_conditional(config=config, latent_cond_dim=latent_cond_dim)
    
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000,latent_cond_dim)
    self.latent_cond_dim = latent_cond_dim
    self.ddpm = DDPM(config=config)
    self.W = self.compute_dct(latent_cond_dim)

  def compute_dct(self, n):
  #computes a matrix of size n*n such that it can be used to perform discrete cosine transform
  # note that axis =1 means that the columns are the basis, i.e fist col has all the same values

    D = dct(np.eye(n), axis =1)
    W = torch.from_numpy(D).float()
    return W

  def encode(self, x):
    out = self.encoder(x)
    out = self.encoder_linear(out)
    out = nn.Softmax(dim=-1)(out)
    return out

  def decode(self, x, labels):
    return self.ddpm(x,labels)

  
  # def decode_conditional(self, x, labels, latent):
    

  def get_latent(self, batch_size):
    m = self.latent_cond_dim
    diagonal = torch.ones(batch_size, m)
    I = torch.diag_embed(diagonal)
    
    return I
  
  def forward_generate(self, x_tilde, labels, latent_factor):
    # score1 = self.decode(x_tilde,labels)
    score1 = self.cond_decoder_1(x_tilde,labels, torch.mm(latent_factor,self.W.to(latent_factor.device)))
    # score2 = self.cond_decoder_2(x_tilde,labels, latent_factor)
    # score3 = self.cond_decoder_3(x_tilde,labels, latent_factor)
    # scores = torch.cat((score1.unsqueeze(-1), score2.unsqueeze(-1), score3.unsqueeze(-1)), dim=-1)
    return score1


  def forward(self, x_tilde, labels, x):
    wt = self.encode(x)
    latent = self.get_latent(x.shape[0]).to(x.device)
    # score1 = self.cond_decoder_1(x_tilde,labels, latent[:, :self.latent_cond_dim])
    scores_stack = []
    for i in range(self.latent_cond_dim):
      score = self.cond_decoder_1(x_tilde,labels, torch.mm(latent[:,:,0], self.W.to(latent.device))).unsqueeze(-1)
      scores_stack.append(score)
      # score2 = self.cond_decoder_1(x_tilde,labels, latent[:,:,1])
      # score3 = self.cond_decoder_1(x_tilde,labels, latent[:,:,2])

    scores = torch.cat(scores_stack, dim=-1)
    return scores, wt


@utils.register_model(name='ddpm_label')
class DDPM_label(nn.Module):
  def __init__(self, config, latent_cond_dim =10):
    super().__init__()
    self.cond_decoder = DDPM_conditional(config=config, latent_cond_dim=latent_cond_dim)
    
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000,latent_cond_dim)
    self.latent_cond_dim = latent_cond_dim
    self.ddpm = DDPM(config=config)

  
  def forward_generate(self, x_tilde, labels, class_label):
    score = self.cond_decoder(x_tilde,labels, class_label)
    return score


  def forward(self, x_tilde, labels, class_label):
    
    score = self.cond_decoder(x_tilde, labels, class_label)
   
    return score


@utils.register_model(name='ddpm_latent_contrastive')
class DDPM_latent_contrastive(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.latent_dim = config.model.latent_dim
    self.cond_decoder = DDPM_conditional(config=config, latent_cond_dim=self.latent_dim)
    
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000,self.latent_dim)
    # self.ddpm = DDPM(config=config)
    self.rotate_basis = config.model.rotate_basis
    self.W = self.compute_dct(self.latent_dim)

  def compute_dct(self, n):
  #computes a matrix of size n*n such that it can be used to perform discrete cosine transform
  # note that axis =1 means that the columns are the basis, i.e fist col has all the same values

    D = dct(np.eye(n), axis =1)
    W = torch.from_numpy(D).float()
    return W

  def encode(self, x):
    if x.shape[1]==1:
      x = x.repeat(1, 3, 1, 1)
    out = self.encoder(x)
    out = self.encoder_linear(out)
    out = nn.Softmax(dim=-1)(out)
    return out
  
  def decode(self, x, labels, latent):
    '''
    A wrapper function to check if we need to rotate basis before applying the conditional latent generation
    '''
    if self.rotate_basis:
      latent = torch.mm(latent, self.W.to(latent.device))
      return self.cond_decoder(x, labels, latent)
    else:
      return self.cond_decoder(x, labels, latent)

  
  def forward_generate(self, x_tilde, labels, latent_factor):   
    score = self.decode(x_tilde,labels, latent_factor)
    return score


  def forward(self, x_tilde, labels, x):
    latent = self.encode(x)
    m = latent.shape[0]
    # assert m%2==0
    m_2 = int(math.ceil(m/2))
    n = m-m_2
    # latent[m_2:] = latent[:n]
    latent = torch.cat((latent[:m_2], latent[:n]), dim=0)
    score = self.decode(x_tilde,labels, latent)

    return score

@utils.register_model(name='ddpm_latent_aniso')
class DDPM_latent_aniso(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.latent_dim = config.model.latent_dim
    self.cond_decoder = DDPM_conditional(config=config, latent_cond_dim=self.latent_dim)
    self.rotate_basis = config.model.rotate_basis
    self.W = self.compute_dct(self.latent_dim)

  def compute_dct(self, n):
  #computes a matrix of size n*n such that it can be used to perform discrete cosine transform
  # note that axis =1 means that the columns are the basis, i.e fist col has all the same values
    D = dct(np.eye(n), axis =1)
    W = torch.from_numpy(D).float()
    return W
  
  def decode(self, x, labels, latent):
    '''
    A wrapper function to check if we need to rotate basis before applying the conditional latent generation
    '''
    if self.rotate_basis:
      latent = torch.mm(latent, self.W.to(latent.device))
    
    return self.cond_decoder(x, labels, latent)
 
  def forward_generate(self, x_tilde, labels, latent_factor):   
    score = self.decode(x_tilde,labels, latent_factor)
    return score


  def forward(self, x_tilde, labels, latent):
    score = self.decode(x_tilde,labels, latent)
    return score

@utils.register_model(name='ddpm_latent_aniso_multi')
class DDPM_latent_aniso_multi(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.latent_dim = config.model.latent_dim
    self.cond_decoder_0 = DDPM_conditional(config=config, latent_cond_dim=self.latent_dim)
    self.cond_decoder_1 = DDPM_conditional(config=config, latent_cond_dim=self.latent_dim)
    self.rotate_basis = config.model.rotate_basis
    self.W = self.compute_dct(self.latent_dim)

  def compute_dct(self, n):
  #computes a matrix of size n*n such that it can be used to perform discrete cosine transform
  # note that axis =1 means that the columns are the basis, i.e fist col has all the same values
    D = dct(np.eye(n), axis =1)
    W = torch.from_numpy(D).float()
    return W
  
  def decode(self, x, labels, latent, i):
    '''
    A wrapper function to check if we need to rotate basis before applying the conditional latent generation
    '''
    x = x[:,:,:,:,i]
    latent = latent[:,:,i]
    if self.rotate_basis:
      latent = torch.mm(latent, self.W.to(latent.device))
    func = 'self.cond_decoder_'+str(i)+'(x, labels, latent)'
    out = eval(func)
    return out
 
  def forward_generate(self, x_tilde, labels, latent_factor): 
    score_list = []  
    for i in range(latent_factor.shape[-1]):
      score = self.decode(x_tilde,labels, latent_factor, i)
      score_list.append(score.unsqueeze(-1))
    out = torch.cat((score_list), dim = -1)
    return out


  def forward(self, x_tilde, labels, latents):
    score_list = []  
    for i in range(latents.shape[-1]):
      score = self.decode(x_tilde,labels, latents, i)
      score_list.append(score.unsqueeze(-1))
    out = torch.cat((score_list), dim=-1)
    return out
    # score = self.decode(x_tilde,labels, latent, 0) + self.decode(x_tilde,labels, latent, 1)
    # return score


    

@utils.register_model(name='encode_resnet')
class EncodeResnet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.latent_dim = config.model.latent_dim   
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear = nn.Linear(1000,self.latent_dim)
    self.decoder_x = LinearConv(self.latent_dim, config.data.image_size)

  def encode(self, x):
    if x.shape[1]==1:
      x = x.repeat(1, 3, 1, 1)
    out = self.encoder(x)
    out = self.encoder_linear(out)
    out = nn.Softmax(dim=-1)(out)
    return out
  
  def latent_to_x(self, latent):
    return self.decoder_x(latent)

  def forward(self, x):
    latent = self.encode(x)
    latent_x = self.latent_to_x(latent)
   
    return latent, latent_x


@utils.register_model(name='encode_resnet_multi')
class EncodeResnet_multi(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.latent_dim = config.model.latent_dim   
    self.encoder = torchvision.models.resnet18(pretrained = False)
    self.encoder_linear_1 = nn.Linear(1000,self.latent_dim)
    self.encoder_linear_2 = nn.Linear(1000,self.latent_dim)
    self.decoder_x_1 = LinearConv(self.latent_dim, config.data.image_size)
    self.decoder_x_2 = LinearConv(self.latent_dim, config.data.image_size)

  def encode(self, x):
    if x.shape[1]==1:
      x = x.repeat(1, 3, 1, 1)
    out = self.encoder(x)
    out_1 = self.encoder_linear_1(out)
    out_1 = nn.Softmax(dim=-1)(out_1)

    out_2 = self.encoder_linear_2(out)
    out_2 = nn.Softmax(dim=-1)(out_2)

    out_ = torch.cat([out_1.unsqueeze(-1), out_2.unsqueeze(-1)], dim=-1)

    return out_
  
  def latent_to_x(self, latent):
    latent_0 , latent_1 = latent[:,:,0], latent[:,:,1]
    out_1 = self.decoder_x_1(latent_0)
    out_2 = self.decoder_x_2(latent_1)
    out = [out_1.unsqueeze(-1), out_2.unsqueeze(-1)]
    out = torch.cat(out, dim =-1)
    return out

  def forward(self, x):
    latent = self.encode(x)
    latent_x = self.latent_to_x(latent)
   
    return latent, latent_x