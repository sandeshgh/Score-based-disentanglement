"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import torchsde
import torch.nn as nn
import copy


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent=None):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        if latent is not None:
          score = score_fn(x, t, latent)
        else:
          score = score_fn(x, t)
          
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

  def reverse_multilatent(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent=None):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        b,c,w,h,i_n = x.shape
        if c ==1:
          drift, diffusion = sde_fn(x.squeeze(), t)
          drift = drift.unsqueeze(1)
        else:
          drift, diffusion = sde_fn(x.permute(0,1,4,2,3).contiguous().view(b,c*i_n, w, h), t)
          drift = drift.contiguous().view(b, c,i_n, w, h).permute(0,1,3,4,2)
        score_list = []
        score_ind_list = []
        N = latent.shape[-1]
        for i in range(N):
          score_ind_list.append(score_fn(x[:,:,:,:,i], t, latent[:,:,i]).unsqueeze(-1))
          score_list.append(score_fn(x[:,:,:,:,N], t, latent[:,:,i]).unsqueeze(-1))

        
        
        score_mean = torch.cat(score_list, dim =-1)

        score_mean = score_mean.mean(dim = -1, keepdim=True)

        score_ind_list.append(score_mean)
        score_stack = torch.cat(score_ind_list, dim=-1)

        # score_stack = torch.cat((score*3, score_mean), dim =-1)
       
          
        drift = drift - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()
  
  def reverse_multilatent_uncond(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent, alpha):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        b,c,w,h,i_n = x.shape
        if c ==1:
          drift, diffusion = sde_fn(x.squeeze(), t)
          drift = drift.unsqueeze(1)
        else:
          drift, diffusion = sde_fn(x.permute(0,1,4,2,3).contiguous().view(b,c*i_n, w, h), t)
          drift = drift.contiguous().view(b, c,i_n, w, h).permute(0,1,3,4,2)
        score_list = []
        score_ind_list = []
        N = latent.shape[-1]
        for i in range(N):
          latent_i = latent[:,:,i]
          latent_u = torch.ones_like(latent_i)
          score_i = score_fn(x[:,:,:,:,i], t, latent_i).unsqueeze(-1)
          score_u = score_fn(x[:,:,:,:,i], t, latent_u).unsqueeze(-1)
          
          score_ind_list.append(alpha*score_i + (1-alpha)*score_u)
          
          if i == 2:
            score_list.append(score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1))
            
          else:
            score_list.append(score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1))

        
        
        score_mean = torch.cat(score_list, dim =-1)

        score_mean = score_mean.mean(dim = -1, keepdim=True)
        score_u_mean = score_fn(x[:,:,:,:,N], t, latent_u).unsqueeze(-1)

        # score_ind_list.append(score_mean + (1-alpha)*score_u_mean*2/3)
        score_ind_list.append(alpha*score_mean + (1-alpha)*score_u_mean)
        score_stack = torch.cat(score_ind_list, dim=-1)

        # score_stack = torch.cat((score*3, score_mean), dim =-1)
       
          
        drift = drift - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()
  
  def reverse_multilatent_tune(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent, alpha):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        b,c,w,h,i_n = x.shape
        if c ==1:
          drift, diffusion = sde_fn(x.squeeze(), t)
          drift = drift.unsqueeze(1)
        else:
          drift, diffusion = sde_fn(x.permute(0,1,4,2,3).contiguous().view(b,c*i_n, w, h), t)
          drift = drift.contiguous().view(b, c,i_n, w, h).permute(0,1,3,4,2)
        score_list = []
        score_ind_list = []
        N = latent.shape[-1]
        for i in range(N):
          latent_i = latent[:,:,i]
          latent_u = torch.ones_like(latent_i)
          score_i = score_fn(x[:,:,:,:,i], t, latent_i).unsqueeze(-1)
          score_u = score_fn(x[:,:,:,:,i], t, latent_u).unsqueeze(-1)
          
          score_ind_list.append(alpha*score_i + (1-alpha)*score_u)
          
          if i == 2:
            score_list.append(score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1))
          
          elif i==0:
            score_list.append((2*alpha)*score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1))
          else:
            score_list.append((2-2*alpha)*score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1))

        
        
        score_mean = torch.cat(score_list, dim =-1)

        score_mean = score_mean.mean(dim = -1, keepdim=True)
        # score_u_mean = score_fn(x[:,:,:,:,N], t, latent_u).unsqueeze(-1)

        score_ind_list.append(score_mean)
        # score_ind_list.append(alpha*score_mean + (1-alpha)*score_u_mean/3)
        score_stack = torch.cat(score_ind_list, dim=-1)

        # score_stack = torch.cat((score*3, score_mean), dim =-1)
       
          
        drift = drift - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


  def reverse_multilatent_mix(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent, alpha):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        b,c,w,h,i_n = x.shape
        if c ==1:
          drift, diffusion = sde_fn(x.squeeze(), t)
          drift = drift.unsqueeze(1)
        else:
          drift, diffusion = sde_fn(x.permute(0,1,4,2,3).contiguous().view(b,c*i_n, w, h), t)
          drift = drift.contiguous().view(b, c,i_n, w, h).permute(0,1,3,4,2)
        score_list = []
        score_ind_list = []
        N = latent.shape[-1]
        for i in range(N):
          latent_i = copy.deepcopy(latent)[:,:,i]
          if i == alpha:
            half_b = b//2
            lat_i_1 = latent_i[:half_b]
            lat_i_2 = latent_i[half_b:]

            lat_i = torch.cat((lat_i_2, lat_i_1), dim = 0)
            latent_i = lat_i
          
          # latent_u = torch.ones_like(latent_i)
          score_i = score_fn(x[:,:,:,:,i], t, latent_i).unsqueeze(-1)
          # score_u = score_fn(x[:,:,:,:,i], t, latent_u).unsqueeze(-1)
          
          score_ind_list.append(score_i )
          score_list_i = score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1)
          
          # if i == alpha:
          #   half_b = b//2
          #   score_list_i_1 = score_list_i[:half_b]
          #   score_list_i_2 = score_list_i[half_b:]

          #   score_list_i = torch.cat((score_list_i_2, score_list_i_1), dim = 0)
            # swap here
          
          score_list.append(score_list_i)
            
          # else:
          #   score_list.append(alpha*score_fn(x[:,:,:,:,N], t, latent_i).unsqueeze(-1))

        
        
        score_mean = torch.cat(score_list, dim =-1)

        score_mean = score_mean.mean(dim = -1, keepdim=True)
        # score_u_mean = score_fn(x[:,:,:,:,N], t, latent_u).unsqueeze(-1)

        score_ind_list.append(score_mean )
        # score_ind_list.append(alpha*score_mean + (1-alpha)*score_u_mean/3)
        score_stack = torch.cat(score_ind_list, dim=-1)

        # score_stack = torch.cat((score*3, score_mean), dim =-1)
       
          
        drift = drift - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

  def reverse_spectral(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent, latent_x):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t, latent_x)
        
        score = score_fn(x, t, latent, latent_x)
          
        # drift = drift - score * (0.5 if self.probability_flow else 1.)
        drift = drift - diffusion ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()
  
  

  def reverse_predict(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent, latent_x):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)

        x = torch.cat((x,latent_x), dim =1)
        
        score = score_fn(x, t, latent)
          
        # drift = drift - score * (0.5 if self.probability_flow else 1.)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

  def reverse_predict_multi(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent, latent_x):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x.squeeze(), t)
        drift = drift.view(x.shape)

        x_score = torch.cat((x[:,:,:,:,:2], latent_x), dim =1)

        # x = torch.cat((x,latent_x), dim =1)
        score = score_fn(x_score, t, latent)
        x_mean = x[:,:,:,:,2]
        x_mean_cat = torch.cat((x_mean[:,:,:,:,None].repeat(1,1,1,1,2), latent_x), dim =1)
        score_x = score_fn(x_mean_cat, t, latent)

        score_sum = score_x.mean(-1).unsqueeze(-1)
        scores = torch.cat((score, score_sum), dim = -1)
          
        # drift = drift - score * (0.5 if self.probability_flow else 1.)
        drift = drift - diffusion[:, None, None, None, None] ** 2 * scores * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

  def reverse_mix(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent=None):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        # if latent is not None:
        score0 = score_fn(x, t, latent[:,:,0])
        score1 = score_fn(x, t, latent[:,:,1])
        score = 0.5*score0+0.5*score1
        # else:
        #   score = score_fn(x, t)
          
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, latent=None):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        if latent is not None:
          f, G = discretize_fn(x, t, latent)
        else:
          f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

  def reverse_tensor(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score= score_fn(x, t, latent)
        # score_stack = torch.cat((score1.unsqueeze(-1), score2.unsqueeze(-1)), dim =-1)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

  def reverse_multivariate(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE_multivariate(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, return_score=None):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score, score1, score2 = score_fn(x, t)
        score_stack = torch.cat((score1.unsqueeze(-1), score2.unsqueeze(-1)), dim =-1)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # multiply by a factor of 2 for each score to account for the fact that they are probably in lower range since score = score1+score2
        drift_stack = drift.unsqueeze(-1) - diffusion[:, None, None, None, None] ** 2 * score_stack * (1 if self.probability_flow else 2.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        if return_score == 'score':
          return drift, diffusion, drift_stack, score_stack
        else:
          return drift, diffusion, drift_stack

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        score, score_stack = score_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        rev_f_stack = f - G[:, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        return rev_f, rev_G, rev_f_stack

    return RSDE_multivariate()

  def reverse_conditional(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
        The reverse conditional can be used for the generation with conditional latent variables.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE_conditional(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score1, score2 = score_fn(x, t, latent)
        score = score1+score2
        # score1 is the unconditional score, score2 is the conditional score with the condition given by the latent vector
        # these two have been stacked together and drift is computed for the stacked tensor
        score_stack = torch.cat((score.unsqueeze(-1), 2*score1.unsqueeze(-1), 2*score2.unsqueeze(-1)), dim =-1)

        # drift = drift - diffusion[:, None, None, None] ** 2 * score1 * (0.5 if self.probability_flow else 1.)
        drift_stack = drift.unsqueeze(-1) - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift_stack, diffusion, score_stack

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        score, score_stack = score_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        rev_f_stack = f - G[:, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        return rev_f, rev_G, rev_f_stack

    return RSDE_conditional()
  
  def reverse_weighted(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
        The reverse conditional can be used for the generation with conditional latent variables.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE_conditional(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, batch, idx):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""

        drift, diffusion = sde_fn(x, t)
        scores_list, alpha = score_fn(x, t, batch)
        if idx == 'sum':
          
          score = (scores_list*alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)).mean(-1)
        else:
          score = 0.1*scores_list[:,:,:,:,idx]

        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # drift_stack = drift.unsqueeze(-1) - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        score, score_stack = score_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        rev_f_stack = f - G[:, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        return rev_f, rev_G, rev_f_stack

    return RSDE_conditional()

  def reverse_conditional_swap(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.
        The reverse conditional can be used for the generation with conditional latent variables.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE_conditional(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, latent):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score01, score02 = score_fn(x, t, latent[:,:,0])
        score11, score12 = score_fn(x, t, latent[:,:,1])
        score0 = score01+score02
        score1 = score11+score12
        score_diag = score01+score12
        score_off_diag = score02+score11

        # score1 is the unconditional score, score2 is the conditional score with the condition given by the latent vector
        # these two have been stacked together and drift is computed for the stacked tensor
        score_stack = torch.cat((score0.unsqueeze(-1), score1.unsqueeze(-1), score_diag.unsqueeze(-1), score_off_diag.unsqueeze(-1)), dim =-1)

        # drift = drift - diffusion[:, None, None, None] ** 2 * score1 * (0.5 if self.probability_flow else 1.)
        drift_stack = drift.unsqueeze(-1) - diffusion[:, None, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift_stack, diffusion, score_stack

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        score, score_stack = score_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        rev_f_stack = f - G[:, None, None, None] ** 2 * score_stack * (0.5 if self.probability_flow else 1.)
        return rev_f, rev_G, rev_f_stack

    return RSDE_conditional()

class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G

class Neural_SDE(nn.Module):
  sde_type = 'stratonovich'
  noise_type = 'scalar'
  def __init__(self, model, beta_min, beta_max, shape):
    super().__init__()
    self.model = model
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.shape = shape
  
  # def get_drift_diffusion(self, t, x):
  #   beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
  #   drift = -0.5 * beta_t[None, None, None, None] * self.model(x)
  #   diffusion = torch.sqrt(beta_t)
  #   return drift, diffusion
      
  def f(self, t, x):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[None, None, None, None] * self.model(x)
    return drift
  
  def g(self, t, x):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    diffusion = torch.sqrt(beta_t)
    return diffusion


class LearnedDiffusionVPSDE(SDE):
  def __init__(self, f_model, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.f = f_model
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * self.f(x)
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def update_model(self, model):
    self.f = model
    return 0
  

  # def marginal_prob(self, x, t):
  #   log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
  #   mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
  #   std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
  #   return mean, std

  def sample_trajectory(self, x_0, t):
    sde_func = Neural_SDE(self.f, beta_min=self.beta_0, beta_max=self.beta_1, shape = x_0.shape )

    pred = torchsde.sdeint_adjoint(sde_func, x_0.view(x_0.shape[0],-1), ts=t.to(x_0.device), method = 'reversible_heun', adjoint_method='adjoint_reversible_heun', dt=0.1)
    return pred


  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class Spectral_VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t, z_x):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None]* z_x * x
    diffusion = torch.sqrt(beta_t[:, None, None, None] * z_x)
    return drift, diffusion

  def marginal_prob(self, u, t, zeta):
    beta_t = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    # u = torch.mm(U, x)
    log_mean_coeff = beta_t[:, None, None, None]*zeta
    mean = torch.exp(log_mean_coeff)*u
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    raise NotImplementedError("Prior logp not implemented.")
    # shape = z.shape
    # N = np.prod(shape[1:])
    # logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    # return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    raise NotImplementedError("Discretization not implemented for Spectral vpsde.")


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std
  
  def marginal_std(self, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    # mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G