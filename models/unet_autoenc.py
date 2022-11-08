from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu
from . import utils

from .latentnet import *
from .unet import *
from .choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # def __init__(self, conf):
        
    #     super(conf).__init__()
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)

# @utils.register_model(name='ddpm')
@utils.register_model(name = 'ddpm_latent_Adain')
class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        
        super().__init__(conf)
        self.conf = conf 

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            # pool=conf.enc_pool,
        ).make_model()

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        cond = self.encoder.forward(x)
        return cond

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S
    
    def forward_generate(self, x, t, latent):
        t_cond = t
        
        cond = latent

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        # style = style or res.style

        # assert (y is not None) == (
        #     self.conf.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    # print(i, j, h.shape)
                    # print('emb', enc_time_emb.shape)
                    # print('cond', enc_cond_emb.shape)
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        # return AutoencReturn(pred=pred, cond=cond)
        return pred


    def forward(self,
                x,
                t,
                x_start=None,
                y=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        # if t_cond is None:
        t_cond = t

        # if noise is not None:
        #     # if the noise is given, we predict the cond from noise
        #     cond = self.noise_to_cond(noise)

        # if cond is None:
        if x is not None:
            assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

        tmp = self.encode(x_start)
        cond = tmp #tmp['cond']

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    # print(i, j, h.shape)
                    # print('emb', enc_time_emb.shape)
                    # print('cond', enc_cond_emb.shape)
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        # return AutoencReturn(pred=pred, cond=cond)
        return pred

@utils.register_model(name = 'ddpm_latent_Adain_multiscore')
class BeatGANsAutoencModel_Multi(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        
        super().__init__(conf)
        self.conf = conf 

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            # pool=conf.enc_pool,
        ).make_model()

        self.decoder_1 = BeatGANsAutoencDecoder(conf)
        self.decoder_2 = BeatGANsAutoencDecoder(conf)

        # if conf.latent_net_conf is not None:
        #     self.latent_net = conf.latent_net_conf.make_model()

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     assert self.conf.is_stochastic
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    # def sample_z(self, n: int, device):
    #     assert self.conf.is_stochastic
    #     return torch.randn(n, self.conf.enc_out_channels, device=device)

    # def noise_to_cond(self, noise: Tensor):
    #     raise NotImplementedError()
    #     assert self.conf.noise_net_conf is not None
    #     return self.noise_net.forward(noise)

    # def encode(self, x):
    #     cond = self.encoder.forward(x)
    #     return cond

    # @property
    # def stylespace_sizes(self):
    #     modules = list(self.input_blocks.modules()) + list(
    #         self.middle_block.modules()) + list(self.output_blocks.modules())
    #     sizes = []
    #     for module in modules:
    #         if isinstance(module, ResBlock):
    #             linear = module.cond_emb_layers[-1]
    #             sizes.append(linear.weight.shape[0])
    #     return sizes

    # def encode_stylespace(self, x, return_vector: bool = True):
    #     """
    #     encode to style space
    #     """
    #     modules = list(self.input_blocks.modules()) + list(
    #         self.middle_block.modules()) + list(self.output_blocks.modules())
    #     # (n, c)
    #     cond = self.encoder.forward(x)
    #     S = []
    #     for module in modules:
    #         if isinstance(module, ResBlock):
    #             # (n, c')
    #             s = module.cond_emb_layers.forward(cond)
    #             S.append(s)

    #     if return_vector:
    #         # (n, sum_c)
    #         return torch.cat(S, dim=1)
    #     else:
    #         return S
    
    def forward_generate(self, x, t, latent):
        out = self.decoder(x, t, latent)
        return out


    def forward(self,
                x,
                t,
                x_start=None,
                ):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        cond = self.encode(x_start)

        pred_1 = self.decoder_1(x, t, cond)
        pred_2 = self.decoder_2(x, t, cond)

        pred = torch.cat((pred_1.unsqueeze(-1), pred_2.unsqueeze(-1)), dim =-1)

        return pred

@utils.register_model(name = 'ddpm_latent_Adain_multilatent')
class BeatGANsAutoencModel_Multi(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        
        super().__init__(conf)
        self.conf = conf 
        # self.ortho_loss = torch.FloatTensor([1])
        # self.K = 16
        # self.latent_vec = nn.Parameter(torch.zeros(self.K,conf.enc_out_channels))

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder_1 = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            # pool=conf.enc_pool,
        ).make_model()

        self.encoder_2 = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            # pool=conf.enc_pool,
        ).make_model()

        self.encoder_3 = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            # pool=conf.enc_pool,
        ).make_model()

        self.decoder = BeatGANsAutoencDecoder(conf)
        self.orthogonal_loss_fn = self.get_orthogonal_loss
        

       
    def forward_generate(self, x, t, latent):
        out = self.decoder(x, t, latent)
        return out

    def encode(self, x):
        latent_1 = self.encoder_1(x)
        latent_2 = self.encoder_2(x)
        latent_3 = self.encoder_3(x)
        out = torch.cat((latent_1, latent_2, latent_3), dim = 0)
        return out
    
    def compute_ortho_loss(self, a, b, c):
        cos = torch.nn.CosineSimilarity(dim =1)
        a = a.contiguous().view(a.shape[0], -1)
        b = b.contiguous().view(b.shape[0], -1)
        c = c.contiguous().view(c.shape[0], -1)
        loss = (cos(a,b)**2+cos(b,c)**2+cos(c,a)**2)/3
        return loss
    
    def get_orthogonal_loss(self):
        return self.ortho_loss



    
    # def compute_loss(self, beta):
    #     # find the closest out of K vectors
    #     ind_1 = find_nearest_index(self.latent_vec, self.latent_1)
    #     ind_2 = find_nearest_index(self.latent_vec, self.latent_2)
    #     ind_3 = find_nearest_index(self.latent_vec, self.latent_3)

    #     loss_quantization = (self.latent_vec[ind_1]-self.latent_1.detach())**2 + (self.latent_vec[ind_2]-self.latent_2.detach())**2 + (self.latent_vec[ind_3]-self.latent_3.detach())**2 

    
    def forward_uncond(self,
                x,
                t):
        
        latent_1 = torch.ones(size = (x.shape[0],self.conf.embed_channels), device = x.device, dtype=x.dtype)

        s = self.decoder(x, t, latent_1)

        return s


    def forward(self,
                x,
                t,
                x_start=None,
                return_loss = False
                ):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

        latent_1 = self.encoder_1(x_start)
        latent_2 = self.encoder_2(x_start)
        latent_3 = self.encoder_3(x_start)
        # self.latent_1 = latent_1
        # self.latent_2 = latent_2
        # self.latent_3 = latent_3
        

        s1 = self.decoder(x, t, latent_1)
        s2 = self.decoder(x, t, latent_2)
        s3 = self.decoder(x, t, latent_3)

        ortho_loss = self.compute_ortho_loss(s1, s2, s3)

        pred = (s1+s2+s3)/3
        if return_loss:
            return pred, ortho_loss
        else:
            return pred


@utils.register_model(name = 'ddpm_latent_Adain_multilatent_new')
class BeatGANsAutoencModel_Multi_new(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        
        super().__init__(conf)
        self.conf = conf 
        self.K = 16
        self.n_encoders = 5
        # self.latent_vec = nn.Parameter(torch.zeros(self.K,conf.enc_out_channels))

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder_list = nn.ModuleList(BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            # pool=conf.enc_pool,
        ).make_model() for i in range(self.n_encoders))

        # self.encoder_2 = BeatGANsEncoderConfig(
        #     image_size=conf.image_size,
        #     in_channels=conf.in_channels,
        #     model_channels=conf.model_channels,
        #     out_hid_channels=conf.enc_out_channels,
        #     out_channels=conf.enc_out_channels,
        #     num_res_blocks=conf.enc_num_res_block,
        #     attention_resolutions=(conf.enc_attn_resolutions
        #                            or conf.attention_resolutions),
        #     dropout=conf.dropout,
        #     channel_mult=conf.enc_channel_mult or conf.channel_mult,
        #     use_time_condition=False,
        #     conv_resample=conf.conv_resample,
        #     dims=conf.dims,
        #     use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
        #     num_heads=conf.num_heads,
        #     num_head_channels=conf.num_head_channels,
        #     resblock_updown=conf.resblock_updown,
        #     use_new_attention_order=conf.use_new_attention_order,
        #     # pool=conf.enc_pool,
        # ).make_model()

        # self.encoder_3 = BeatGANsEncoderConfig(
        #     image_size=conf.image_size,
        #     in_channels=conf.in_channels,
        #     model_channels=conf.model_channels,
        #     out_hid_channels=conf.enc_out_channels,
        #     out_channels=conf.enc_out_channels,
        #     num_res_blocks=conf.enc_num_res_block,
        #     attention_resolutions=(conf.enc_attn_resolutions
        #                            or conf.attention_resolutions),
        #     dropout=conf.dropout,
        #     channel_mult=conf.enc_channel_mult or conf.channel_mult,
        #     use_time_condition=False,
        #     conv_resample=conf.conv_resample,
        #     dims=conf.dims,
        #     use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
        #     num_heads=conf.num_heads,
        #     num_head_channels=conf.num_head_channels,
        #     resblock_updown=conf.resblock_updown,
        #     use_new_attention_order=conf.use_new_attention_order,
        #     # pool=conf.enc_pool,
        # ).make_model()

        self.decoder = BeatGANsAutoencDecoder(conf)
        

       
    def forward_generate(self, x, t, latent):
        out = self.decoder(x, t, latent)
        return out

    def encode(self, x):
        latents = []
        # scores = []
        for encoder in self.encoder_list:
            lat_i = encoder(x)
            latents.append(lat_i)
        return latents
    
    # def compute_loss(self, beta):
    #     # find the closest out of K vectors
    #     ind_1 = find_nearest_index(self.latent_vec, self.latent_1)
    #     ind_2 = find_nearest_index(self.latent_vec, self.latent_2)
    #     ind_3 = find_nearest_index(self.latent_vec, self.latent_3)

    #     loss_quantization = (self.latent_vec[ind_1]-self.latent_1.detach())**2 + (self.latent_vec[ind_2]-self.latent_2.detach())**2 + (self.latent_vec[ind_3]-self.latent_3.detach())**2 

    
    def forward_uncond(self,
                x,
                t):
        
        latent_1 = torch.ones(size = (x.shape[0],self.conf.embed_channels), device = x.device, dtype=x.dtype)

        s = self.decoder(x, t, latent_1)

        return s


    def forward(self,
                x,
                t,
                x_start=None,
                ):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        latents = []
        scores = []
        for encoder in self.encoder_list:
            lat_i = encoder(x_start)
            latents.append(lat_i)
            s_i = self.decoder(x,t,lat_i)
            scores.append(s_i.unsqueeze(-1))

        score = torch.cat(scores, dim = -1)
        pred = score.mean(-1)

        return pred



# @utils.register_model(name = 'ddpm_latent_Adain')
class BeatGANsAutoencDecoder(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        
        super().__init__(conf)
        self.conf = conf 

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        # self.encoder = BeatGANsEncoderConfig(
        #     image_size=conf.image_size,
        #     in_channels=conf.in_channels,
        #     model_channels=conf.model_channels,
        #     out_hid_channels=conf.enc_out_channels,
        #     out_channels=conf.enc_out_channels,
        #     num_res_blocks=conf.enc_num_res_block,
        #     attention_resolutions=(conf.enc_attn_resolutions
        #                            or conf.attention_resolutions),
        #     dropout=conf.dropout,
        #     channel_mult=conf.enc_channel_mult or conf.channel_mult,
        #     use_time_condition=False,
        #     conv_resample=conf.conv_resample,
        #     dims=conf.dims,
        #     use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
        #     num_heads=conf.num_heads,
        #     num_head_channels=conf.num_head_channels,
        #     resblock_updown=conf.resblock_updown,
        #     use_new_attention_order=conf.use_new_attention_order,
        #     # pool=conf.enc_pool,
        # ).make_model()

        # if conf.latent_net_conf is not None:
        #     self.latent_net = conf.latent_net_conf.make_model()

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     assert self.conf.is_stochastic
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    # def sample_z(self, n: int, device):
    #     assert self.conf.is_stochastic
    #     return torch.randn(n, self.conf.enc_out_channels, device=device)

    # def noise_to_cond(self, noise: Tensor):
    #     raise NotImplementedError()
    #     assert self.conf.noise_net_conf is not None
    #     return self.noise_net.forward(noise)

    # def encode(self, x):
    #     cond = self.encoder.forward(x)
    #     return cond

    # @property
    # def stylespace_sizes(self):
    #     modules = list(self.input_blocks.modules()) + list(
    #         self.middle_block.modules()) + list(self.output_blocks.modules())
    #     sizes = []
    #     for module in modules:
    #         if isinstance(module, ResBlock):
    #             linear = module.cond_emb_layers[-1]
    #             sizes.append(linear.weight.shape[0])
    #     return sizes

    # def encode_stylespace(self, x, return_vector: bool = True):
    #     """
    #     encode to style space
    #     """
    #     modules = list(self.input_blocks.modules()) + list(
    #         self.middle_block.modules()) + list(self.output_blocks.modules())
    #     # (n, c)
    #     cond = self.encoder.forward(x)
    #     S = []
    #     for module in modules:
    #         if isinstance(module, ResBlock):
    #             # (n, c')
    #             s = module.cond_emb_layers.forward(cond)
    #             S.append(s)

    #     if return_vector:
    #         # (n, sum_c)
    #         return torch.cat(S, dim=1)
    #     else:
    #         return S
    
    def forward(self, x, t, latent):
        t_cond = t
        
        cond = latent

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        # style = style or res.style

        # assert (y is not None) == (
        #     self.conf.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        # if self.conf.num_classes is not None:
        #     raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    # print(i, j, h.shape)
                    # print('emb', enc_time_emb.shape)
                    # print('cond', enc_cond_emb.shape)
                    h = self.input_blocks[k](h,
                                             emb=enc_time_emb,
                                             cond=enc_cond_emb)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        # return AutoencReturn(pred=pred, cond=cond)
        return pred


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style) 