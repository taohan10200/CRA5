# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from typing import Optional
from compressai.ans import BufferedRansEncoder, RansDecoder
from cra5.models.compressai.entropy_models import EntropyBottleneck, GaussianConditional
from cra5.models.compressai.layers import GDN, MaskedConv2d
from cra5.models.compressai.registry import register_model

from cra5.models.compressai.models.base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from cra5.models.compressai.models.utils import conv, deconv
from cra5.registry import MODELS
from cra5.models.vaeformer.vit_nlc import Encoder, Decoder,HyperPriorEncoder, HyperPriorDecoder
from cra5.models.vaeformer.modules.distributions import DiagonalGaussianDistribution
from cra5.models.compressai.models.google import ScaleHyperprior
from collections import OrderedDict
        
__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
    "VaritionInVaration_CNN_Prior"
]



# @register_model("mbt2018-mean")

class VaritionInVaration_CNN_Prior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self,
                 embed_dim,
                 z_channels,
                 y_channels,
                 sample_posterior,
                 ddconfig,
                 rate_distortion_loss,
                 kl_loss=None,
                 pretrained_vae=None,
                 frozen_encoder=False,
                 lower_dim = False,
                 ignore_keys:list=[],
                 **kwargs):
        super().__init__(N=z_channels, M=embed_dim, **kwargs)
        self.sample_posterior = sample_posterior
        self.lower_dim= lower_dim
        self.frozen_encoder = frozen_encoder
        
        self.g_a = Encoder(**ddconfig)
        self.g_s = Decoder(**ddconfig)
        if  self.lower_dim:
            self.quant_conv = torch.nn.Conv2d(2*y_channels, 2*embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, y_channels, 1)
        self.h_a = nn.Sequential(
            conv(embed_dim, z_channels, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(z_channels, z_channels),
            nn.LeakyReLU(inplace=True),
            conv(z_channels, z_channels),
        )

        self.h_s = nn.Sequential(
            deconv(z_channels, embed_dim),
            nn.LeakyReLU(inplace=True),
            deconv(embed_dim, embed_dim * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(embed_dim * 3 // 2, embed_dim * 2, stride=1, kernel_size=3),
        )
        if rate_distortion_loss is not None:
            self.criterion = MODELS.build(rate_distortion_loss)
        if kl_loss is not None:
            self.kl_loss = MODELS.build(kl_loss)
        if pretrained_vae is not None:
            self.init_from_ckpt(pretrained_vae, ignore_keys=ignore_keys)

    def init_from_ckpt(self, ckpt, ignore_keys=list()):
        last_saved: Optional[str]
        if isinstance(ckpt, str):
            if ckpt.endswith('.pth'):
                last_saved = ckpt
            else:
                save_file = osp.join(ckpt, 'last_checkpoint')

                if osp.exists(save_file):
                    with open(save_file) as f:
                        last_saved = f.read().strip()
                else:
                    raise ValueError(f"You do not have a saved checkpoint to restore, "
                                    f"please set the load path: {ckpt} as None in config file")

            sd = torch.load(last_saved, map_location="cpu")["state_dict"]
        else:
            sd=ckpt
        
        from collections import OrderedDict
        ga_state_dict = OrderedDict()
        gs_state_dict = OrderedDict()
        loss_state_dict = OrderedDict()
        for k, v in sd.items(): #if k in model_dict.keys()
            skip = [ True  for ik in ignore_keys if k.startswith(ik)]
            if len(skip) > 0:
                print("Deleting key {} from state_dict.".format(k))
                continue
            if 'encoder' in k:
                ga_state_dict[k.replace("backbone.encoder.","")] = v
            if 'decoder' in k:
                gs_state_dict[k.replace("backbone.decoder.","")] = v
            if 'logvar' in k:
                loss_state_dict[k.replace("backbone.loss.","")] = v
        self.g_a.load_state_dict(ga_state_dict, strict=True)
        self.g_s.load_state_dict(gs_state_dict, strict=True)
        self.kl_loss.load_state_dict(loss_state_dict, strict=True)
        if self.frozen_encoder:
            for param in self.g_a.parameters():
                param.requires_grad = False

        print(f"Restored from {last_saved}, and make the frozen_encoder as {self.frozen_encoder}" )
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def training_step(self, inputs, batch_idx, optimizer_idx):
        out_net = self(inputs)
        out_criterion = self.criterion(out_net, inputs)
        discloss = self.kl_loss(inputs, out_net['x_hat'], out_net['posterior'],
                                optimizer_idx, 0,
                                last_layer=self.get_last_layer(), split="train")

        # import pdb
        # pdb.set_trace()
        return  {**discloss, **out_criterion, "aux_loss":self.aux_loss()}
    def prediction(self, inputs):
        # import pdb
        # pdb.set_trace()
        t1 = time.time()
        out = self.compress(inputs)
        t2 =  time.time()
        x_hat = self.decompress(out['strings'], out['shape'])
        t3 =  time.time()

        # import numpy as np
        # np.save('./exp/vivt_69dim_gt.npy',inputs.cpu().numpy())
        # np.save('./exp/vivt_69dim_pred.npy',x_hat['x_hat'].cpu().numpy())
        # import pdb
        # pdb.set_trace()

        return {
            **x_hat,
            "strings":out['strings'],
            'shape':inputs.shape,
            'encoding_time':(t2-t1)/inputs.size(0),
            'decoding_time':(t3-t2)/inputs.size(0)}


    def forward(self, x):
        moments = self.g_a(x)
        posterior = None
        if self.lower_dim:
            moments = self.quant_conv(moments)

        posterior = DiagonalGaussianDistribution(moments)
        if self.sample_posterior:
            y = posterior.sample()
        else:
            y = posterior.mode()
        
        z = self.h_a(y.detach())

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        if self.lower_dim:
            y_hat = self.post_quant_conv(y_hat)

        x_hat = self.g_s(y_hat)

        return {

            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "posterior": posterior
        }

    def compress(self, x):
        moments = self.g_a(x)
        if self.lower_dim:
            moments = self.quant_conv(moments)
        
        posterior = DiagonalGaussianDistribution(moments)
        if self.sample_posterior:
            y = posterior.sample()
        else:
            y = posterior.mode()            

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        # bpp = sum(len(s[0]) for s in [y_strings, z_strings]) * 8.0 #/ num_pixels

        # import numpy as np
        # np.save('./exp/test_npy_size_4x69x128x256.npy',x.cpu().numpy())
        #
        # np.save('./exp/test_npy_size_4x69x8x16.npy',z.cpu().numpy())
        # np.save('./exp/test_npy_size_4x69x32x64.npy',y.cpu().numpy())
        # import pdb
        # pdb.set_trace()
        # #
        # import pickle
        # with open('./exp/test_y_strings_size_1x256x72x144.csv', 'wb') as f:
        #     test_data = pickle.dump(y_strings, f)
        # with open('./exp/test_z_strings_size_1x256x18x36.csv', 'wb') as f:
        #     test_data = pickle.dump(z_strings, f)
        # import pdb
        # pdb.set_trace()

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        
        if self.lower_dim:
            y_hat = self.post_quant_conv(y_hat)
        x_hat = self.g_s(y_hat) #.clamp_(0, 1)

        return {"x_hat": x_hat}

    def get_last_layer(self):
        return self.g_s.final.weight


@MODELS.register_module()
class MeanScaleHyperprior_Baseline(ScaleHyperprior):
    """Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self,
                 embed_dim,
                 z_channels,
                 sample_posterior,
                 ddconfig,
                 rate_distortion_loss,

                 **kwargs):
        super().__init__(N=z_channels, M=embed_dim, **kwargs)
        self.sample_posterior = sample_posterior
        self.g_a = Encoder(**ddconfig)
        self.g_s = Decoder(**ddconfig)
        self.h_a = nn.Sequential(
            conv(embed_dim, z_channels, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(z_channels, z_channels),
            nn.LeakyReLU(inplace=True),
            conv(z_channels, z_channels),
        )

        self.h_s = nn.Sequential(
            deconv(z_channels, embed_dim),
            nn.LeakyReLU(inplace=True),
            deconv(embed_dim, embed_dim * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(embed_dim * 3 // 2, embed_dim * 2, stride=1, kernel_size=3),
        )
        self.lower_channel=conv(2*embed_dim, embed_dim, stride=1, kernel_size=3)
        self.criterion = MODELS.build(rate_distortion_loss)

    def training_step(self, inputs, batch_idx, optimizer_idx):
        out_net = self(inputs)
        out_criterion = self.criterion(out_net, inputs)

        # import pdb
        # pdb.set_trace()
        return  { **out_criterion, "aux_loss":self.aux_loss()}
    def prediction(self, inputs):
        # import pdb
        # pdb.set_trace()
        t1 = time.time()
        out = self.compress(inputs)
        t2 =  time.time()
        x_hat = self.decompress(out['strings'], out['shape'])
        t3 =  time.time()

        return {
            **x_hat,
            "strings":out['strings'],
            'shape':inputs.shape,
            'encoding_time':(t2-t1)/inputs.size(0),
            'decoding_time':(t3-t2)/inputs.size(0)}

    def forward(self, x):
        y = self.g_a(x)
        y = self.lower_channel(y)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        y = self.lower_channel(y)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)   #.clamp_(0, 1)
        return {"x_hat": x_hat}

    def get_last_layer(self):
        return self.g_s.final.weight

@MODELS.register_module()
class VaritionInVaration_Former_Baseline(MeanScaleHyperprior_Baseline):
    """
    Args:
    N (int): Number of channels
    M (int): Number of channels in the expansion layers (last layer of the
    encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, embed_dim, z_channels,sample_posterior,
                 pretraind_vae=None,
                 frozen_encoder=None,
                 ddconfig=None,
                 priorconfig=None,
                 rate_distortion_loss=None,
                 ignore_keys:list=[],
                 **kwargs):
        super().__init__(embed_dim=embed_dim,
                         z_channels=z_channels,
                         sample_posterior=sample_posterior,
                         ddconfig=ddconfig,
                         rate_distortion_loss=rate_distortion_loss,
                          **kwargs)
        self.sample_posterior = sample_posterior
        self.frozen_encoder = frozen_encoder
        self.g_a = Encoder(**ddconfig)
        self.g_s = Decoder(**ddconfig)
        self.h_a = HyperPriorEncoder(**priorconfig)
        self.h_s = HyperPriorDecoder(**priorconfig)

        if pretraind_vae is not None:
            self.init_from_ckpt(pretraind_vae, ignore_keys=ignore_keys)


    def init_from_ckpt(self,path,ignore_keys=list()):
        last_saved: Optional[str]
        if path.endswith('.pth'):
            last_saved = path
        else:
            save_file = osp.join(path, 'last_checkpoint')

            if osp.exists(save_file):
                with open(save_file) as f:
                    last_saved = f.read().strip()
            else:
                raise ValueError(f"You do not have a saved checkpoint to restore, "
                                 f"please set the load path: {path} as None in config file")

        sd = torch.load(last_saved, map_location="cpu")["state_dict"]
        from collections import OrderedDict
        ga_state_dict = OrderedDict()
        gs_state_dict = OrderedDict()

        for k, v in sd.items(): #if k in model_dict.keys()
            skip = [ True  for ik in ignore_keys if k.startswith(ik)]
            if len(skip) > 0:
                print("Deleting key {} from state_dict.".format(k))
                continue
            if 'encoder' in k:
                ga_state_dict[k.replace("backbone.encoder.","")] = v
            if 'decoder' in k:
                gs_state_dict[k.replace("backbone.decoder.","")] = v

        self.g_a.load_state_dict(ga_state_dict, strict=True)
        self.g_s.load_state_dict(gs_state_dict, strict=True)

        if self.frozen_encoder:
            for param in self.g_a.parameters():
                param.requires_grad = False

        print(f"Restored from {last_saved}, and make the frozen_encoder as {self.frozen_encoder}" )

