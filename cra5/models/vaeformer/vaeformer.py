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


@MODELS.register_module()
class VAEformer(CompressionModel):
    """
    Args:
    N (int): Number of channels
    M (int): Number of channels in the expansion layers (last layer of the
    encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, 
                 model_version,
                 embed_dim=None, 
                 z_channels=None,
                 y_channels=None,
                 sample_posterior=None, 
                 pretrained_vae=None, 
                 frozen_encoder=None, 
                 ddconfig=None, 
                 priorconfig=None,
                 rate_distortion_loss=None, 
                 kl_loss=None, 
                 ignore_keys:list=[], 
                 lower_dim= False,
                 **kwargs):
        if model_version == 268:
            embed_dim=256
            z_channels=256
            y_channels=1024
            lower_dim=True
            sample_posterior =False
            pretrained_vae = None #'./exp/comp/era5_autoencoder_ps10_159v/iter_150000.pth',
            frozen_encoder=False
            ddconfig=dict(
                arch = 'vit_large',
                pretrained_model = '',
                patch_size=(11,10),
                patch_stride=(10,10),
                in_chans=268,
                out_chans=268,
                kwargs=dict(
                    z_dim =  None,
                    learnable_pos= True,
                    window= True,
                    window_size = [(24, 24), (12, 48), (48, 12)],
                    interval = 4,
                    drop_path_rate= 0.,
                    round_padding= True,
                    pad_attn_mask= True , # to_do: ablation
                    test_pos_mode= 'learnable_simple_interpolate', # to_do: ablation
                    lms_checkpoint_train= True,
                    img_size= (721, 1440)
                ),
            )
            priorconfig = dict(
                pretrained_model = '', # '../PretrainedModels/maevit/mae_pretrain_vit_large.pth',
                patch_size=(4,4),
                in_chans=256,
                out_chans=256,
                kwargs=dict(
                    z_dim = 256,
                    embed_dim=360,
                    depth=8,
                    num_heads=5,
                    interval=1,
                    learnable_pos= True,
                    window= False,
                    drop_path_rate= 0.,
                    round_padding= True,
                    pad_attn_mask= True , # to_do: ablation
                    test_pos_mode= 'learnable_simple_interpolate', # to_do: ablation
                    lms_checkpoint_train= False,
                    img_size= (72,144)
                ),
            )
        super().__init__(**kwargs)
        self.sample_posterior = sample_posterior
        self.lower_dim = lower_dim
        self.frozen_encoder = frozen_encoder
        
        self.entropy_bottleneck = EntropyBottleneck(z_channels)
        
        self.g_a = Encoder(**ddconfig)
        self.g_s = Decoder(**ddconfig)
        
        if  self.lower_dim:
            self.quant_conv = torch.nn.Conv2d(2*y_channels, 2*embed_dim, 1)
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, y_channels, 1)
            
        self.h_a = HyperPriorEncoder(**priorconfig)
        self.h_s = HyperPriorDecoder(**priorconfig)
        self.gaussian_conditional = GaussianConditional(None)
        
        if rate_distortion_loss is not None:
            self.criterion = MODELS.build(rate_distortion_loss)
        if kl_loss is not None:
            self.kl_loss = MODELS.build(kl_loss)
        if pretrained_vae is not None:
            self.init_from_ckpt(pretrained_vae, ignore_keys=ignore_keys)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""

        variable_num = state_dict["backbone.g_a.patch_embed.proj.weight"].size(1)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'kl_loss.logvar' not in k:
                new_state_dict[k.replace("backbone.", "")] = v
            

        net = cls(variable_num)
        # net.update(force=True)
        
        net.load_state_dict(new_state_dict)
        
        return net

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
            
        ga_state_dict = OrderedDict()
        gs_state_dict = OrderedDict()
        quant_conv_state = OrderedDict()
        post_quant_conv_state = OrderedDict()
        
        loss_state_dict = OrderedDict()

        for k, v in sd.items(): #if k in model_dict.keys()
            skip = [ True  for ik in ignore_keys if k.startswith(ik)]
            if len(skip) > 0:
                print("Deleting key {} from state_dict.".format(k))
                continue
            if 'encoder' in k:
                ga_state_dict[k.replace("backbone.encoder.", "")] = v
            if 'decoder' in k:
                gs_state_dict[k.replace("backbone.decoder.", "")] = v
            if 'logvar' in k:
                loss_state_dict[k.replace("backbone.loss.", "")] = v
            if 'quant_conv' in k and 'post_quant_conv' not in k:
                quant_conv_state[k.replace("backbone.quant_conv.", "")] = v
            if 'post_quant_conv' in k:
                post_quant_conv_state[k.replace("backbone.post_quant_conv.", "")] = v

        self.g_a.load_state_dict(ga_state_dict, strict=True)
        self.g_s.load_state_dict(gs_state_dict, strict=True)
        self.quant_conv.load_state_dict(quant_conv_state,strict=True)
        self.post_quant_conv.load_state_dict(post_quant_conv_state, strict=True)
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
            "z_shape":out['shape'],
            'x_shape':inputs.shape,
            'encoding_time':(t2-t1)/inputs.size(0),
            'decoding_time':(t3-t2)/inputs.size(0)}


    def encode_latent(self, x, type='quantized'):
        moments = self.g_a(x)
        posterior = None
        if self.lower_dim:
            moments = self.quant_conv(moments)

        posterior = DiagonalGaussianDistribution(moments)
        if self.sample_posterior:
            y = posterior.sample()
        else:
            y = posterior.mode()
            
        if type == "quantized":
            z = self.h_a(y.detach())
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
            return y, y_hat, y_likelihoods
        else:
            return y, None, None
        
    def decode_latent(self, y, type='quantized'):
        if self.lower_dim:
            y_hat = self.post_quant_conv(y)

        x_hat = self.g_s(y_hat)
        
        return x_hat
         
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

    def compress(self, 
                 x,
             ):
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

        return {"strings": [y_strings, z_strings], "z_shape": z.size()[-2:]}

    def decompress(self, 
                   strings, 
                   shape,
                   return_format: str='reconstructed'):
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
            
        if return_format=='latent':
            return y_hat
        
        x_hat = self.g_s(y_hat) #.clamp_(0, 1)

        return {"x_hat": x_hat}

    def get_last_layer(self):
        return self.g_s.final.weight

