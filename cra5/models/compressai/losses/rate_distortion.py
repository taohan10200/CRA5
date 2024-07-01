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

import math

import torch
import torch.nn as nn

from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, bpp_weight = 0.01,  metric="mse", return_type="all",
                 learn_log_variance=False,
                 logvar_init=0.0,
                 in_channels=69):
        super().__init__()
        self.metric=None
        self.learn_log_variance=learn_log_variance
        
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        # else:
        #     raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.bpp_weight =bpp_weight
        self.return_type = return_type

        if learn_log_variance:
            self.logvar = nn.Parameter(torch.ones(size=(1, in_channels,1,1)) * logvar_init)

        
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] =  self.bpp_weight * sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric is not None:
            if self.metric == ms_ssim:
                out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
                distortion = 1 - out["ms_ssim_loss"]
            else:
                if self.learn_log_variance:   
                    rec_loss =torch.square(target.contiguous()-output["x_hat"]) 
                    rec_loss = rec_loss / torch.exp(self.logvar) + self.logvar
                    out["mse_loss"] =rec_loss.mean()
                else:
                    out["mse_loss"] = self.lmbda*self.metric(output["x_hat"], target)

        # out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        # out["bpp"] = out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
