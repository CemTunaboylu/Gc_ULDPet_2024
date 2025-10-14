from typing import Union

import einops
import torch
from torch import nn

LAST_LAYER = -1

""" Feature-wise Linear Modulation """
class FiLMIntercept(nn.Module):
    def __init__(self, seq:nn.Sequential, module_key='adn'):
        super().__init__()
        self.seq = seq
        """
            By the default, Convolution creates a nn.Sequential with ADN module within,
            forward pass pipeline thus includes ADN, and needs to be intercepted by the FiLM layer.
            The parent Block/Layer to contain this block, will have an interception module instead, which
            steals the ADN module from the desired convolution layer (default=last). Thus, during its forward pass,
            the parent will inject the current gamma and beta values to the wrapping intercept. Before the ADN module,
            which is removed from the _modules of Convolution module, will be applied after the intercept's application of FiLM.
                For 2Conv ->  x -> | Conv | ADN | Conv | FiLM | ADN(applied by intercept)
        """
        self.stolen_module = self.seq._modules.pop(module_key)
        self.gamma : torch.Tensor = torch.tensor([])
        self.beta : torch.Tensor = torch.tensor([])

    def inject(self, gamma:torch.Tensor, beta:torch.Tensor):
        self.gamma = gamma
        self.beta = beta

    # be lazy reshape when called
    def modulate(self, feature_map:torch.Tensor)->torch.Tensor:
        if 0 == self.gamma.shape[0] or 0 == self.beta.shape[0]:
            raise ValueError("Gamma and beta must be set before calling forward.")

        # gamma and beta have shape: (batch_size, num_features)
        # we want to reshape them to (batch_size, num_features, 1, 1, 1)
        reshape = 'b nf -> b nf 1 1 1'
        gamma = einops.rearrange(self.gamma, reshape)
        beta = einops.rearrange(self.beta, reshape)
        # reset them to be safe
        self.gamma = torch.tensor([])
        self.beta = torch.tensor([])

        return gamma * feature_map + beta

    def forward(self, x:torch.Tensor)->torch.Tensor:
        for m in [self.seq, self.modulate, self.stolen_module]:
            x = m(x)
        return x

    @staticmethod
    def module_name()->str:
        return 'filmed'
