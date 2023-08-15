from torch import nn


class SplitModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError(" Every model should implement a forward pass returning a dict with"
                                  "        output = {y_hat: Tensor with grads,"
                                  "                  likelihoods: Tensor with grads,"
                                  "                  num_bytes: float,"
                                  "                  compression_loss: Tensor with grads }")

    def compress(self, x):
        raise NotImplementedError(" Every model should implement compress returning a dict with"
                                  "        output = {num_bytes: float}  ")
