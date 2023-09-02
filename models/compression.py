from typing import Any, Dict, List, Optional, Tuple, Mapping
import torch
import math
import torch.nn as nn
from compressai.latent_codecs import LatentCodec, GaussianConditionalLatentCodec, GainHyperLatentCodec

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

def GainHyperLatentCodecWrapper():
    entropy_bottleneck = EntropyBottleneck(40)
    return GainHyperLatentCodec(entropy_bottleneck)


def GaussianConditionalLatentCodecWrapper():
    entropy_parameters = nn.Conv2d(40, 80, 1, padding=0)
    # entropy_parameters = nn.Linear(20, 40)
    return GaussianConditionalLatentCodec(entropy_parameters=entropy_parameters)




class GainHyperpriorLatentCodecFixed(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Gain-controlled hyperprior introduced in
    `"Asymmetric Gained Deep Image Compression With Continuous Rate Adaptation"
    <https://arxiv.org/abs/2003.02012>`_, by Ze Cui, Jing Wang,
    Shangyin Gao, Bo Bai, Tiansheng Guo, and Yihui Feng, CVPR, 2021.

    .. code-block:: none

                z_gain  z_gain_inv
                   │        │
                   ▼        ▼
                  ┌┴────────┴┐
            ┌──►──┤ lc_hyper ├──►─┐
            │     └──────────┘    │
            │                     │
            │     y_gain          ▼ params   y_gain_inv
            │        │            │              │
            │        ▼            │              ▼
            │        │         ┌──┴───┐          │
        y ──┴────►───×───►─────┤ lc_y ├────►─────×─────►── y_hat
                               └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                        z_gain                      z_gain_inv
                           │                             │
                           ▼                             ▼
                 ┌───┐  z  │ z_g ┌───┐ z_hat      z_hat  │       ┌───┐
            ┌─►──┤h_a├──►──×──►──┤ Q ├───►───····───►────×────►──┤h_s├──┐
            │    └───┘           └───┘        EB                 └───┘  │
            │                                                           │
            │                              ┌──────────────◄─────────────┘
            │                              │            params
            │                           ┌──┴──┐
            │    y_gain                 │  EP │    y_gain_inv
            │       │                   └──┬──┘        │
            │       ▼                      │           ▼
            │       │       ┌───┐          ▼           │
        y ──┴───►───×───►───┤ Q ├────►────····───►─────×─────►── y_hat
                            └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """

    latent_codec: Mapping[str, LatentCodec]

    def __init__(
        self, latent_codec: Optional[Mapping[str, LatentCodec]] = None, **kwargs
    ):
        super().__init__()

        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                # "y": GaussianConditionalLatentCodecWrapper,
                "y": GaussianConditionalLatentCodecWrapper,
                "hyper": GainHyperLatentCodecWrapper,
            },
            save_direct=True,
        )

        self.entropy_bottleneck = self.latent_codec["hyper"].entropy_bottleneck

    def forward(
        self,
        y: Tensor,
        y_gain: Tensor,
        z_gain: Tensor,
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
    ) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"](y, z_gain, z_gain_inv)
        a = y * y_gain
        y_out = self.latent_codec["y"](y * y_gain, hyper_out["params"])
        y_hat = y_out["y_hat"] * y_gain_inv
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
            "y_hat": y_hat,
        }

    def compress(
        self,
        y: Tensor,
        y_gain: Tensor,
        z_gain: Tensor,
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
    ) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"].compress(y, z_gain, z_gain_inv)
        y_out = self.latent_codec["y"].compress(y * y_gain, hyper_out["params"])
        y_hat = y_out["y_hat"] * y_gain_inv
        return {
            "strings": [*y_out["strings"], *hyper_out["strings"]],
            "shape": {"y": y_out["shape"], "hyper": hyper_out["shape"]},
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Dict[str, Tuple[int, ...]],
        y_gain_inv: Tensor,
        z_gain_inv: Tensor,
    ) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec["hyper"].decompress(
            [z_strings], shape["hyper"], z_gain_inv
        )
        y_out = self.latent_codec["y"].decompress(
            y_strings_, shape["y"], hyper_out["params"]
        )
        y_hat = y_out["y_hat"] * y_gain_inv
        return {"y_hat": y_hat}
