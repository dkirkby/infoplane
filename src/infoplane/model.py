"""The paper's fully connected network, plus hidden-activation capture."""

from __future__ import annotations

import torch
from torch import nn

# Shwartz-Ziv & Tishby (2017) §4.1 writes the network as "12-10-7-5-4-3-2" and
# calls it "7 fully connected hidden layers".  Read literally that gives seven
# hidden layers, but the paper's own Figure 4 caption writes a network as
# "input=12-10-8-6-4-2-1=output" -- the same seven-number format with the first
# entry the input and the last the output -- and §4.1 adds that "layer 1 is the
# hidden layer closest to the input and the highest is the output layer".  Under
# that convention "12-10-7-5-4-3-2" is a 12-dim input, FIVE hidden layers of
# 10-7-5-4-3, and a 2-unit output, which is exactly what the authors' code builds
# (`select_network_arch('1') -> [10, 7, 5, 4, 3]`, IDNNs
# `idnns/networks/network_paramters.py`).  Figure 2 shows about five or six
# clusters rather than seven, which agrees.
#
# So these are the five hidden widths; the input is separately 12-dimensional and
# the output is a single sigmoid unit (the authors use a 2-way softmax, which is
# the same model up to parameterisation).  The figures' "Layer 1" ... "Layer 5"
# are exactly these.  Taking the literal seven-layer reading instead inserts a
# 12-wide and a 2-wide hidden layer: the extra wide layer is pinned at
# `I(X;T) = H(X) = 12` bits and can never move, and the 2-unit tanh bottleneck
# makes the network measurably harder to optimise.
PAPER_HIDDEN_WIDTHS: tuple[int, ...] = (10, 7, 5, 4, 3)


class InfoPlaneNet(nn.Module):
    """``d -> widths -> n_out`` MLP with ``tanh`` hidden units.

    Two properties matter for the information-plane measurement and should be
    preserved by anything built on this class:

    * hidden activations live in ``[-1, 1]`` (``tanh``), which is what lets
      :mod:`infoplane.mi` bin them on a fixed, known support.  Swapping in ReLU
      breaks the binning estimator *and* the phenomenon being replicated.
    * the model is seeded through ``torch.random.fork_rng``, so constructing it
      never perturbs the caller's global RNG -- worker processes get reproducible
      weights independent of anything else they do.
    """

    def __init__(
        self,
        n_features: int,
        widths: tuple[int, ...] = PAPER_HIDDEN_WIDTHS,
        n_out: int = 1,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.n_out = n_out
        sizes = [n_features, *self.widths]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.ModuleList(
                nn.Linear(a, b, dtype=dtype) for a, b in zip(sizes[:-1], sizes[1:])
            )
            self.readout = nn.Linear(sizes[-1], n_out, dtype=dtype)
            for layer in (*self.hidden, self.readout):
                # Truncated normal with std 1/sqrt(fan_in), as in the authors' code.
                fan_in = layer.weight.shape[1]
                nn.init.trunc_normal_(layer.weight, std=fan_in**-0.5, a=-2 * fan_in**-0.5, b=2 * fan_in**-0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the readout *logits* (no sigmoid/softmax applied)."""
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        return self.readout(x)

    @torch.no_grad()
    def activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Hidden-layer activations, one ``(N, width)`` tensor per hidden layer."""
        out = []
        for layer in self.hidden:
            x = torch.tanh(layer(x))
            out.append(x)
        return out

    def layer_names(self) -> list[str]:
        return [f"Layer {i + 1}" for i in range(len(self.hidden))]
