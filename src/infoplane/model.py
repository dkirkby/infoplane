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
    def activations(self, x: torch.Tensor, include_output: bool = False) -> list[torch.Tensor]:
        """Activations to measure: one ``(N, width)`` tensor per hidden layer.

        ``include_output`` appends the readout as well.  The paper counts it as a
        layer -- §4.1: "layer 1 is the hidden layer closest to the input and the
        highest is the output layer" -- and the authors' code measures it without
        comment, because ``Model.hidden_layers`` appends the final softmax layer to
        the same list the activations are read from (IDNNs
        ``idnns/networks/model.py``, and ``network.exctract_activity`` then reads
        all of them).  It is the most compressed point on every trajectory, so
        leaving it out drops the left end of the paper's figures.
        """
        out = []
        for layer in self.hidden:
            x = torch.tanh(layer(x))
            out.append(x)
        if include_output:
            out.append(self.output_probs(x))
        return out

    @torch.no_grad()
    def output_probs(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """The readout as a ``(N, n_classes)`` probability vector.

        A single sigmoid unit and a 2-way softmax are the same model, but they are
        not the same *measurement*: the authors bin both columns of the softmax, so
        two samples share an output symbol only if ``p`` and ``1 - p`` both land in
        the same bin.  Returning the pair keeps our binning identical to theirs
        whichever readout the network was trained with.
        """
        logits = self.readout(last_hidden)
        if self.n_out == 1:
            p = torch.sigmoid(logits)
            return torch.cat([p, 1.0 - p], dim=1)
        return torch.softmax(logits, dim=1)

    def layer_names(self, include_output: bool = False) -> list[str]:
        names = [f"Layer {i + 1}" for i in range(len(self.hidden))]
        if include_output:
            names.append("Output")
        return names
