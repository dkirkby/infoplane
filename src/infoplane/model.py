"""The paper's fully connected network, plus hidden-activation capture."""

from __future__ import annotations

import torch
from torch import nn

# Shwartz-Ziv & Tishby (2017): "7 fully connected hidden layers of widths
# 12-10-7-5-4-3-2".  These are the *hidden* widths; the input is separately
# 12-dimensional and the output is a single sigmoid unit.  The figures' "Layer 1"
# ... "Layer 7" are exactly these.
PAPER_HIDDEN_WIDTHS: tuple[int, ...] = (12, 10, 7, 5, 4, 3, 2)


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
