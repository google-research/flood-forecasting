"""Utilities for lstm setup."""

from collections.abc import Iterable

import torch
from torch import nn

from googlehydrology.utils.config import WeightInitOpt

LSTM_IH_XAVIER = WeightInitOpt.LSTM_IH_XAVIER
LSTM_HH_ORTHOGONAL = WeightInitOpt.LSTM_HH_ORTHOGONAL
FC_XAVIER = WeightInitOpt.FC_XAVIER


def lstm_init(
    *,
    lstms: Iterable[nn.LSTM],
    forget_bias: float | None = None,
    weight_opts: Iterable[WeightInitOpt] = (),
) -> None:
    """Initialize LSTM weights."""
    with torch.no_grad():
        for lstm in lstms:
            if forget_bias is not None:
                lstm.bias_hh_l0.data[_forget_gate_slice(lstm)] = forget_bias

            for name, param in lstm.named_parameters():
                if 'weight_ih' in name and LSTM_IH_XAVIER in weight_opts:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name and LSTM_HH_ORTHOGONAL in weight_opts:
                    nn.init.orthogonal_(param)


def _forget_gate_slice(lstm: nn.LSTM) -> slice:
    """Return the slice to access an LGTM forget gate params.

    Gates' data lengths are the hidden size, and appear in order of:
      input, forget, cell, output
    """
    return slice(lstm.hidden_size, 2 * lstm.hidden_size)
