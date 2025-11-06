# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Class to create a positional encoding vector for timeseries inputs to a model without an explicit time dimension.

    This class implements a sin/cos type embedding vector with a specified maximum length. Adapted from the PyTorch
    example here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Parameters
    ----------
    embedding_dim : int
        Dimension of the model input, which is typically output of an embedding layer.
    dropout : float
        Dropout rate [0, 1) applied to the embedding vector.
    max_len : int, optional
        Maximum length of positional encoding. This must be larger than the largest sequence length in the sample.
    """

    def __init__(self, embedding_dim, position_type, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(embedding_dim / 2) * 2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(max_len * 2) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :embedding_dim].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        if position_type.lower() == 'concatenate':
            self._concatenate = True
        elif position_type.lower() == 'sum':
            self._concatenate = False
        else:
            raise RuntimeError(
                f'Unrecognized positional encoding type: {position_type}'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding. Either concatenates or adds positional encoding to encoder input data.

        Parameters
        ----------
        x : torch.Tensor
            Dimension is ``[sequence length, batch size, embedding output dimension]``.
            Data that is to be the input to a transformer encoder after including positional encoding.
            Typically this will be output from an embedding layer.

        Returns
        -------
        torch.Tensor
            Dimension is ``[sequence length, batch size, encoder input dimension]``.
            The encoder input dimension is either equal to the embedding output dimension (if ``position_type == sum``)
            or twice the embedding output dimension (if ``position_type == concatenate``).
            Encoder input augmented with positional encoding.

        """
        if self._concatenate:
            x = torch.cat(
                (x, self.pe[: x.size(0), :].repeat(1, x.size(1), 1)), 2
            )
        else:
            x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
