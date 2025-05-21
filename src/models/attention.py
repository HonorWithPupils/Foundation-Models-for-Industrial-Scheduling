import torch
from torch import nn, Tensor
from torch.nn import functional as F


from typing import Optional, Union, Callable


class SelfAttentions(nn.Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_head: int,
        dim_mlp: int,
        dropout: float = 0.0,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            # Decoder-only in LLMs is more similar to Encoder in Transformer
            # So, we use TransformerEncoderLayer
            self.layers.append(
                torch.nn.TransformerEncoderLayer(
                    dim,
                    n_head,
                    dim_mlp,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )
            )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return x
