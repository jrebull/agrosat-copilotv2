"""U-TAE — U-Net with Lightweight Temporal Attention Encoder (Garnot & Landrieu 2021).

Temporal semantic segmentation architecture for Sentinel-2 series: a 2D U-Net
encodes each timestep independently, an L-TAE (Lightweight Temporal Attention
Encoder) aggregates the temporal dimension at the bottleneck, and the temporal
attention weights re-weight the skip connections before the decoder.

Input ``(B, T, C_in, H, W)`` (multi-temporal patch) + ``batch_positions``
``(B, T)`` (day-of-year for the positional encoding). Output
``(B, num_classes, H, W)``.

This implementation is the verbatim port (same dimensions and module names) of
the model trained by Isaac in ``notebooks/segmentation/04j_segmentation_utae``,
so its checkpoint (``best_model.pt``, ``model_state_dict`` with keys
``in_conv`` / ``down_convs`` / ``temporal_encoder`` / ...) loads without
renaming. It is ported to a module to be able to build the model outside the
notebook (Optuna, inference) respecting separation of concerns (CLAUDE.md rule 8).

Reference: V. Sainte Fare Garnot, L. Landrieu, "Panoptic Segmentation of
Satellite Image Time Series with Convolutional Temporal Attention Networks",
ICCV 2021.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UTAE", "build_utae"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to a ``(B, T, C, H, W)`` series."""

    def __init__(self, d: int, T: int = 1000, repeat: int | None = None) -> None:
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Add the positional encoding per timestep.

        Args:
            x: Series ``(B, T, C, H, W)``.
            positions: Temporal positions ``(B, T)`` (integer day-of-year).

        Returns:
            ``x`` with the positional encoding added, same shape.
        """
        _, _, C, _, _ = x.shape
        pe = self._get_pe(positions, C, x.device)  # (B, T, C)
        pe = pe.unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
        return x + pe

    def _get_pe(self, positions: torch.Tensor, d: int, device: torch.device) -> torch.Tensor:
        B, T = positions.shape
        div = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(self.T) / d)
        )  # (d//2,)
        pos = positions.float().unsqueeze(-1)  # (B, T, 1)
        pe = torch.zeros(B, T, d, device=device)
        pe[:, :, 0::2] = torch.sin(pos * div)
        pe[:, :, 1::2] = torch.cos(pos * div)
        return pe


class LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder applied per spatial position.

    Input ``(B, T, C, H, W)`` -> output ``(B, C_out, H, W)`` (time-aggregated
    map). Optionally returns the attention weights ``(B, n_head, T, H, W)`` to
    re-weight the skip connections.
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp_in: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.return_att = return_att
        self.d_k = d_k

        if positional_encoding:
            self.positional_encoder: PositionalEncoding | None = PositionalEncoding(
                d=in_channels, T=T, repeat=None
            )
        else:
            self.positional_encoder = None

        self.inlayernorm = nn.LayerNorm(in_channels)
        self.outlayernorm = nn.LayerNorm(mlp_in[-1])
        self.key_net = nn.Sequential(nn.Linear(in_channels // n_head, d_k))
        self.query = nn.Sequential(nn.Linear(in_channels, in_channels), nn.ReLU())

        layers: list[nn.Module] = []
        for i in range(len(mlp_in) - 1):
            layers.extend(
                [nn.Linear(mlp_in[i], mlp_in[i + 1]), nn.BatchNorm1d(mlp_in[i + 1]), nn.ReLU()]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        batch_positions: torch.Tensor | None = None,
        return_att: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Aggregate the temporal dimension by attention.

        Args:
            x: Series ``(B, T, C, H, W)``.
            batch_positions: Temporal positions ``(B, T)`` or ``None``.
            return_att: If ``True`` also returns the attention weights.

        Returns:
            ``out`` ``(B, C_out, H, W)``, or ``(out, att)`` with
            ``att`` ``(B, n_head, T, H, W)`` if ``return_att``/``self.return_att``.
        """
        B, T, C, H, W = x.shape

        if self.positional_encoder is not None and batch_positions is not None:
            x = self.positional_encoder(x, batch_positions)

        x_flat = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        x_flat = self.inlayernorm(x_flat)

        q = self.query(x_flat.mean(dim=1))  # (B*H*W, C)

        x_heads = x_flat.view(B * H * W, T, self.n_head, C // self.n_head)
        q_heads = q.view(B * H * W, self.n_head, C // self.n_head)

        k = self.key_net(x_heads)  # (B*H*W, T, n_head, d_k)
        q_k = self.key_net(q_heads)  # (B*H*W, n_head, d_k)

        att = torch.einsum("bnd,btnd->bnt", q_k, k) / math.sqrt(self.d_k)
        att = F.softmax(att, dim=-1)  # (B*H*W, n_head, T)
        att = self.dropout(att)

        out: torch.Tensor = torch.einsum(
            "bnt,btnc->bnc", att, x_heads
        )  # (B*H*W, n_head, C//n_head)
        out = out.view(B * H * W, C)
        out = self.outlayernorm(self.mlp(out))
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C_out, H, W)

        if self.return_att or return_att:
            att_out = att.view(B, H, W, self.n_head, T).permute(0, 3, 4, 1, 2)
            return out, att_out
        return out


class ConvLayer(nn.Module):
    """Conv -> GroupNorm/BatchNorm -> ReLU block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        p: int = 1,
        norm: str = "group",
        n_groups: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=norm is None)
        ]
        if norm == "group":
            layers.append(nn.GroupNorm(n_groups, out_ch))
        elif norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        return out


class DownConv(nn.Module):
    """Stride-2 Conv for downsampling."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1, norm: str = "group"
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(4, out_ch) if norm == "group" else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        return out


class UpConv(nn.Module):
    """Stride-2 ConvTranspose for upsampling."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1, norm: str = "group"
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(4, out_ch) if norm == "group" else nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        return out


class UTAE(nn.Module):
    """U-Net with L-TAE at the bottleneck and re-weighted skip connections.

    Input ``(B, T, C_in, H, W)`` (multi-temporal S2 series) + ``batch_positions``
    ``(B, T)``. Output ``(B, num_classes, H, W)``.
    """

    def __init__(
        self,
        input_dim: int = 10,
        encoder_widths: tuple[int, ...] = (32, 32, 64, 128),
        decoder_widths: tuple[int, ...] = (32, 32, 64, 128),
        out_conv: tuple[int, ...] = (32, 20),
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder_norm: str = "group",
        agg_mode: str = "att_group",
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_widths = list(encoder_widths)
        self.decoder_widths = list(decoder_widths)
        self.pad_value = pad_value
        self.agg_mode = agg_mode
        n_levels = len(encoder_widths)

        self.in_conv = ConvLayer(input_dim, encoder_widths[0], norm=encoder_norm)
        self.down_convs = nn.ModuleList(
            [
                DownConv(encoder_widths[i], encoder_widths[i + 1], norm=encoder_norm)
                for i in range(n_levels - 1)
            ]
        )

        ltae_in = encoder_widths[-1]
        self.temporal_encoder = LTAE2d(
            in_channels=ltae_in,
            n_head=n_head,
            d_k=d_k,
            mlp_in=(ltae_in, ltae_in),
            d_model=d_model,
            return_att=True,
        )

        self.skip_agg = nn.ModuleList(
            [
                ConvLayer(encoder_widths[i], encoder_widths[i], norm=encoder_norm)
                for i in range(n_levels - 1)
            ]
        )

        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for i in range(n_levels - 1, 0, -1):
            in_ch = decoder_widths[i]
            skip_ch = encoder_widths[i - 1]
            out_ch = decoder_widths[i - 1]
            self.up_convs.append(UpConv(in_ch, out_ch, norm=encoder_norm))
            self.dec_convs.append(ConvLayer(out_ch + skip_ch, out_ch, norm=encoder_norm))

        head_layers: list[nn.Module] = []
        in_ch = decoder_widths[0]
        for out_ch in out_conv:
            head_layers.append(nn.Conv2d(in_ch, out_ch, 1))
            if out_ch != out_conv[-1]:
                head_layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.out_conv = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor, batch_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Segment a temporal series.

        Args:
            x: Series ``(B, T, C_in, H, W)``.
            batch_positions: Temporal positions ``(B, T)`` (day-of-year).

        Returns:
            Dense logits ``(B, num_classes, H, W)``.
        """
        B, T, C, H, W = x.shape

        skips: list[torch.Tensor] = []
        xt = x.view(B * T, C, H, W)
        xt = self.in_conv(xt)
        _, w0, h0, w0s = xt.shape
        skips.append(xt.view(B, T, w0, h0, w0s))

        for down in self.down_convs:
            xt = down(xt)
            _, wi, hi, wis = xt.shape
            skips.append(xt.view(B, T, wi, hi, wis))

        bottleneck = skips[-1]
        feat, att = self.temporal_encoder(bottleneck, batch_positions)

        agg_skips: list[torch.Tensor] = []
        for i, skip in enumerate(skips[:-1]):
            _, _, _Wi, Hi, Wis = skip.shape
            att_up = F.interpolate(
                att.reshape(B * att.shape[1], T, att.shape[3], att.shape[4]),
                size=(Hi, Wis),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, att.shape[1], T, Hi, Wis)
            att_mean = att_up.mean(dim=1)  # (B, T, Hi, Wis)
            att_mean = F.softmax(att_mean, dim=1).unsqueeze(2)  # (B, T, 1, Hi, Wis)
            agg = (skip * att_mean).sum(dim=1)  # (B, Wi, Hi, Wis)
            agg = self.skip_agg[i](agg)
            agg_skips.append(agg)

        d = feat
        for up, dec, skip in zip(self.up_convs, self.dec_convs, reversed(agg_skips), strict=False):
            d = up(d)
            d = torch.cat([d, skip], dim=1)
            d = dec(d)

        logits: torch.Tensor = self.out_conv(d)
        return logits


def build_utae(num_classes: int = 20, input_dim: int = 10) -> UTAE:
    """Build the U-TAE with the config of Isaac's checkpoint (04j).

    Args:
        num_classes: Number of output classes. Isaac's checkpoint was trained
            with 20 (18 PASTIS crops + background + void), hence the default; for
            18 classes the head must be retrained.
        input_dim: Input bands (10 S2 bands).

    Returns:
        :class:`UTAE` model ready to load ``model_state_dict``.
    """
    return UTAE(
        input_dim=input_dim,
        encoder_widths=(32, 32, 64, 128),
        decoder_widths=(32, 32, 64, 128),
        out_conv=(32, num_classes),
        n_head=16,
        d_model=256,
        d_k=4,
        encoder_norm="group",
        agg_mode="att_group",
    )
