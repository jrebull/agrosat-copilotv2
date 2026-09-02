"""Native temporal architectures: TempCNN + InceptionTime (US-022b-C).

Own implementation (PyTorch) of the two official architectures of the
BreizhCrops benchmark (Russwurm et al. 2020), ported directly from the
original papers and from the reference implementation (MIT license) in
``breizhcrops`` 0.0.4.1. Previously this module imported the classes from
``breizhcrops.models``; the reframing (updated ADR-006 D-ARQ-2) brings them
into the repo to gain control over MLflow serialization, custom layers,
isolated testing and independent evolution from the external dependency.

Models
------

- :class:`TempCNN` (Pelletier, Webb & Petitjean 2019). 1D CNN with three
  convolutional blocks + a dense head. Designed for crop classification
  over Sentinel-2 series.
  DOI: 10.3390/rs11050523. Reference code: MIT.
- :class:`InceptionTime` (Fawaz et al. 2020). Stack of Inception blocks
  with residual shortcut; winning architecture of the UCR benchmark.
  DOI: 10.1007/s10618-020-00710-y. Reference code: MIT.

Both accept input ``(B, T, C)`` (batch, time, channels = spectral
indices) and produce logits ``(B, num_classes)``.

Technical decisions
-------------------

- He uniformly initialized weights (kaiming_uniform_) per PyTorch
  convention.
- BatchNorm1d between conv and activation (Conv-BN-ReLU order).
- Global Average Pooling before the dense head (InceptionTime); for
  TempCNN, flatten after the last conv block per the original
  implementation.
- Configurable dropout; defaults inherited from the papers (0.5 for
  TempCNN, 0.2 for InceptionTime).
- No ``breizhcrops`` dependency at runtime; only numpy + torch.

Attribution
-----------

Adapted from:
- breizhcrops 0.0.4.1, ``breizhcrops/models/TempCNN.py``,
  ``breizhcrops/models/InceptionTime.py`` (MIT License).
- Pelletier, Webb & Petitjean. ``Temporal Convolutional Neural Network for
  the Classification of Satellite Image Time Series``. Remote Sensing
  11(5):523, 2019.
- Fawaz et al. ``InceptionTime: Finding AlexNet for Time Series
  Classification``. Data Mining and Knowledge Discovery 34, 2020.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["InceptionTime", "TempCNN", "build_temporal_model"]


# ---------------------------------------------------------------------------
# TempCNN  (Pelletier, Webb & Petitjean 2019)
# ---------------------------------------------------------------------------


class _TempCNNBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU + Dropout block (kernel_size=5 default)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.dropout(self.act(self.bn(self.conv(x))))
        return out


class TempCNN(nn.Module):
    """TempCNN: 3 Conv1D blocks + dense head for series classification.

    Args:
        input_dim: Number of channels C (spectral indices).
        num_classes: Number of classes in the final head.
        sequencelength: Temporal length T of each series.
        hidden_dim: Filters per convolutional block (default 64).
        kernel_size: Temporal kernel size (default 5).
        dropout: Dropout after each conv block and before the dense
            (default 0.5, as in the paper).

    Input:
        Tensor ``(B, T, C)``.

    Output:
        Logits ``(B, num_classes)``.

    Reference:
        Pelletier, Webb & Petitjean (2019), Remote Sensing 11(5):523.
        DOI 10.3390/rs11050523.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequencelength: int,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.sequencelength = sequencelength
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dropout_p = dropout

        self.block1 = _TempCNNBlock(input_dim, hidden_dim, kernel_size, dropout)
        self.block2 = _TempCNNBlock(hidden_dim, hidden_dim, kernel_size, dropout)
        self.block3 = _TempCNNBlock(hidden_dim, hidden_dim, kernel_size, dropout)

        flatten_size = hidden_dim * sequencelength
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flatten_size, hidden_dim * 4)
        self.dense_bn = nn.BatchNorm1d(hidden_dim * 4)
        self.dense_act = nn.ReLU(inplace=True)
        self.dense_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input (B, T, C) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dense_dropout(self.dense_act(self.dense_bn(self.dense(x))))
        logits: torch.Tensor = self.classifier(x)
        return logits


# ---------------------------------------------------------------------------
# InceptionTime  (Fawaz et al. 2020)
# ---------------------------------------------------------------------------


class _InceptionModule(nn.Module):
    """1D Inception module: bottleneck + 3 parallel convolutions + maxpool.

    Args:
        in_channels: Input channels.
        nb_filters: Filters per branch (4 branches total, output = 4*nb_filters).
        kernel_sizes: Kernel sizes for the 3 parallel convolutions.
        bottleneck_channels: Channels of the initial bottleneck (0 = no
            bottleneck, recommended if in_channels > 1).
        use_bias: Whether the convolutions carry bias.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        bottleneck_channels: int = 32,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if bottleneck_channels > 0 and in_channels > 1:
            self.bottleneck: nn.Conv1d | None = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=use_bias
            )
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    nb_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                    bias=use_bias,
                )
                for ks in kernel_sizes
            ]
        )
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=use_bias),
        )

        total_out = nb_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck is not None:
            bottlenecked = self.bottleneck(x)
        else:
            bottlenecked = x
        branches = [conv(bottlenecked) for conv in self.conv_branches]
        # Adjust to the minimum temporal length (large kernels can
        # produce T+1 depending on parity).
        target_t = min(b.size(-1) for b in branches)
        branches = [b[..., :target_t] for b in branches]

        maxpool_out = self.maxpool_branch(x)[..., :target_t]
        merged = torch.cat([*branches, maxpool_out], dim=1)
        out: torch.Tensor = self.act(self.bn(merged))
        return out


class _ShortcutBlock(nn.Module):
    """Residual shortcut (Conv1d 1x1 + BatchNorm + ReLU)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, residual: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if residual.size(-1) != out.size(-1):
            residual = residual[..., : out.size(-1)]
        shortcut = self.bn(self.conv(residual))
        fused: torch.Tensor = self.act(shortcut + out)
        return fused


class InceptionTime(nn.Module):
    """InceptionTime: 6 Inception modules with residual shortcut every 3.

    Args:
        input_dim: Number of channels C (spectral indices).
        num_classes: Number of classes in the final head.
        nb_filters: Filters per branch of the Inception module.
        depth: Number of stacked Inception modules (default 6).
        kernel_sizes: Parallel kernel sizes in each module.
        bottleneck_channels: Bottleneck channels (default 32).
        dropout: Dropout before the final classifier.

    Input:
        Tensor ``(B, T, C)``.

    Output:
        Logits ``(B, num_classes)``.

    Reference:
        Fawaz, Lucas, Forestier et al. (2020), Data Mining and Knowledge
        Discovery 34. DOI 10.1007/s10618-020-00710-y.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        nb_filters: int = 32,
        depth: int = 6,
        kernel_sizes: tuple[int, int, int] = (39, 19, 9),
        bottleneck_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.nb_filters = nb_filters
        self.depth = depth

        out_channels_per_module = nb_filters * (len(kernel_sizes) + 1)

        modules: list[nn.Module] = []
        shortcuts: list[nn.Module | None] = []
        current_in = input_dim
        for d in range(depth):
            module = _InceptionModule(
                in_channels=current_in,
                nb_filters=nb_filters,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bias=False,
            )
            modules.append(module)
            current_in = out_channels_per_module
            # Shortcut every 3 blocks (per paper).
            shortcuts.append(None if (d + 1) % 3 != 0 else _ShortcutBlock(0, 0))

        self.inception_modules = nn.ModuleList(modules)
        # We build shortcuts knowing the exact residual input in forward.
        # Here we only store placeholders; the 1x1 Conv1d are created lazily.
        self._build_shortcuts(input_dim, out_channels_per_module)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_channels_per_module, num_classes)

        self._init_weights()

    def _build_shortcuts(self, input_dim: int, out_channels_per_module: int) -> None:
        """Create the 1x1 shortcuts knowing the real channels at each jump.

        Shortcuts every 3 blocks: the first compares input vs the output of
        block 2 (index 2 with +1=3); the second compares block 2 output
        (out_ch_per) vs block 5 output (out_ch_per).
        """
        sc_modules: list[nn.Module] = []
        for d in range(self.depth):
            if (d + 1) % 3 != 0:
                continue
            if d == 2:
                # Shortcut input -> block 2 output
                sc_modules.append(_ShortcutBlock(input_dim, out_channels_per_module))
            else:
                # Subsequent shortcuts: previous output -> current output
                sc_modules.append(_ShortcutBlock(out_channels_per_module, out_channels_per_module))
        self.shortcuts = nn.ModuleList(sc_modules)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                if module.weight.numel() == 0:
                    continue
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input (B, T, C) -> Conv1d expects (B, C, T).
        x = x.transpose(1, 2)
        residual = x
        shortcut_idx = 0
        out = x
        for d, module in enumerate(self.inception_modules):
            out = module(out)
            if (d + 1) % 3 == 0:
                out = self.shortcuts[shortcut_idx](residual, out)
                residual = out
                shortcut_idx += 1
        pooled = self.global_pool(out).squeeze(-1)  # (B, out_channels)
        pooled = self.dropout(pooled)
        logits: torch.Tensor = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------------
# Factory helper for selecting by name.
# ---------------------------------------------------------------------------


def build_temporal_model(
    model_kind: str,
    *,
    input_dim: int,
    num_classes: int,
    sequence_length: int,
    **overrides: object,
) -> nn.Module:
    """Build a temporal model by name.

    Args:
        model_kind: ``"tempcnn"`` or ``"inceptiontime"``.
        input_dim: Number of channels C.
        num_classes: Number of effective classes.
        sequence_length: Temporal length T.
        **overrides: Additional hyperparameters passed to the model
            constructor (``hidden_dim``, ``dropout``, ``depth``, etc.).

    Returns:
        ``nn.Module`` ready to train.

    Raises:
        ValueError: if ``model_kind`` is not one of the supported ones.
    """
    if model_kind == "tempcnn":
        return TempCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            sequencelength=sequence_length,
            **overrides,  # type: ignore[arg-type]
        )
    if model_kind == "inceptiontime":
        return InceptionTime(
            input_dim=input_dim,
            num_classes=num_classes,
            **overrides,  # type: ignore[arg-type]
        )
    raise ValueError(f"model_kind={model_kind!r} not supported. Use 'tempcnn' or 'inceptiontime'.")
