from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        pad = kernel // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(max(1, min(32, out_ch // 8)), out_ch),
            nn.GELU(),
        )


class SimpleEncoder(nn.Module):
    """Dependency-light encoder used for smoke tests and CPU prototyping.

    Production training should normally use a pretrained timm backbone, but this
    encoder makes the architecture testable without downloading external weights.
    """

    def __init__(self, base: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, base, 3, 2),
            ConvNormAct(base, base, 3, 1),
        )
        self.s1 = nn.Sequential(ConvNormAct(base, base * 2, 3, 2), ConvNormAct(base * 2, base * 2))
        self.s2 = nn.Sequential(ConvNormAct(base * 2, base * 4, 3, 2), ConvNormAct(base * 4, base * 4))
        self.s3 = nn.Sequential(ConvNormAct(base * 4, base * 8, 3, 2), ConvNormAct(base * 8, base * 8))
        self.s4 = nn.Sequential(ConvNormAct(base * 8, base * 12, 3, 2), ConvNormAct(base * 12, base * 12))
        self.channels = [base * 2, base * 4, base * 8, base * 12]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)          # 1/2
        f0 = self.s1(x)           # 1/4
        f1 = self.s2(f0)          # 1/8
        f2 = self.s3(f1)          # 1/16
        f3 = self.s4(f2)          # 1/32
        return [f0, f1, f2, f3]


class TimmEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, out_indices: Iterable[int] = (0, 1, 2, 3)):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - actionable runtime error
            raise ImportError("CutoutNet pretrained backbones require `timm`. Run repair_windows.bat.") from exc
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.channels = list(self.model.feature_info.channels())
        if len(self.channels) != 4:
            raise ValueError(f"CutoutNet expects four feature levels, got {self.channels} from {name}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.model(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.skip = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        self.in_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.fuse = nn.Sequential(
            ConvNormAct(out_ch * 2, out_ch),
            ConvNormAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.in_proj(x)
        s = self.skip(skip)
        return self.fuse(torch.cat([x, s], dim=1))


class PredictionHead(nn.Sequential):
    def __init__(self, channels: int):
        hidden = max(32, channels // 2)
        super().__init__(
            ConvNormAct(channels, hidden),
            nn.Conv2d(hidden, 1, 1),
        )


@dataclass
class CutoutNetConfig:
    backbone: str = "convnext_tiny"
    pretrained_backbone: bool = True
    decoder_channels: int = 128
    simple_encoder_base: int = 48


class CutoutNet(nn.Module):
    """Multi-task foreground extraction network.

    Heads are intentionally explicit rather than hidden inside one opaque mask:
      * foreground: coarse semantic foreground
      * boundary: true object contour
      * detail: thin / fragile structures
      * residual_background: background mistakenly included in foreground
      * uncertainty: regions requiring extra attention / refinement
      * objectness: support for multiple independent significant objects

    A learned fusion head combines all six signals into the final mask logit.
    """

    HEADS = (
        "foreground",
        "boundary",
        "detail",
        "residual_background",
        "uncertainty",
        "objectness",
    )

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained_backbone: bool = True,
        decoder_channels: int = 128,
        simple_encoder_base: int = 48,
    ):
        super().__init__()
        if backbone in {"simple", "internal", "tiny_internal"}:
            self.encoder = SimpleEncoder(simple_encoder_base)
        else:
            self.encoder = TimmEncoder(backbone, pretrained_backbone)

        c0, c1, c2, c3 = self.encoder.channels
        d = int(decoder_channels)
        self.top = nn.Sequential(nn.Conv2d(c3, d, 1, bias=False), ConvNormAct(d, d))
        self.dec2 = DecoderBlock(d, c2, d)
        self.dec1 = DecoderBlock(d, c1, d)
        self.dec0 = DecoderBlock(d, c0, d)
        self.refine = nn.Sequential(ConvNormAct(d, d), ConvNormAct(d, d))

        self.heads = nn.ModuleDict({name: PredictionHead(d) for name in self.HEADS})
        self.fusion = nn.Sequential(
            ConvNormAct(d + len(self.HEADS), d),
            ConvNormAct(d, d // 2),
            nn.Conv2d(d // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        f0, f1, f2, f3 = self.encoder(x)
        y = self.top(f3)
        y = self.dec2(y, f2)
        y = self.dec1(y, f1)
        y = self.dec0(y, f0)
        y = self.refine(y)

        native = {name: head(y) for name, head in self.heads.items()}
        probs = [torch.sigmoid(native[name]) for name in self.HEADS]
        final_native = self.fusion(torch.cat([y, *probs], dim=1))

        out: dict[str, torch.Tensor] = {
            name: F.interpolate(logit, size=input_size, mode="bilinear", align_corners=False)
            for name, logit in native.items()
        }
        out["final"] = F.interpolate(final_native, size=input_size, mode="bilinear", align_corners=False)
        return out

    def freeze_encoder(self, frozen: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not frozen


class ROIRefiner(nn.Module):
    """Optional high-resolution patch refiner.

    Input channels: RGB (3) + coarse probability (1) + uncertainty (1).
    The model predicts a correction logit for the local patch.  It is kept
    separate so the base CutoutNet can be trained and benchmarked first.
    """

    def __init__(self, base: int = 48):
        super().__init__()
        self.enc1 = nn.Sequential(ConvNormAct(5, base), ConvNormAct(base, base))
        self.enc2 = nn.Sequential(ConvNormAct(base, base * 2, stride=2), ConvNormAct(base * 2, base * 2))
        self.enc3 = nn.Sequential(ConvNormAct(base * 2, base * 4, stride=2), ConvNormAct(base * 4, base * 4))
        self.mid = nn.Sequential(ConvNormAct(base * 4, base * 4), ConvNormAct(base * 4, base * 4))
        self.up2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.up1 = DecoderBlock(base * 2, base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, rgb: torch.Tensor, coarse: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, coarse, uncertainty], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        y = self.mid(e3)
        y = self.up2(y, e2)
        y = self.up1(y, e1)
        return self.out(y)
