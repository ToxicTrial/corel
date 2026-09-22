import numpy as np
import torch
import cv2

from cutoutnet.model import CutoutNet
from cutoutnet.targets import build_targets
from cutoutnet.losses import CutoutNetLoss


def test_cutoutnet_forward_all_heads():
    model = CutoutNet(backbone="simple", pretrained_backbone=False, decoder_channels=32, simple_encoder_base=8)
    x = torch.randn(1, 3, 128, 160)
    out = model(x)
    expected = {
        "foreground", "boundary", "detail", "residual_background",
        "uncertainty", "objectness", "final",
    }
    assert set(out) == expected
    for value in out.values():
        assert tuple(value.shape) == (1, 1, 128, 160)


def test_targets_preserve_multiple_objects_and_internal_hole():
    mask = np.zeros((128, 128), np.uint8)
    cv2.rectangle(mask, (10, 10), (45, 55), 255, -1)
    cv2.rectangle(mask, (70, 20), (115, 100), 255, -1)
    cv2.circle(mask, (92, 60), 10, 0, -1)
    t = build_targets(mask)
    assert t["final"].sum() > 0
    assert t["objectness"][25, 25] > 0
    assert t["objectness"][50, 90] > 0
    assert t["residual_background"][60, 92] > 0
    assert t["boundary"].sum() > 0
    assert t["detail"].sum() > 0


def test_cutoutnet_multitask_loss_backward():
    model = CutoutNet(backbone="simple", pretrained_backbone=False, decoder_channels=24, simple_encoder_base=8)
    rgb = torch.randn(1, 3, 96, 96)
    mask = np.zeros((96, 96), np.uint8)
    cv2.circle(mask, (48, 48), 25, 255, -1)
    targets_np = build_targets(mask)
    targets = {k: torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float() for k, v in targets_np.items()}
    out = model(rgb)
    loss, parts = CutoutNetLoss()(out, targets)
    assert torch.isfinite(loss)
    assert set(parts) == set(targets)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_cutoutnet_checkpoint_roundtrip_and_binary_mask(tmp_path):
    model_cfg = {
        "backbone": "simple",
        "pretrained_backbone": False,
        "decoder_channels": 16,
        "simple_encoder_base": 8,
    }
    model = CutoutNet(**model_cfg)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "model_config": model_cfg, "image_size": 64}, ckpt)

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    loaded = CutoutNet(**saved["model_config"])
    loaded.load_state_dict(saved["model"])
    loaded.eval()
    with torch.inference_mode():
        logits = loaded(torch.randn(1, 3, 64, 64))["final"]
    alpha = (torch.sigmoid(logits) >= 0.5).to(torch.uint8) * 255
    assert set(torch.unique(alpha).tolist()).issubset({0, 255})
