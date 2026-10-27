import numpy as np
import cv2
import torch
from PIL import Image

from cutoutnet.instances import InstanceNet, make_instance_targets, instance_loss, decode_instances, infer_instance_labels, component_labels
from cutoutnet.instance_train import InstanceDataset
from cutoutnet.colors import ColorCleanNet, apply_color_net
from cutoutnet.color_train import ColorDataset


def test_instance_forward_loss_and_gradient():
    m = InstanceNet(base=8)
    rgb = torch.randn(1, 3, 32, 48)
    mask = torch.ones(1, 1, 32, 48)
    outputs = m(rgb, mask)
    assert outputs['center'].shape == (1, 1, 32, 48)
    assert outputs['offset'].shape == (1, 2, 32, 48)
    ids = np.zeros((32,48), np.uint16)
    ids[3:23,3:15] = 1
    ids[12:30,20:45] = 2
    c, offsets, fg = make_instance_targets(ids)
    loss = instance_loss(outputs, torch.from_numpy(c)[None], torch.from_numpy(offsets)[None], torch.from_numpy(fg)[None])
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_split_touching_with_perfect_centers_offsets_and_preserve_hole():
    ids = np.zeros((72,90), np.int32)
    ids[10:55,10:46] = 1
    ids[15:61,46:82] = 2 # touching along vertical border
    ids[20:26,20:27] = 0
    center, offsets, fg = make_instance_targets(ids)
    result = decode_instances(fg[0], center[0], offsets, min_peak_distance=4)
    assert np.array_equal(result > 0, ids > 0)
    assert len(set(result[result > 0])) == 2
    assert result[22,23] == 0


def test_connected_components_does_not_delete_tiny_part():
    mask = np.zeros((45,45), np.uint8)
    mask[4,4] = 255
    mask[20:40,20:40] = 255
    labels = component_labels(mask)
    assert len(np.unique(labels)) == 3


def test_instance_inference_no_lost_original_pixels():
    class Blank(torch.nn.Module):
        def forward(self, rgb, fg):
            return {'center': torch.full_like(fg, -10), 'offset': torch.zeros((rgb.shape[0],2,*fg.shape[-2:]), device=rgb.device)}
    image = np.ones((99,135,3), np.uint8)*127
    prob = np.zeros((99,135), np.float32)
    prob[3,3] = 1
    prob[30:70,20:60] = 1
    prob[46:55,36:41] = 0
    labels = infer_instance_labels(Blank(), image, prob, device=torch.device('cpu'), side=64)
    assert np.array_equal(labels > 0, prob >= .5)
    assert labels[3,3] > 0
    assert labels[50,38] == 0


def test_color_unchanged_without_border_and_changes_only_edge():
    model = ColorCleanNet(base=8)
    rgb = np.full((55,67,3), 100, np.uint8)
    alpha = np.zeros((55,67), np.uint8)
    alpha[8:46,7:57] = 255
    original = rgb.copy()
    out = apply_color_net(rgb, alpha, model, torch.device('cpu'), patch=48, margin=12)
    assert np.array_equal(rgb, original)
    assert out.shape == rgb.shape
    assert np.array_equal(out[alpha == 0], rgb[alpha == 0])
    assert np.array_equal(out[25:30,25:30], rgb[25:30,25:30])


def test_datasets_support_integer_instance_ids_and_color_png(tmp_path):
    folder = tmp_path/'instances'
    (folder/'images').mkdir(parents=True)
    (folder/'instances').mkdir()
    rgb = np.zeros((30,40,3),np.uint8)
    ids = np.zeros((30,40),np.uint16)
    ids[5:15,4:20] = 257
    Image.fromarray(rgb).save(folder/'images'/'a.png')
    Image.fromarray(ids).save(folder/'instances'/'a.png')
    item = InstanceDataset(folder, size=64)[0]
    assert item['fg'].shape == (1,64,64)
    assert item['fg'].sum() > 0
    rgba_dir=tmp_path/'rgba'
    rgba_dir.mkdir()
    rgba=np.zeros((32,30,4),np.uint8)
    rgba[4:30,3:27,:3] = 205
    rgba[4:30,3:27,3] = 255
    Image.fromarray(rgba,'RGBA').save(rgba_dir/'a.png')
    ci=ColorDataset(str(rgba_dir),crop=32)[0]
    assert ci['dirty'].shape == (3,32,32)


def test_topology_loss_and_sticker_loss_backward():
    from cutoutnet.advanced_losses import soft_cldice, StickerLoss
    gt = torch.zeros((1,1,32,48),dtype=torch.float32)
    gt[:,:,12,4:44] = 1
    logits = torch.randn((1,1,32,48),requires_grad=True)
    penalty = soft_cldice(logits,gt,iterations=4)
    assert torch.isfinite(penalty)
    pred = {'final':logits}
    target = {'final':gt}
    overall, parts = StickerLoss(edge_weight=.5,topology_weight=.2)(pred,target)
    assert set(parts) == {'final','soft_cldice','weighted_edge'}
    overall.backward()
    assert torch.isfinite(logits.grad).all()


def test_calibrator_prefers_correct_threshold():
    from calibrate_v04 import best_from_hist
    fg = np.zeros(256, np.int64)
    bg = np.zeros(256, np.int64)
    fg[200:] = 10
    bg[:90] = 10
    cal = best_from_hist(fg,bg)
    assert .35 < cal['threshold'] < .8
    assert cal['dice'] > .99999
