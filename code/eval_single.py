"""单模型评估: python eval_single.py <model_name>"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from monai import transforms, data
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

sys.path.insert(0, '/data/xinyuan/CBCT/LLM_ssl/Large-Scale-Medical-main/Downstream/VoCOVID')
sys.path.insert(0, '/data1/xinyuan/CBCT/ImplantFairy')
from clinical_metrics import compute_clinical_metrics_from_masks

ROI_SIZE = (128, 128, 128)
OUT_CHANNELS = 2
DATA_DIR = '/data1/xinyuan/CBCT/ImplantFairy'
CACHE_DIR = '/data1/xinyuan/CBCT/ImplantFairy/cache'

from monai.transforms import MapTransform
import numpy as np

class CropAroundLabeld(MapTransform):
    """根据 label 中心裁剪 128³ 区域"""
    def __init__(self, keys, roi_size=(128, 128, 128)):
        super().__init__(keys)
        self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)
        lbl = d[self.keys[1]]  # label tensor (C, D, H, W)
        sz = self.roi_size[0]
        half = sz // 2

        # 找 GT mask 中心
        gt = (lbl[0] > 0.5).numpy().astype(np.uint8)
        coords = np.argwhere(gt).astype(np.float64)
        if len(coords) == 0:
            # fallback: 中心裁剪
            D, H, W = lbl.shape[1:]
            z1, y1, x1 = (D - sz) // 2, (H - sz) // 2, (W - sz) // 2
        else:
            cz, cy, cx = coords.mean(axis=0).astype(int)
            D, H, W = lbl.shape[1:]
            z1 = max(0, min(cz - half, D - sz))
            y1 = max(0, min(cy - half, H - sz))
            x1 = max(0, min(cx - half, W - sz))

        for key in self.keys:
            d[key] = d[key][:, z1:z1+sz, y1:y1+sz, x1:x1+sz]
        return d


val_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
    transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    CropAroundLabeld(keys=["image", "label"], roi_size=ROI_SIZE),
    transforms.ToTensord(keys=["image", "label"]),
])

MODEL_CONFIGS = {
    'DAF3D': {
        'ckpt': 'DAF3D_49_0.pt',
        'build': lambda: __import__('daf3d').DAF3D(in_channels=1, out_channels=OUT_CHANNELS),
    },
    'SwinUNETRv2': {
        'ckpt': 'SwinUNETR_49_0.pt',
        'build': lambda: __import__('monai.networks.nets', fromlist=['SwinUNETR']).SwinUNETR(
            img_size=ROI_SIZE, in_channels=1, out_channels=OUT_CHANNELS, feature_size=48,
            drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0, use_checkpoint=True, use_v2=True),
    },
    'UNET': {
        'ckpt': 'UNET_49_0.pt',
        'build': lambda: __import__('monai.networks.nets', fromlist=['BasicUNet']).BasicUNet(
            in_channels=1, out_channels=OUT_CHANNELS),
    },
    'UNETR_PP': {
        'ckpt': 'UNETR_PP_49_0.pt',
        'build': lambda: __import__('unetr_pp1').UNETR_PP(
            img_size=ROI_SIZE, in_channels=1, out_channels=OUT_CHANNELS,
            hidden_size=768, num_heads=12, mlp_dim=3072, norm_name="instance", res_block=True),
    },
    'VNet': {
        'ckpt': 'VNet_49_0.pt',
        'build': lambda: __import__('monai.networks.nets', fromlist=['VNet']).VNet(
            in_channels=1, out_channels=OUT_CHANNELS),
    },
    'UNET++': {
        'ckpt': 'UNET_PP_49_0.pt',
        'build': lambda: __import__('unetplusplus').BasicUNetPlusPlus(
            in_channels=1, out_channels=OUT_CHANNELS),
    },
    'SegMambav1': {
        'ckpt': 'SegMambav1_49.pt',
        'build': lambda: __import__('segmamba').SegMamba(in_chans=1, out_chans=OUT_CHANNELS),
    },
    'SegMambav2': {
        'ckpt': 'SegMambav2_49.pt',
        'build': lambda: __import__('segmambav2').SegMamba(in_chans=1, out_chans=OUT_CHANNELS),
    },
    'UXNET': {
        'ckpt': 'UXNET_49_0.pt',
        'build': lambda: __import__('uxnet').UXNET(in_chans=1, out_chans=OUT_CHANNELS),
    },
    'UNETR': {
        'ckpt': 'UNETR_49_0.pt',
        'build': lambda: __import__('monai.networks.nets', fromlist=['UNETR']).UNETR(
            img_size=ROI_SIZE, in_channels=1, out_channels=OUT_CHANNELS,
            hidden_size=768, num_heads=12, mlp_dim=3072,
            pos_embed="conv", norm_name="instance", res_block=True),
    },
}


def build_loader():
    test_mask_dir = os.path.join(DATA_DIR, "test", "mask")
    test_label_dir = os.path.join(DATA_DIR, "test", "label")
    samples, label_paths = [], []
    for fname in sorted(os.listdir(test_mask_dir)):
        lp = os.path.join(test_label_dir, fname)
        if os.path.exists(lp):
            samples.append({"image": os.path.join(test_mask_dir, fname), "label": lp})
            label_paths.append(lp)
    ds = data.PersistentDataset(data=samples, transform=val_transform,
                                 pickle_protocol=5, cache_dir=CACHE_DIR)
    return data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True), label_paths


def eval_model(name):
    cfg = MODEL_CONFIGS[name]
    ckpt_path = os.path.join(DATA_DIR, 'weight', cfg['ckpt'])

    print(f"[{name}] Building model...")
    model = cfg['build']().cuda()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"[{name}] Loading data...")
    loader, label_paths = build_loader()
    print(f"[{name}] {len(loader)} samples, evaluating...")

    inferer = partial(sliding_window_inference, roi_size=ROI_SIZE,
                      sw_batch_size=4, predictor=model, overlap=0.25, mode='gaussian')
    post_label = AsDiscrete(to_onehot=OUT_CHANNELS)
    post_pred = AsDiscrete(argmax=True, to_onehot=OUT_CHANNELS)
    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    hd_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean",
                                         percentile=95, reduction=MetricReduction.MEAN, get_not_nans=True)

    clinical_all = {'entry_err': [], 'apex_err': [], 'depth_err': [], 'angle_err': [], 'center_err': []}
    all_dice, all_iou, all_hd95 = 0.0, 0.0, 0.0
    n = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            img, lbl = batch_data["image"].cuda(), batch_data["label"].cuda()
            logits = inferer(img)

            # Dice
            vl = data.decollate_batch(lbl)
            vo = data.decollate_batch(logits)
            dice_metric.reset()
            dice_metric(y_pred=[post_pred(t) for t in vo], y=[post_label(t) for t in vl])
            ds, _ = dice_metric.aggregate()

            # IoU
            po = F.one_hot(logits.argmax(1), num_classes=OUT_CHANNELS).permute(0, 4, 1, 2, 3).float()
            go = F.one_hot(lbl.squeeze(1).long(), num_classes=OUT_CHANNELS).permute(0, 4, 1, 2, 3).float()
            inter = (po * go).sum(dim=(2, 3, 4))
            union = (po + go).clamp(0, 1).sum(dim=(2, 3, 4))
            iou = (inter / (union + 1e-7)).mean(dim=0)[1:].mean()

            # HD95
            hd_metric.reset()
            hd_metric(y_pred=[post_pred(t) for t in vo], y=[post_label(t) for t in vl])
            hd, _ = hd_metric.aggregate()

            # Clinical
            pm = logits.argmax(1).cpu().numpy().astype(np.uint8)[0]
            gm = lbl.squeeze(1).cpu().numpy().astype(np.uint8)[0]
            if gm.sum() > 0:
                c = compute_clinical_metrics_from_masks(pm, gm, filepath=label_paths[idx])
                for k in clinical_all:
                    clinical_all[k].append(c[k])

            all_dice += ds.mean().item()
            all_iou += iou.item()
            all_hd95 += hd.mean().item()
            n += 1

            if (idx + 1) % 50 == 0:
                print(f"  [{name}] {idx+1}/{len(loader)} Dice={all_dice/n:.4f} ({time.time()-t0:.0f}s)")

    def stat(arr): return float(np.nanmean([v for v in arr if not np.isnan(v)]))

    result = {
        'model': name,
        'dice': all_dice / n,
        'iou': all_iou / n,
        'hd95': all_hd95 / n,
        'entry_err': stat(clinical_all['entry_err']),
        'apex_err': stat(clinical_all['apex_err']),
        'depth_err': stat(clinical_all['depth_err']),
        'angle_err': stat(clinical_all['angle_err']),
        'center_err': stat(clinical_all['center_err']),
    }

    print(f"[{name}] Dice={result['dice']:.4f} IoU={result['iou']:.4f} "
          f"Entry={result['entry_err']:.2f}mm Apex={result['apex_err']:.2f}mm "
          f"Depth={result['depth_err']:.2f}mm Angle={result['angle_err']:.2f}° "
          f"Center={result['center_err']:.2f}mm ({time.time()-t0:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    name = sys.argv[1]
    r = eval_model(name)
    import json
    out_path = f'{DATA_DIR}/result_{name}.json'
    with open(out_path, 'w') as f:
        json.dump(r, f, indent=2)
    print(f'Saved: {out_path}')
