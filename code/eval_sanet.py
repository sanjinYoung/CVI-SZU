"""评估 SANet 系列模型 — 只取分割分支"""
import sys, os, re, json, time
import numpy as np
import torch, torch.nn.functional as F
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
WEIGHT_DIR = os.path.join(DATA_DIR, 'weight')


class CropAroundLabeld(transforms.MapTransform):
    def __init__(self, keys, roi_size=(128, 128, 128)):
        super().__init__(keys); self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)
        lbl = d[self.keys[1]]
        sz = self.roi_size[0]; half = sz // 2
        gt = (lbl[0] > 0.5).numpy().astype(np.uint8)
        coords = np.argwhere(gt).astype(np.float64)
        if len(coords) == 0:
            D, H, W = lbl.shape[1:]
            z1, y1, x1 = (D - sz) // 2, (H - sz) // 2, (W - sz) // 2
        else:
            cz, cy, cx = coords.mean(axis=0).astype(int)
            D, H, W = lbl.shape[1:]
            z1 = max(0, min(cz - half, D - sz))
            y1 = max(0, min(cy - half, H - sz))
            x1 = max(0, min(cx - half, W - sz))
        for key in self.keys:
            d[key] = d[key][:, z1:z1 + sz, y1:y1 + sz, x1:x1 + sz]
        return d


val_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
    transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                        mode=("bilinear", "nearest")),
    transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=500.0,
                                    b_min=0.0, b_max=1.0, clip=True),
    CropAroundLabeld(keys=["image", "label"], roi_size=ROI_SIZE),
    transforms.ToTensord(keys=["image", "label"]),
])


def build_loader():
    test_mask_dir = os.path.join(DATA_DIR, "test", "mask")
    test_label_dir = os.path.join(DATA_DIR, "test", "label")
    samples = []
    for fname in sorted(os.listdir(test_mask_dir)):
        lp = os.path.join(test_label_dir, fname)
        if os.path.exists(lp):
            samples.append({"image": os.path.join(test_mask_dir, fname), "label": lp})
    ds = data.PersistentDataset(data=samples, transform=val_transform,
                                 pickle_protocol=5, cache_dir=CACHE_DIR)
    return data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


def parse_dilation(filename):
    m = re.search(r'SANet\((\d{3})\)', filename)
    if m:
        return tuple(int(c) for c in m.group(1))
    return (2, 3, 4)


def eval_sanet(ckpt_path, dilation_rates):
    name = f'SANet{dilation_rates[0]}{dilation_rates[1]}{dilation_rates[2]}'
    print(f"\n{'='*60}")
    print(f"  {name}  dilation_rates={dilation_rates}")
    print(f"{'='*60}")

    from dilated_unet_reg_yuanbao_noreg_1212_reg import EnhancedBasicUNet
    model = EnhancedBasicUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        features=(32, 32, 64, 128, 256, 32),
        use_enhanced_head=True, fusion_type="graph",
        dilation_rates=dilation_rates,
    ).cuda()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    # 包一层：只返回分割输出，忽略回归分支
    class SegOnly(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
        def forward(self, x):
            out = self.base(x)
            if isinstance(out, tuple):
                return out[0]
            return out
    seg_model = SegOnly(model)

    loader = build_loader()
    print(f"  {len(loader)} samples")

    inferer = partial(sliding_window_inference, roi_size=ROI_SIZE,
                      sw_batch_size=4, predictor=seg_model, overlap=0.25, mode='gaussian')
    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    hd_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean",
                                         percentile=95, reduction=MetricReduction.MEAN, get_not_nans=True)
    post_label = AsDiscrete(to_onehot=OUT_CHANNELS)
    post_pred = AsDiscrete(argmax=True, to_onehot=OUT_CHANNELS)

    clinical_all = {'entry_err': [], 'apex_err': [], 'depth_err': [], 'angle_err': [], 'center_err': []}
    all_dice, all_iou, all_hd95 = 0.0, 0.0, 0.0
    n = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            img, lbl = batch_data["image"].cuda(), batch_data["label"].cuda()
            logits = inferer(img)

            vl = data.decollate_batch(lbl)
            vo = data.decollate_batch(logits)

            dice_metric.reset()
            dice_metric(y_pred=[post_pred(t) for t in vo], y=[post_label(t) for t in vl])
            ds, _ = dice_metric.aggregate()

            po = F.one_hot(logits.argmax(1), OUT_CHANNELS).permute(0, 4, 1, 2, 3).float()
            go = F.one_hot(lbl.squeeze(1).long(), OUT_CHANNELS).permute(0, 4, 1, 2, 3).float()
            inter = (po * go).sum(dim=(2, 3, 4))
            union = (po + go).clamp(0, 1).sum(dim=(2, 3, 4))
            iou = (inter / (union + 1e-7)).mean(dim=0)[1:].mean()

            hd_metric.reset()
            hd_metric(y_pred=[post_pred(t) for t in vo], y=[post_label(t) for t in vl])
            hd, _ = hd_metric.aggregate()

            pm = logits.argmax(1).cpu().numpy().astype(np.uint8)[0]
            gm = lbl.squeeze(1).cpu().numpy().astype(np.uint8)[0]
            if gm.sum() > 0:
                c = compute_clinical_metrics_from_masks(pm, gm)
                for k in clinical_all:
                    clinical_all[k].append(c[k])

            all_dice += ds.mean().item()
            all_iou += iou.item()
            all_hd95 += hd.mean().item()
            n += 1

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(loader)}] Dice={all_dice/n:.4f} ({time.time()-t0:.0f}s)")

    def stat(arr):
        return float(np.nanmean([v for v in arr if not np.isnan(v)]))

    result = {
        'model': name,
        'dilation_rates': list(dilation_rates),
        'dice': all_dice / n,
        'iou': all_iou / n,
        'hd95': all_hd95 / n,
        'entry_err': stat(clinical_all['entry_err']),
        'apex_err': stat(clinical_all['apex_err']),
        'depth_err': stat(clinical_all['depth_err']),
        'angle_err': stat(clinical_all['angle_err']),
        'center_err': stat(clinical_all['center_err']),
    }
    print(f"  Dice={result['dice']:.4f} IoU={result['iou']:.4f} "
          f"Entry={result['entry_err']:.2f}mm Apex={result['apex_err']:.2f}mm "
          f"Angle={result['angle_err']:.2f}° ({time.time()-t0:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    all_weights = sorted([f for f in os.listdir(WEIGHT_DIR) if f.startswith('SANet(')])
    print(f"Found {len(all_weights)} SANet checkpoints")

    results = []
    for w in all_weights:
        path = os.path.join(WEIGHT_DIR, w)
        dilation = parse_dilation(w)
        try:
            r = eval_sanet(path, dilation)
            results.append(r)
            with open(f'{DATA_DIR}/result_{r["model"]}.json', 'w') as f:
                json.dump(r, f, indent=2)
        except Exception as e:
            name = f'SANet_{w}'
            print(f"  FAIL: {e}")

    # 汇总
    print(f"\n{'='*80}")
    header = f"{'Model':<15} {'Dil':<8} {'Dice':>8} {'IoU':>8} {'Entry':>8} {'Apex':>8} {'Depth':>8} {'Angle':>8} {'Center':>8}"
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x['dice']):
        dil = ''.join(str(d) for d in r['dilation_rates'])
        print(f"{r['model']:<15} {dil:<8} {r['dice']:>8.4f} {r['iou']:>8.4f} {r['entry_err']:>8.2f} "
              f"{r['apex_err']:>8.2f} {r['depth_err']:>8.2f} {r['angle_err']:>8.2f} {r['center_err']:>8.2f}")
