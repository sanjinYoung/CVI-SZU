"""RegFreeNet (含 64 节点 message passing GCN) DDP 多卡训练

启动方式（4 卡，GPU 0,1,2,4）:
    CUDA_VISIBLE_DEVICES=0,1,2,4 torchrun --nproc_per_node=4 --master_port=28890 train_ddp_msg.py

训练数据: train/mask (masked CBCT) + train/label (植入体 mask)
验证数据: test/mask + test/label
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from monai import transforms, data
from monai.losses import DiceCELoss
from monai.transforms import MapTransform

sys.path.insert(0, '/data1/xinyuan/CBCT/ImplantFairy')
from dilated_unet_reg_msg import EnhancedBasicUNet

ROI_SIZE = (128, 128, 128)
OUT_CHANNELS = 2
DATA_DIR = '/data1/xinyuan/CBCT/ImplantFairy'
CACHE_DIR = '/data1/xinyuan/CBCT/ImplantFairy/cache'


def setup_ddp():
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


class CropAroundLabeld(MapTransform):
    """围绕植入体中心裁剪 128^3（训练时带随机偏移，验证时不偏移）"""

    def __init__(self, keys, roi_size=ROI_SIZE, rand_offset=8):
        super().__init__(keys)
        self.roi_size = roi_size
        self.rand_offset = rand_offset

    def __call__(self, d):
        d = dict(d)
        lbl = d[self.keys[1]]
        sz = self.roi_size[0]
        half = sz // 2
        gt = (lbl[0] > 0.5).numpy().astype(np.uint8)
        coords = np.argwhere(gt).astype(np.float64)
        D, H, W = lbl.shape[1:]
        if len(coords) == 0:
            z1, y1, x1 = (D - sz) // 2, (H - sz) // 2, (W - sz) // 2
        else:
            cz, cy, cx = coords.mean(axis=0).astype(int)
            if self.rand_offset > 0:
                cz += np.random.randint(-self.rand_offset, self.rand_offset + 1)
                cy += np.random.randint(-self.rand_offset, self.rand_offset + 1)
                cx += np.random.randint(-self.rand_offset, self.rand_offset + 1)
            z1 = max(0, min(cz - half, D - sz))
            y1 = max(0, min(cy - half, H - sz))
            x1 = max(0, min(cx - half, W - sz))
        for key in self.keys:
            d[key] = d[key][:, z1:z1 + sz, y1:y1 + sz, x1:x1 + sz]
        return d


def train_transform():
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                            mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=500.0,
                                        b_min=0.0, b_max=1.0, clip=True),
        CropAroundLabeld(keys=["image", "label"], roi_size=ROI_SIZE, rand_offset=8),
        transforms.ToTensord(keys=["image", "label"]),
    ])


def val_transform():
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                            mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=500.0,
                                        b_min=0.0, b_max=1.0, clip=True),
        CropAroundLabeld(keys=["image", "label"], roi_size=ROI_SIZE, rand_offset=0),
        transforms.ToTensord(keys=["image", "label"]),
    ])


def compute_slope(label):
    """从植入体 mask 计算斜率参数 [k1, k2]（论文 Eq. k^1/k^2，x-z 与 y-z 平面）
    label: [B, 1, D, H, W] binary tensor -> [B, 2]
    """
    B = label.shape[0]
    out = []
    for b in range(B):
        coords = torch.nonzero(label[b, 0] > 0.5).float()  # [N, 3] = (z, y, x)
        if coords.shape[0] < 3:
            out.append(torch.zeros(2, device=label.device))
            continue
        x, y, z = coords[:, 2], coords[:, 1], coords[:, 0]
        N = x.shape[0]
        den = N * (z * z).sum() - z.sum() * z.sum()
        k1 = (N * (x * z).sum() - x.sum() * z.sum()) / (den + 1e-8)
        k2 = (N * (y * z).sum() - y.sum() * z.sum()) / (den + 1e-8)
        out.append(torch.stack([k1, k2]))
    return torch.stack(out)  # [B, 2]


def build_loaders(rank, world_size, batch_size):
    mask_dir = os.path.join(DATA_DIR, 'train', 'mask')
    label_dir = os.path.join(DATA_DIR, 'train', 'label')
    train_samples = []
    for fname in sorted(os.listdir(mask_dir)):
        lp = os.path.join(label_dir, fname)
        if os.path.exists(lp):
            train_samples.append({'image': os.path.join(mask_dir, fname), 'label': lp})

    tmask_dir = os.path.join(DATA_DIR, 'test', 'mask')
    tlabel_dir = os.path.join(DATA_DIR, 'test', 'label')
    val_samples = []
    for fname in sorted(os.listdir(tmask_dir)):
        lp = os.path.join(tlabel_dir, fname)
        if os.path.exists(lp):
            val_samples.append({'image': os.path.join(tmask_dir, fname), 'label': lp})

    train_ds = data.Dataset(data=train_samples, transform=train_transform())
    val_ds = data.Dataset(data=val_samples, transform=val_transform())

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader, len(train_samples), len(val_samples)


def run(rank, world_size, args):
    setup_ddp()
    device = torch.device(f'cuda:{rank}')

    train_loader, val_loader, n_train, n_val = build_loaders(rank, world_size, args.batch_size)
    if is_main():
        print(f'[main] train={n_train} val={n_val} world={world_size} '
              f'batch/GPU={args.batch_size} effective_batch={args.batch_size * world_size}')

    model = EnhancedBasicUNet(
        spatial_dims=3, in_channels=1, out_channels=OUT_CHANNELS,
        features=(32, 32, 64, 128, 256, 32),
        use_enhanced_head=True, fusion_type='graph', dilation_rates=(1, 2, 5),
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    seg_criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    slope_criterion = nn.L1Loss()
    # 邻接矩阵单独用更大的学习率，使其更快学到结构化聚合
    adj_params = [p for n, p in model.named_parameters() if 'adj' in n]
    other_params = [p for n, p in model.named_parameters() if 'adj' not in n]
    optimizer = torch.optim.AdamW([
        {'params': other_params},
        {'params': adj_params, 'lr': args.lr * args.adj_lr_scale},
    ], lr=args.lr, weight_decay=5e-5)

    # warmup + cosine
    warmup_iters = args.warmup_epochs * len(train_loader)
    total_iters = args.epochs * len(train_loader)

    def lr_lambda(it):
        if it < warmup_iters:
            return it / max(1, warmup_iters)
        prog = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output, exist_ok=True)
    best_dice = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        seg_loss_sum, slope_loss_sum, n_b = 0.0, 0.0, 0

        for batch in train_loader:
            img = batch['image'].to(device)
            lbl = batch['label'].to(device)

            seg, reg = model(img)
            seg_loss = seg_criterion(seg, lbl)
            slope_gt = compute_slope(lbl)
            slope_loss = slope_criterion(reg, slope_gt)
            loss = seg_loss + args.slope_weight * slope_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            seg_loss_sum += seg_loss.item()
            slope_loss_sum += slope_loss.item()
            n_b += 1
            global_step += 1

        # DDP 聚合
        def red(v):
            t = torch.tensor(v, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        n_b_all = red(float(n_b))
        seg_avg = red(seg_loss_sum) / n_b_all
        slope_avg = red(slope_loss_sum) / n_b_all

        # 验证
        model.eval()
        val_dice_sum, val_n = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                lbl = batch['label'].to(device)
                seg, reg = model(img)
                pred = seg.argmax(1, keepdim=True).float()
                inter = (pred * lbl).sum()
                union = pred.sum() + lbl.sum()
                val_dice_sum += (2 * inter / (union + 1e-8)).item()
                val_n += 1.0

        val_n_all = red(val_n)
        val_dice = red(val_dice_sum) / val_n_all

        if is_main():
            print(f'Epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s) '
                  f'seg_loss={seg_avg:.4f} slope_loss={slope_avg:.4f} '
                  f'val_dice={val_dice:.4f} lr={scheduler.get_last_lr()[0]:.2e}')

            # 每个 epoch 都保存权重
            ckpt_ep = os.path.join(args.output, f'RegFreeNet_msg_ep{epoch + 1}.pt')
            torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(),
                        'val_dice': val_dice, 'optimizer': optimizer.state_dict()}, ckpt_ep)

            if val_dice > best_dice:
                best_dice = val_dice
                ckpt = os.path.join(args.output, f'RegFreeNet_msg_best.pt')
                torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(),
                            'val_dice': val_dice, 'optimizer': optimizer.state_dict()}, ckpt)
                print(f'  -> saved best {ckpt} (val_dice={val_dice:.4f})')

    if is_main():
        print(f'Done. best val_dice={best_dice:.4f}')
    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--slope_weight', type=float, default=1.0)
    parser.add_argument('--adj_lr_scale', type=float, default=10.0)
    parser.add_argument('--output', type=str, default='/data1/xinyuan/CBCT/ImplantFairy/weight')
    args = parser.parse_args()

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    run(rank, world_size, args)
