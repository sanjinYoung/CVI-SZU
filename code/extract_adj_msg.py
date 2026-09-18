"""提取 RegFreeNet_msg 的 64x64 邻接矩阵并可视化"""
import torch, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dilated_unet_reg_msg import EnhancedBasicUNet

CKPT = '/data1/xinyuan/CBCT/ImplantFairy/weight/RegFreeNet_msg_best.pt'
OUT = '/data1/xinyuan/CBCT/ImplantFairy'

model = EnhancedBasicUNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    features=(32, 32, 64, 128, 256, 32),
    use_enhanced_head=True, fusion_type='graph', dilation_rates=(1, 2, 5),
)
ckpt = torch.load(CKPT, map_location='cpu')
sd = ckpt.get('state_dict', ckpt)
sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)

# 提取邻接矩阵
adj = model.conv_0.enhanced_fusion.adj.detach().cpu()   # [64, 64]
alpha = torch.softmax(adj, dim=-1)                        # message passing 系数（非负）

print('adj shape:', tuple(adj.shape))
print('adj    min/max/mean/std = %.4f / %.4f / %.4f / %.4f' %
      (adj.min(), adj.max(), adj.mean(), adj.std()))
print('alpha  min/max/mean = %.4f / %.4f / %.4f' %
      (alpha.min(), alpha.max(), alpha.mean()))

# 每个节点的出边熵（信息分布）
ent = -(alpha * torch.log(alpha + 1e-12)).sum(dim=-1)
print('alpha 每行熵: min=%.4f max=%.4f mean=%.4f' % (ent.min(), ent.max(), ent.mean()))

# 对角 vs 非对角（自环 vs 跨节点）
diag = alpha.diagonal()
off = alpha - torch.diag_embed(diag)
print('alpha 自环均值=%.4f  跨节点均值=%.4f' % (diag.mean(), off.mean()))

# 可视化原始 adj（coolwarm 对称，展示正负）
vmax = adj.abs().max().item()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(adj.numpy(), cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='A (learned adjacency)')
ax.set_title('Learned 64x64 GCN adjacency matrix', fontsize=14)
ax.set_xlabel('Target node (j)', fontsize=12)
ax.set_ylabel('Source node (i)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/figs/gcn_adjacency_64x64.png', dpi=200, bbox_inches='tight')
print('saved png')

# 保存数值（供 rebuttal 描述）
json.dump({'adj': adj.numpy().tolist(), 'alpha': alpha.numpy().tolist()},
          open(f'{OUT}/gcn_adjacency_64x64.json', 'w'))
print('saved json')
