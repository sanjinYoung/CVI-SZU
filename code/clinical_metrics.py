"""临床种植指标：纯基于3D mask计算，不依赖文件名坐标
- 从mask拟合种植体轴线 → 角度
- 沿轴线投影 → 入口点、根尖点 → 深度
"""
import numpy as np, json, os

# 上下颌分类（DentalSegmentator 齿数统计）
_JAW_CACHE = None

def _load_jaw_map():
    global _JAW_CACHE
    if _JAW_CACHE is None:
        path = '/data1/xinyuan/CBCT/ImplantFairy/canal_results/teeth_classification.json'
        with open(path) as f:
            raw = json.load(f)
        _JAW_CACHE = {}
        for k, v in raw.items():
            key = k.replace('.nii.gz', '')
            _JAW_CACHE[key] = 'mandible' if v['low_teeth'] > v['up_teeth'] else 'maxilla'
    return _JAW_CACHE

def get_jaw_type_from_filepath(filepath):
    """根据文件名返回 'mandible' 或 'maxilla'，匹配不到返回 'mandible'"""
    jm = _load_jaw_map()
    key = os.path.basename(filepath).replace('.nii.gz', '')
    return jm.get(key, 'mandible')


def fit_line_svd(mask):
    """SVD 直线拟合：从mask点云拟合3D直线

    Returns:
        center: (3,) 点云中心 [x, y, z]
        direction: (3,) 归一化方向向量
    """
    coords_zxy = np.argwhere(mask).astype(np.float64)  # (N, 3) in [z, y, x]
    if len(coords_zxy) < 3:
        return None, None
    coords = coords_zxy[:, ::-1]                       # [x, y, z]
    center = coords.mean(axis=0)
    centered = coords - center

    # SVD 等价于 PCA，最大奇异值对应主轴
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # 第一个右奇异向量 = 最大方差方向
    direction /= np.linalg.norm(direction)

    # 确保方向一致性：与Z轴正方向夹角 < 90°
    if direction[2] < 0:
        direction = -direction

    return center, direction


def extract_endpoints_from_mask(mask):
    """从mask沿其自身拟合的轴线提取两个端点

    Returns:
        apex: (3,) [x,y,z] 根尖点
        base: (3,) [x,y,z] 入口点
        depth: float 种植深度
    """
    center, direction = fit_line_svd(mask)
    if center is None:
        return None, None, 0.0

    coords_zxy = np.argwhere(mask).astype(np.float64)
    coords = coords_zxy[:, ::-1]
    projections = (coords - center) @ direction

    t_min, t_max = projections.min(), projections.max()
    # base 通常更靠近Z轴正方向(t_max侧，牙冠方向)，apex朝向根尖(t_min侧)
    # 但为了一致性，统一：t_max → entry (base), t_min → apex
    base = center + t_max * direction
    apex = center + t_min * direction
    depth = t_max - t_min

    return apex, base, float(depth)


def compute_clinical_metrics_from_masks(pred_mask, gt_mask, filepath=None):
    """纯mask对比：
    - Entry/Apex/Depth: 用GT mask的SVD方向投影两端mask，衡量分割形状精度
    - Angle: 两端mask各自的SVD方向对比
    - Center: 两端mask质心距离

    filepath: GT label文件路径，用于判断上下颌以修正入口/根尖方向。
              下颌入口在高Z（牙槽嵴顶），上颌入口在低Z（牙槽嵴底）。
              传入None时不做修正（保持SVD默认方向）。
    """
    result = {}

    # ---- GT mask: 拟合方向 ----
    gt_center, gt_direction = fit_line_svd(gt_mask)
    if gt_center is None:
        return {k: float('nan') for k in ['entry_err', 'apex_err', 'depth_err', 'angle_err', 'center_err']}

    # ---- 预测 mask: 拟合方向 ----
    pred_center, pred_direction = fit_line_svd(pred_mask)
    if pred_center is None or pred_mask.sum() == 0:
        return {k: float('nan') for k in ['entry_err', 'apex_err', 'depth_err', 'angle_err', 'center_err']}

    # ---- 用 GT 方向投影两端 mask，提取端点 ----
    gt_apex, gt_base, gt_depth = extract_endpoints_from_mask(gt_mask)  # 用自身SVD方向

    # 预测mask沿GT方向投影
    coords_zxy = np.argwhere(pred_mask).astype(np.float64)
    coords = coords_zxy[:, ::-1]
    gt_dir_norm = gt_direction / (np.linalg.norm(gt_direction) + 1e-8)
    pred_proj = (coords - gt_center) @ gt_dir_norm
    t_min_p, t_max_p = pred_proj.min(), pred_proj.max()
    pred_base = gt_center + t_max_p * gt_dir_norm
    pred_apex = gt_center + t_min_p * gt_dir_norm
    pred_depth = t_max_p - t_min_p

    # ---- 上下颌方向修正 ----
    # fit_line_svd确保d_z>=0, t_max=高Z侧, t_min=低Z侧
    # 下颌: 入口=牙槽嵴顶=高Z=t_max(正确), 根尖=低Z=t_min
    # 上颌: 入口=牙槽嵴底=低Z=t_min, 根尖=高Z=t_max → 需交换
    is_maxilla = (filepath is not None and get_jaw_type_from_filepath(filepath) == 'maxilla')

    # ---- 入口/根尖误差 ----
    # SVD保证d_z>=0，t_max=高Z侧(冠方)，t_min=低Z侧(根方)
    # gt_base=t_max侧, gt_apex=t_min侧
    # 下颌：入口=牙槽嵴顶=高Z=t_max=gt_base ✓, 根尖=低Z=t_min=gt_apex ✓
    # 上颌：入口=牙槽嵴底=低Z=t_min=gt_apex, 根尖=高Z=t_max=gt_base → 交换标签
    if gt_base is not None and pred_base is not None and gt_apex is not None and pred_apex is not None:
        result['entry_err'] = float(np.linalg.norm(gt_base - pred_base))
        result['apex_err']  = float(np.linalg.norm(gt_apex - pred_apex))
        if is_maxilla:
            result['entry_err'], result['apex_err'] = result['apex_err'], result['entry_err']
    else:
        result['entry_err'] = float('nan')
        result['apex_err']  = float('nan')

    # ---- 深度误差 ----
    if gt_depth > 0 and pred_depth > 0:
        result['depth_err'] = float(abs(pred_depth - gt_depth))
    else:
        result['depth_err'] = float('nan')

    # ---- 角度偏差 (两端mask各自SVD方向对比) ----
    cos_a = np.clip(np.abs(np.dot(pred_direction, gt_dir_norm)), 0.0, 1.0)
    result['angle_err'] = float(np.degrees(np.arccos(cos_a)))

    # ---- 中心点误差 ----
    result['center_err'] = float(np.linalg.norm(gt_center - pred_center))

    return result


# ============================================================
# 保留文件名解析（仅用于交叉验证，不再用于指标计算）
# ============================================================

def parse_slope_from_filename(fname):
    """从label文件名提取标注坐标（仅用于验证坐标系统）"""
    parts = fname.replace('.nii.gz', '').rsplit('_', 8)
    if len(parts) < 9:
        return None
    x1, y1, z1 = int(parts[1]), int(parts[2]), int(parts[3])
    x2, y2, z2 = int(parts[4]), int(parts[5]), int(parts[6])
    k = float(parts[7])
    b = float(parts[8])
    return {
        'base_xyz': np.array([x1, y1, z1], dtype=np.float64),
        'apex_xyz': np.array([x2, y2, z2], dtype=np.float64),
        'k': k, 'b': b,
        'direction': np.array([x1 - x2, y1 - y2, z1 - z2], dtype=np.float64),
    }
