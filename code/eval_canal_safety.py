"""NDP消融: 下颌管安全距离评估 (110 mandibular samples)"""
import os, json, time, gc
import numpy as np, torch, nibabel
from scipy.ndimage import zoom
from sklearn.neighbors import NearestNeighbors

# ---- DentalSegmentator canal mask ----
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
MODEL_DIR = '/data1/xinyuan/CBCT/ImplantFairy/dental_segmentator/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres'
DATA_DIR = '/data1/xinyuan/CBCT/ImplantFairy'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---- 加载 canal 预测器 ----
print("Loading DentalSegmentator...")
predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, device=DEVICE, verbose=False)
predictor.initialize_from_trained_model_folder(MODEL_DIR, use_folds=(0,), checkpoint_name='checkpoint_final.pth')
canal_model = predictor.network.to(DEVICE).eval()

# ---- 识别下颌样本 ----
with open(f'{DATA_DIR}/canal_results/teeth_classification.json') as f:
    tc = json.load(f)

mask_dir = f'{DATA_DIR}/test/mask'
label_dir = f'{DATA_DIR}/test/label'
mand_files = []
for fname in sorted(os.listdir(mask_dir)):
    key = fname.replace('.nii.gz', '')
    if fname in tc:
        info = tc[fname]
    elif key in tc:
        info = tc[key]
    else:
        info = None
    if info is not None and info['low_teeth'] > info['up_teeth']:
        if os.path.exists(os.path.join(label_dir, fname)):
            mand_files.append(fname)

print(f"Mandibular samples: {len(mand_files)} (should be ~110)")

# ---- 加载模型 ----
import sys
sys.path.insert(0, '/data/xinyuan/CBCT/LLM_ssl/Large-Scale-Medical-main/Downstream/VoCOVID')
sys.path.insert(0, DATA_DIR)
from dilated_unet_reg_yuanbao_noreg_1212_reg import EnhancedBasicUNet
from monai.inferers import sliding_window_inference

FEATURES = (32, 32, 64, 128, 256, 32)
ROI_SIZE = (128, 128, 128)

def load_model(ckpt_path, use_enhanced):
    model = EnhancedBasicUNet(spatial_dims=3, in_channels=1, out_channels=2,
                               features=FEATURES, use_enhanced_head=use_enhanced,
                               fusion_type='graph', dilation_rates=(1,2,5)).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

print("Loading models...")
model_nondp = load_model(f'{DATA_DIR}/weight/SANet-onlyreg.pt_49.pt', use_enhanced=False)
model_withdp = load_model(f'{DATA_DIR}/weight/SANet(finetune)-onlyreg.pt_49.pt', use_enhanced=True)

# ---- 主循环 ----
def crop_128_around_implant(img_data, lbl_data):
    """Crop 128³ around implant center"""
    coords = np.argwhere(lbl_data > 0.5).astype(np.float64)
    if len(coords) == 0:
        return None, None, None
    cz, cy, cx = coords.mean(axis=0).astype(int)
    half = 64
    D, H, W = img_data.shape
    z1 = max(0, min(cz - half, D - 128))
    y1 = max(0, min(cy - half, H - 128))
    x1 = max(0, min(cx - half, W - 128))
    img_crop = img_data[z1:z1+128, y1:y1+128, x1:x1+128].copy()
    lbl_crop = lbl_data[z1:z1+128, y1:y1+128, x1:x1+128].copy()
    offset = (z1, y1, x1)
    return img_crop, lbl_crop, offset

def get_canal_mask(data):
    """DentalSegmentator canal mask (label 5)"""
    data_clipped = np.clip(data, -1000, 3000)
    data_norm = (data_clipped - data_clipped.mean()) / (data_clipped.std() + 1e-8)
    vol = torch.from_numpy(data_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = sliding_window_inference(vol, roi_size=(128,128,112), sw_batch_size=2,
                                         predictor=canal_model, overlap=0.5, mode='gaussian')
    labels = pred.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return (labels == 5).astype(np.uint8)

def predict_implant(model, img_crop):
    """Model inference on 128³ crop"""
    img = np.clip(img_crop, -1000, 500)
    img = (img + 1000) / 1500
    vol = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(vol)
        if isinstance(logits, tuple):
            logits = logits[0]
    pm = logits.argmax(1).cpu().numpy().astype(np.uint8)[0]
    return pm

def compute_canal_dist(pred_mask, canal_mask):
    """Min Euclidean distance from pred implant voxels to canal voxels"""
    if pred_mask.sum() == 0 or canal_mask.sum() == 0:
        return None, None
    pred_coords = np.argwhere(pred_mask).astype(np.float64)
    canal_coords = np.argwhere(canal_mask).astype(np.float64)
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(canal_coords)
    dists, _ = nn.kneighbors(pred_coords)
    min_dist = float(dists.min())
    collision_rate = float((dists < 1.0).mean() * 100)
    return min_dist, collision_rate

results = {'No-NDP': [], 'With-NDP': []}
t0 = time.time()

for idx, fname in enumerate(mand_files):
    img_path = os.path.join(mask_dir, fname)
    lbl_path = os.path.join(label_dir, fname)

    src = nibabel.load(img_path)
    data = src.get_fdata().astype(np.float32)
    lbl_data = (nibabel.load(lbl_path).get_fdata() > 0.5).astype(np.uint8)

    # Canal mask (full image)
    canal_mask = get_canal_mask(data)
    if canal_mask.sum() == 0:
        continue

    # Crop around implant
    img_crop, lbl_crop, offset = crop_128_around_implant(data, lbl_data)
    if img_crop is None:
        continue

    # Offsets for mapping pred back to full image
    z1, y1, x1 = offset

    for model_name, model in [('No-NDP', model_nondp), ('With-NDP', model_withdp)]:
        pm = predict_implant(model, img_crop)

        # Map pred back to full image coords
        pm_full = np.zeros(data.shape, dtype=np.uint8)
        pm_full[z1:z1+128, y1:y1+128, x1:x1+128] = pm

        min_dist, coll_rate = compute_canal_dist(pm_full, canal_mask)
        if min_dist is not None:
            results[model_name].append({'fname': fname, 'min_dist': min_dist, 'collision_rate': coll_rate})

    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{len(mand_files)}] ({time.time()-t0:.0f}s)")

# ---- 汇总 ----
print(f"\n{'='*60}")
print(f"Completed: {len(results['No-NDP'])} mandibular samples evaluated")
for name in ['No-NDP', 'With-NDP']:
    rlist = results[name]
    if not rlist:
        continue
    dists = [r['min_dist'] for r in rlist]
    colls = [r['collision_rate'] for r in rlist]
    print(f"{name:10s}: MinDist={np.mean(dists):.2f}mm (min={np.min(dists):.2f}), Collision={np.mean(colls):.2f}%")

# Save
with open(f'{DATA_DIR}/canal_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {DATA_DIR}/canal_ablation_results.json")

del model_nondp, model_withdp, canal_model; torch.cuda.empty_cache()
