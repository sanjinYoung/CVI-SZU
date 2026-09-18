# ImplantFairy

**A Registration-Free and Slope-aware Framework for CBCT-based Dental Implant Planning**

*IEEE Transactions on Medical Imaging, Paper ID: TMI-2026-1309*

ImplantFairy is the first comprehensive dental implant dataset constructed **without** pre-/post-operative scan registration. It contains **1,622 CBCT scans** (1,369 training / 253 test, split at the patient level) with voxel-level 3D annotations of implant position, collected from the Department of Stomatology at Shenzhen University General Hospital. All scans are acquired with a KaVo 3D eXam system at an isotropic voxel resolution of 0.2 mm.

## Download

| Split | Scans | Size | Access |
|-------|-------|------|--------|
| Test set | 253 | ~25 GB | [Google Drive](https://drive.google.com/drive/folders/1kDhMiK2ipAVcFJeOBE43OpJ0Eo2tebZe?usp=drive_link) |
| Training set | 1,369 | ~135 GB | Upon request (academic, non-commercial use only). Email: `944285052@qq.com` |

To request the training set, please send an email with your affiliation and a brief description of the intended use. By requesting access, you agree to use the data solely for academic research and not for any commercial purpose.

## Data Format

Each scan is a `.nii.gz` volume. For each implant, clinicians annotated three landmarks along the implant's longitudinal axis (vertex, midpoint, and base point), which are converted into a cylindrical 3D mask. The directory layout is:

```
├── mask/     # CBCT volume with the implant region masked
└── label/    # corresponding binary implant mask
```

The implant mask covers the implant body and a surrounding region of ~7--9 mm, so that the model predicts the implant position solely from the surrounding oral anatomy (adjacent teeth, alveolar ridge, and bone density).

## Code

The `code/` directory contains the model definition, training, and evaluation scripts of the RegFreeNet framework:

- `dilated_unet_reg_msg.py` — RegFreeNet network (multi-scale dilated perception + graph-based context aggregation + slope prediction branch)
- `train_ddp_msg.py` — multi-GPU (DDP) training script
- `clinical_metrics.py` — Entry/Apex/Angle/Center clinical metrics
- `eval_single.py`, `eval_sanet.py` — evaluation scripts
- `eval_canal_safety.py` — mandibular canal safety analysis
- `extract_adj_msg.py` — graph adjacency visualization

## Citation

If you use ImplantFairy in your research, please cite:

```
<TODO: 论文完整引用，接收后填卷期页码>
```

## License

The dataset and code are released for academic research purposes only. Commercial use requires a separate agreement.
