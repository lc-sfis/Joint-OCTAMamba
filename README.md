# Joint-OCTAMamba
Joint-OCTAMamba official source code repository.
Our code is shown in our_model.The pth will be published soon.
# OCTA Dataset Description
OCTA_500_3M_XX is the dataset layout used for single-network training.
## Dataset Structure
```
dataset/
├── OCTA500_3M/                 # 3M resolution OCTA dataset (multi-task: FAZ + RV)
│   ├── train/
│   │   ├── image/             # Original OCTA scan images
│   │   ├── label_FAZ/         # FAZ segmentation labels
│   │   └── label_RV/          # Retinal vessel segmentation labels
│   ├── val/
│   │   ├── image/
│   │   ├── label_FAZ/
│   │   └── label_RV/
│   └── test/                  # Test set (10451–10500)
│       ├── image/
│       ├── label_FAZ/
│       └── label_RV/
│
├── OCTA500_3M_FAZ/             # 3M resolution OCTA dataset (FAZ task only)
│   ├── train/
│   │   ├── image/
│   │   └── label/             # FAZ labels only
│   ├── val/
│   │   ├── image/
│   │   └── label/
│   └── test/
│       ├── image/
│       └── label/
│
├── OCTA500_3M_RV/              # 3M resolution OCTA dataset (RV task only)
│   ├── train/
│   │   ├── image/
│   │   └── label/             # RV labels only
│   ├── val/
│   │   ├── image/
│   │   └── label/
│   └── test/
│       ├── image/
│       └── label/
│
└── OCTA500_6M/                 # 6M resolution OCTA dataset
    ├── train/
    │   ├── image/
    │   ├── label_FAZ/
    │   └── label_RV/
    ├── val/
    │   ├── image/
    │   ├── label_FAZ/
    │   └── label_RV/
    └── test/
        ├── image/
        ├── label_FAZ/
        └── label_RV/
mage/
        ├── label_FAZ/
        └── label_RV/
```
# Usage Instructions
## Training

You can start training by using the commands provided below.

### Multi-task Training on OCTA500 3M

```bash
python run_multitask_aligned.py \
    --dataset OCTA500_3M \
    --gpu 0 \
    --batch 2 \
    --epochs 100 \
    --faz_weight 6.1 \
    --use_tta \
    --faz_crop 224
```

### Multi-task Training on OCTA500 6M

```bash
python run_multitask_aligned.py \
    --dataset OCTA500_6M \
    --gpu 0 \
    --batch 2 \
    --epochs 100 \
    --faz_weight 4.0 \
    --use_tta \
    --faz_crop 224
```

### Notes

* `--dataset`: Choose between **OCTA500\_3M** and **OCTA500\_6M** depending on resolution.
* `--gpu`: Specify the GPU ID to use (e.g., 0 for the first GPU).
* `--batch`: Batch size.
* `--epochs`: Number of training epochs.
* `--faz_weight`: Loss weight for the FAZ segmentation task.
* `--use_tta`: Simple test-time augmentation.
* `--faz_crop`: Crop size for FAZ region extraction.

👉 *You can start training by using the commands provided above.*




