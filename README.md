# Joint-OCTAMamba
Joint-OCTAMamba official source code repository.
Our code is shown in our_model.The pth will be published soon.
# OCTA 数据集说明
OCTA_500_3M_XX is the dataset layout used for single-network training.
## 数据集结构
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
# OCTA 数据集说明
OCTA_500_3M_XX is the dataset layout used for single-network training.
