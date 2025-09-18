# Joint-OCTAMamba
Joint-OCTAMamba official source code repository.
Our code is shown in our_model.
More details will be published soon.
# OCTA 数据集说明

## 数据集结构
```
dataset/
├── OCTA500_3M/                 # 3M分辨率OCTA数据集（多任务：FAZ+RV）
│   ├── train/
│   │   ├── image/             # 原始OCTA扫描图像
│   │   ├── label_FAZ/        # FAZ分割标注
│   │   └── label_RV/         # 视网膜血管分割标注
│   ├── val/
│   │   ├── image/
│   │   ├── label_FAZ/
│   │   └── label_RV/
│   └── test/                  # 测试集（10451-10500）
│       ├── image/
│       ├── label_FAZ/
│       └── label_RV/
│
├── OCTA500_3M_FAZ/            # 3M分辨率OCTA数据集（仅FAZ任务）
│   ├── train/
│   │   ├── image/
│   │   └── label/            # 仅FAZ标注
│   ├── val/
│   │   ├── image/
│   │   └── label/
│   └── test/
│       ├── image/
│       └── label/
│
└── OCTA500_6M/                # 6M分辨率OCTA数据集
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
