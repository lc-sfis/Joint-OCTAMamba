# filename: dataset.py (已更新以支持多数据集)

import os
import cv2
import torch
import numpy as np
import albumentations as alb
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

# [MODIFIED] 基础数据目录，而不是特定数据集的目录
DATA_ROOT_BASE = "dataset"
# [NEW] 定义已知的数据集名称
KNOWN_DATASETS = ["OCTA500_3M", "OCTA500_6M"]

def prepareDatasets() -> Dict[str, Dict[str, Dataset]]:
    """
    [MODIFIED] 工厂函数，自动检测并创建所有已知数据集的实例。
    它会查找 "dataset/OCTA500_3M" 和 "dataset/OCTA_6M"。
    """
    all_datasets = {}
    print("--- 正在准备数据集 ---")
    for name in KNOWN_DATASETS:
        dataset_path = os.path.join(DATA_ROOT_BASE, name)
        if not os.path.exists(dataset_path):
            print(f"🔍 未找到数据集 '{name}' 的路径: {dataset_path}, 已跳过。")
            continue

        print(f"✅ 成功找到数据集: '{name}'")
        splits = {}
        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                is_training = (split == "train")
                splits[split] = SegmentationDataset(split_path, is_training=is_training)
        
        if splits:
            all_datasets[name] = splits
            
    if not all_datasets:
        raise FileNotFoundError("错误：在 'dataset/' 目录下未找到任何有效的数据集 ('OCTA500_3M' 或 'OCTA_6M')。")
        
    return all_datasets

class SegmentationDataset(Dataset):
    """
    为多任务分割准备数据。此类无需修改，因为它本身就是通用的。
    """
    def __init__(self, path: str, *, is_training: bool):
        super().__init__()
        self.is_training = is_training
        self.items: List[Dict[str, str]] = []
        
        image_dir = os.path.join(path, "image")
        rv_dir = os.path.join(path, "label_RV")
        faz_dir = os.path.join(path, "label_FAZ")

        for f_name in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, f_name)
            if os.path.exists(img_path):
                self.items.append({
                    "name": f_name,
                    "img": img_path,
                    "rv": os.path.join(rv_dir, f_name),
                    "faz": os.path.join(faz_dir, f_name)
                })

        if self.is_training:
            p = 0.2
            self.aug = alb.Compose([
                alb.RandomBrightnessContrast(p=p),
                alb.CLAHE(p=p),
                alb.Rotate(limit=15, p=p),
                alb.VerticalFlip(p=p),
                alb.HorizontalFlip(p=p),
                alb.PiecewiseAffine(p=p),
            ], additional_targets={"mask_rv": "mask", "mask_faz": "mask"})
        else:
            self.aug = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, Dict[str, torch.Tensor]]:
        item = self.items[idx]
        
        img = cv2.imread(item["img"], cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        rv_mask = cv2.imread(item["rv"], cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        faz_mask = cv2.imread(item["faz"], cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        
        if self.aug:
            augmented = self.aug(image=img, mask_rv=rv_mask, mask_faz=faz_mask)
            img = augmented["image"]
            rv_mask = augmented["mask_rv"]
            faz_mask = augmented["mask_faz"]
        
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        rv_tensor = torch.from_numpy(rv_mask).unsqueeze(0)
        faz_tensor = torch.from_numpy(faz_mask).unsqueeze(0)
        
        labels = {"rv": rv_tensor, "faz": faz_tensor}
        
        return item["name"], img_tensor, labels

