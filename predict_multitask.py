# filename: predict_multitask.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import cv2
from typing import Dict, Tuple, List
import importlib
import sys


try:
    from dataset import SegmentationDataset
except ImportError:
    sys.exit(1)


def import_model_from_our_model(model_name: str):
    try:
        module_path = f"our_model.{model_name}"
        module = importlib.import_module(module_path)
        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        sys.exit(1)


@torch.no_grad()
def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="（RV & FAZ）")
    
    parser.add_argument('--model_name', type=str, required=True, help="")
    parser.add_argument('--weight_path', type=str, required=True, help='(.pth)')
    parser.add_argument('--dataset_name', type=str, required=True, help=' OCTA500_3M')
    
    parser.add_argument('--data_root', type=str, default='./dataset', help='')
    parser.add_argument('--base_output_dir', type=str, default='./predictions', help='')
    parser.add_argument('--faz_crop_size', type=int, default=192, help="")
    parser.add_argument('--num_samples', type=int, default=-1, help='')
    parser.add_argument('--gpu_id', type=int, default=0, help='')
    
    parser.add_argument(
        '--format', 
        type=str, 
        default='png', 
        choices=['png', 'pdf', 'svg'], 
        help=''
    )
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f" {device}\n")
    
    print(f"--- d2: ")
    test_data_path = os.path.join(args.data_root, args.dataset_name, "test")
    if not os.path.exists(test_data_path):
        print(f" '{test_data_path}'。 --data_root  --dataset_name 。")
        return

    test_dataset = SegmentationDataset(path=test_data_path, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    ModelClass = import_model_from_our_model(args.model_name)
    
    model = ModelClass(
        tasks=[args.dataset_name], 
        faz_crop_size=args.faz_crop_size,
        end_to_end=False  
    ).to(device)
    print(f" '{args.model_name}' 。")

    if os.path.exists(args.weight_path):
        try:
            model.load_state_dict(torch.load(args.weight_path, map_location=device))
        except RuntimeError as e:
            return
    else:
        print(f" '{args.weight_path}' not exsit！"); return
    
    model.eval()

    print(f"--- 4:  ---")
    

    output_dir_rv = os.path.join(args.base_output_dir, args.model_name, args.dataset_name, f'rv_{args.format}')
    output_dir_faz = os.path.join(args.base_output_dir, args.model_name, args.dataset_name, f'faz_{args.format}')
    os.makedirs(output_dir_rv, exist_ok=True)
    os.makedirs(output_dir_faz, exist_ok=True)
    
    num_to_process = len(test_loader) if args.num_samples == -1 else min(args.num_samples, len(test_loader))
    pbar = tqdm(enumerate(test_loader), total=num_to_process, desc="处理样本", ncols=100)

    for i, (name, image_tensor, _) in pbar:
        if i >= num_to_process: break
        
        image_tensor_gpu = image_tensor.to(device)
        

        pred_dict = model(image_tensor_gpu)
        

        for task_name, pred_tensor in pred_dict.items():

            pred_binary = (pred_tensor > 0.5).float()
            pred_display = pred_binary.squeeze().cpu().numpy()
            

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(pred_display, cmap='gray')
            ax.axis('off') 
            fig.tight_layout(pad=0) 
            

            base_name = os.path.splitext(name[0])[0]
            output_filename = f"{base_name}.{args.format}"
            
            if 'rv' in task_name.lower():
                output_path = os.path.join(output_dir_rv, output_filename)
            elif 'faz' in task_name.lower():
                output_path = os.path.join(output_dir_faz, output_filename)
            else:

                plt.close(fig)
                continue
            if args.format == 'png':
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            else: 
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                
            plt.close(fig) 


if __name__ == '__main__':
    main()
