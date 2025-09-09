# filename: predict_multitask.py
"""
多任务（RV & FAZ）模型预测脚本。

该脚本能够加载一个预训练的多任务模型，对指定的测试集进行预测，
并为每个任务（RV, FAZ）分别生成和保存高质量的预测分割图。
支持输出PNG、PDF、SVG等多种格式，方便用于论文插图。
"""
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

# --- 确保能正确导入项目中的模块 ---
# 将项目根目录添加到Python路径中
# 这使得脚本可以从任何位置运行，只要它能找到 'our_model' 和 'dataset' 目录
# CWD = os.getcwd()
# if CWD not in sys.path:
#     sys.path.append(CWD)

try:
    from dataset import SegmentationDataset
except ImportError:
    print("错误: 无法导入 'SegmentationDataset'。")
    print("请确保 'predict_multitask.py' 与 'dataset.py' 在同一项目结构下，")
    print("并且您从项目的根目录运行此脚本。")
    sys.exit(1)


def import_model_from_our_model(model_name: str):
    """
    动态从 'our_model' 目录导入模型类。
    """
    try:
        # 假设模型文件和类名相同
        module_path = f"our_model.{model_name}"
        module = importlib.import_module(module_path)
        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        print(f"错误：无法从 'our_model' 目录动态导入模型 '{model_name}'。")
        print("请检查以下几点：")
        print(f"1. 项目根目录下是否存在 'our_model' 文件夹。")
        print(f"2. 'our_model' 文件夹中是否存在一个名为 '{model_name}.py' 的文件。")
        print(f"3. '{model_name}.py' 文件中是否存在一个名为 '{model_name}' 的类。")
        print(f"4. 'our_model' 文件夹中是否存在一个 (可以是空的) '__init__.py' 文件。")
        print(f"\n原始错误信息: {e}")
        sys.exit(1)


@torch.no_grad()
def main():
    # --- 1. 参数解析 ---
    parser = argparse.ArgumentParser(description="多任务（RV & FAZ）模型预测脚本")
    
    parser.add_argument('--model_name', type=str, required=True, help="要加载的多任务模型名称 (例如: MultiTaskOCTAMamba_FARGO_Interactive)")
    parser.add_argument('--weight_path', type=str, required=True, help='预训练权重文件的路径 (.pth)')
    parser.add_argument('--dataset_name', type=str, required=True, help='要测试的数据集名称 (例如: OCTA500_3M)')
    
    parser.add_argument('--data_root', type=str, default='./dataset', help='数据集的根目录')
    parser.add_argument('--base_output_dir', type=str, default='./predictions', help='保存预测结果的基础目录')
    parser.add_argument('--faz_crop_size', type=int, default=192, help="FAZ 任务的裁剪尺寸，必须与训练时一致")
    parser.add_argument('--num_samples', type=int, default=-1, help='要处理的样本数量 (-1 表示全部)')
    parser.add_argument('--gpu_id', type=int, default=0, help='要使用的GPU ID')
    
    parser.add_argument(
        '--format', 
        type=str, 
        default='png', 
        choices=['png', 'pdf', 'svg'], 
        help='输出图片的格式。pdf 和 svg 是矢量格式，推荐用于论文。'
    )
    
    args = parser.parse_args()
    
    # --- 2. 环境设置 ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"--- 步骤1: 环境设置 ---\n将使用设备: {device}\n")
    
    # --- 3. 准备数据 ---
    print(f"--- 步骤2: 数据准备 ---")
    test_data_path = os.path.join(args.data_root, args.dataset_name, "test")
    if not os.path.exists(test_data_path):
        print(f"错误: 找不到测试数据路径 '{test_data_path}'。请检查 --data_root 和 --dataset_name 参数。")
        return

    test_dataset = SegmentationDataset(path=test_data_path, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f"在 '{args.dataset_name}' 测试集中找到 {len(test_dataset)} 个样本。\n")

    # --- 4. 加载模型与权重 ---
    print(f"--- 步骤3: 模型与权重加载 ---")
    ModelClass = import_model_from_our_model(args.model_name)
    
    # 根据模型定义实例化，传入必要的参数
    model = ModelClass(
        tasks=[args.dataset_name], 
        faz_crop_size=args.faz_crop_size,
        end_to_end=False  # 推理时梯度不重要，设为False
    ).to(device)
    print(f"模型 '{args.model_name}' 已实例化。")

    if os.path.exists(args.weight_path):
        try:
            model.load_state_dict(torch.load(args.weight_path, map_location=device))
            print(f"成功从 '{args.weight_path}' 加载权重。")
        except RuntimeError as e:
            print(f"错误: 加载权重时发生错误。可能是模型定义与权重文件不匹配。")
            print(f"请确保您使用的 --model_name 和 --faz_crop_size 与训练时完全一致。")
            print(f"原始错误: {e}")
            return
    else:
        print(f"错误: 权重文件 '{args.weight_path}' 不存在！"); return
    
    model.eval()
    print("模型已设置为评估模式 (model.eval())。\n")

    # --- 5. 开始预测并保存结果 ---
    print(f"--- 步骤4: 开始预测并保存结果 ---")
    
    # 创建更详细的输出目录
    output_dir_rv = os.path.join(args.base_output_dir, args.model_name, args.dataset_name, f'rv_{args.format}')
    output_dir_faz = os.path.join(args.base_output_dir, args.model_name, args.dataset_name, f'faz_{args.format}')
    os.makedirs(output_dir_rv, exist_ok=True)
    os.makedirs(output_dir_faz, exist_ok=True)
    print(f"RV  预测结果将以 {args.format.upper()} 格式保存在: {output_dir_rv}")
    print(f"FAZ 预测结果将以 {args.format.upper()} 格式保存在: {output_dir_faz}\n")
    
    num_to_process = len(test_loader) if args.num_samples == -1 else min(args.num_samples, len(test_loader))
    pbar = tqdm(enumerate(test_loader), total=num_to_process, desc="处理样本", ncols=100)

    for i, (name, image_tensor, _) in pbar:
        if i >= num_to_process: break
        
        image_tensor_gpu = image_tensor.to(device)
        
        # 模型返回一个包含多个任务输出的字典
        pred_dict = model(image_tensor_gpu)
        
        # 处理每个任务的输出
        for task_name, pred_tensor in pred_dict.items():
            # 应用阈值得到二值图像
            pred_binary = (pred_tensor > 0.5).float()
            pred_display = pred_binary.squeeze().cpu().numpy()
            
            # --- 创建并保存只包含预测图的图像 ---
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(pred_display, cmap='gray')
            ax.axis('off') # 关闭坐标轴，得到纯净的图像
            fig.tight_layout(pad=0) # 调整布局，确保没有多余的白边
            
            # 构建输出文件名
            base_name = os.path.splitext(name[0])[0]
            output_filename = f"{base_name}.{args.format}"
            
            # 根据任务名称选择正确的输出目录
            if 'rv' in task_name.lower():
                output_path = os.path.join(output_dir_rv, output_filename)
            elif 'faz' in task_name.lower():
                output_path = os.path.join(output_dir_faz, output_filename)
            else:
                # 如果有其他未知的任务输出，跳过保存
                plt.close(fig)
                continue

            # 根据格式动态保存
            if args.format == 'png':
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            else: # PDF, SVG (矢量格式)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                
            plt.close(fig) # 及时关闭图像，防止内存占用

    print(f"\n--- 所有样本处理完毕 ---")
    print(f"✅ 共 {num_to_process} 张图像的 RV 和 FAZ 预测结果已保存。")
    print(f"   -> RV  保存在: '{output_dir_rv}'")
    print(f"   -> FAZ 保存在: '{output_dir_faz}'")

if __name__ == '__main__':
    main()
