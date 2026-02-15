from __future__ import annotations
import argparse
import datetime
import json
import os
import collections
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
import numpy as np
import cv2

from dataset import prepareDatasets
from our_model.JointOCTAFormer import RVPriorFormer as MultiTaskOCTFormer, center_crop_tensor
from our_model.JointOCTAMamba import RVPriorMamba as MultiTaskOCTAMamba, center_crop_tensor # Original import name
from loss import DiceLoss, BoundaryLoss, HausdorffLoss, TverskyLoss

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def calc_result(np_pred:np.ndarray, np_label:np.ndarray, thresh_value=None):
    temp = cv2.normalize(np_pred, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    if thresh_value is None:
        _, np_pred = cv2.threshold(temp, 0.0, 1.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, np_pred = cv2.threshold(temp, thresh_value, 1.0, cv2.THRESH_BINARY)
    
    np_pred = np_pred.flatten()
    np_label = np_label.flatten()
    uni = np.unique(np_label)
    if not ((len(uni)==2) and (1 in uni) and (0 in uni)):
        if np.sum(np_label) == 0: 
            if np.sum(np_pred) == 0: 
                return {'acc': 1.0, 'fdr': 0.0, 'sen': 1.0, 'spe': 1.0, 'gmean':1.0, 'iou': 1.0, 'dice': 1.0}
            else: 
                return {'acc': 0.0, 'fdr': 1.0, 'sen': 0.0, 'spe': 0.0, 'gmean':0.0, 'iou': 0.0, 'dice': 0.0}
        
    FP = np.sum(np.logical_and(np_pred == 1, np_label == 0)).astype(float)
    FN = np.sum(np.logical_and(np_pred == 0, np_label == 1)).astype(float)
    TP = np.sum(np.logical_and(np_pred == 1, np_label == 1)).astype(float)
    TN = np.sum(np.logical_and(np_pred == 0, np_label == 0)).astype(float)
    result = {}
    smooth = 1e-12
    result['acc'] = (TP + TN + smooth) / (FP + FN + TP + TN + smooth)
    result['fdr'] = (FP + smooth)  / (FP + TP + smooth)
    sen = (TP + smooth) / (FN + TP + smooth)
    spe = (TN + smooth) / (FP + TN + smooth)
    result['sen'] = sen
    result['spe'] = spe
    result['gmean'] = np.sqrt(sen * spe)
    result['iou'] = (TP + smooth) / (FP + FN + TP + smooth)
    result['dice'] = (2.0 * TP + smooth) / (FP + FN + 2.0 * TP + smooth)
    
    return result

def avg_result(ls_result: List[Dict[str, float]]) -> Dict[str, float]:
    total_result = collections.defaultdict(list)
    for r in ls_result:
        for key, value in r.items():
            total_result[key].append(value)
    
    avg_metrics = {}
    for key, values in total_result.items():
        avg_metrics[key] = float(np.mean(values))
        
    return avg_metrics

def faz_loss(pred, gt):
    return 0.8 * DiceLoss()(pred, gt) + 0.2 * BoundaryLoss()(pred, gt)
def rv_loss(pred, gt):
    return 0.6 * DiceLoss()(pred, gt) + 0.2 * BoundaryLoss()(pred, gt) + 0.1 * TverskyLoss()(pred, gt) +  0.1 * HausdorffLoss()(pred, gt)

@torch.no_grad()
def apply_simple_tta(model, data):
    model.eval()
    task_outputs = collections.defaultdict(list)
    
    pred_original = model(data)
    for task_name, tensor in pred_original.items():
        if isinstance(tensor, torch.Tensor):
            task_outputs[task_name].append(tensor)
    data_hflip = torch.flip(data, dims=[3])
    pred_hflip_dict = model(data_hflip)
    for task_name, tensor in pred_hflip_dict.items():
         if isinstance(tensor, torch.Tensor):
            task_outputs[task_name].append(torch.flip(tensor, dims=[3]))
    data_vflip = torch.flip(data, dims=[2])
    pred_vflip_dict = model(data_vflip)
    for task_name, tensor in pred_vflip_dict.items():
        if isinstance(tensor, torch.Tensor):
            task_outputs[task_name].append(torch.flip(tensor, dims=[2]))
    
    ensembled_output = {}
    for task_name, preds_list in task_outputs.items():
        ensembled_output[task_name] = torch.mean(torch.stack(preds_list), dim=0)
    for task_name, value in pred_original.items():
        if not isinstance(value, torch.Tensor):
            ensembled_output[task_name] = value
    return ensembled_output

def train_one_epoch(model, loader, device, optimizer,
                    faz_crop_size, rv_weight, faz_weight):
    model.train()
    total_loss = 0.0
    total_loss_rv = 0.0  # <-- MODIFIED: Initialize accumulator for RV loss
    total_loss_faz = 0.0 # <-- MODIFIED: Initialize accumulator for FAZ loss
    num_samples = 0
    
    pbar_desc = f"Train (rv_w={rv_weight:.1f}, fz_w={faz_weight:.1f})"
    pbar = tqdm(loader, ncols=140, desc=pbar_desc)
    for _, imgs, lbls in pbar:
        imgs = imgs.to(device, non_blocking=True)
        rv_gt_full = lbls["rv"].to(device, non_blocking=True)
        faz_gt_full = lbls["faz"].to(device, non_blocking=True)
        batch_size = imgs.size(0)
        with torch.set_grad_enabled(True):
            out = model(imgs)
            
            faz_gt_cropped = center_crop_tensor(faz_gt_full, faz_crop_size)
            loss_rv = rv_loss(out["rv"], rv_gt_full)
            loss_faz = faz_loss(out["faz_cropped"], faz_gt_cropped) 
            batch_loss = rv_weight * loss_rv + faz_weight * loss_faz
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            print(f"⚠️ NaN/Inf detected in loss, skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate all losses
        total_loss += batch_loss.item() * batch_size
        total_loss_rv += loss_rv.item() * batch_size   # <-- MODIFIED: Accumulate RV loss
        total_loss_faz += loss_faz.item() * batch_size # <-- MODIFIED: Accumulate FAZ loss
        num_samples += batch_size
        
        # Update progress bar with detailed losses
        pbar.set_postfix(
            loss=f"{total_loss / num_samples:.4f}",
            l_rv=f"{total_loss_rv / num_samples:.4f}",   # <-- MODIFIED: Show RV loss
            l_faz=f"{total_loss_faz / num_samples:.4f}", # <-- MODIFIED: Show FAZ loss
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    # Return all metrics in the final dictionary
    final_metrics = {
        'loss': total_loss / num_samples if num_samples > 0 else 0,
        'loss_rv': total_loss_rv / num_samples if num_samples > 0 else 0,   # <-- MODIFIED: Add rv loss to results
        'loss_faz': total_loss_faz / num_samples if num_samples > 0 else 0, # <-- MODIFIED: Add faz loss to results
    }
    final_metrics['lr'] = optimizer.param_groups[0]['lr']
    return final_metrics

@torch.no_grad()
def evaluate_one_epoch(model, loader, device, use_tta=False):
    model.eval()
    
    eval_results_rv = []
    eval_results_faz = []
    
    split_name = getattr(loader.dataset, 'split', 'dataset') # Robustly get split name
    desc = f"Eval {'(TTA)' if use_tta else ''} on {split_name} set"
    pbar = tqdm(loader, ncols=140, desc=desc)
    for _, imgs, lbls in pbar:
        imgs = imgs.to(device, non_blocking=True)
        rv_gt_full = lbls["rv"].to(device, non_blocking=True)
        faz_gt_full = lbls["faz"].to(device, non_blocking=True)
        batch_size = imgs.size(0)
        if use_tta:
            out = apply_simple_tta(model, imgs)
        else:
            out = model(imgs)
        
        for i in range(batch_size):
            pred_rv = out['rv'][i][0].cpu().numpy()
            gt_rv = rv_gt_full[i][0].cpu().numpy()
            eval_results_rv.append(calc_result(pred_rv, gt_rv))
            
            pred_faz = out['faz'][i][0].cpu().numpy() 
            gt_faz = faz_gt_full[i][0].cpu().numpy()
            eval_results_faz.append(calc_result(pred_faz, gt_faz))
        dice_rv_avg = avg_result(eval_results_rv).get('dice', 0)
        dice_faz_avg = avg_result(eval_results_faz).get('dice', 0)
        pbar.set_postfix(d_rv=f"{dice_rv_avg:.4f}", d_fz=f"{dice_faz_avg:.4f}")
    final_metrics = {}
    avg_metrics_rv = avg_result(eval_results_rv)
    avg_metrics_faz = avg_result(eval_results_faz)
    
    for key, value in avg_metrics_rv.items():
        final_metrics[f'{key}_rv'] = value
    for key, value in avg_metrics_faz.items():
        final_metrics[f'{key}_faz'] = value
        
    for m in ['dice', 'iou', 'sen', 'spe', 'acc']:
        final_metrics[f'{m}_avg'] = (final_metrics.get(f'{m}_rv', 0) + final_metrics.get(f'{m}_faz', 0)) / 2
        
    return final_metrics

def main():
    ap = argparse.ArgumentParser(description="Multi-task training, fully aligned with benchmark evaluation.")
    ap.add_argument("--model", type=str, default="MultiTaskOCTAMamba", choices=["MultiTaskOCTFormer", "MultiTaskOCTAMamba"])
    ap.add_argument("--dataset", type=str, default="OCTA500_3M", help="Dataset name.")
    ap.add_argument("--epochs", type=int, default=100, help="Total epochs.")
    ap.add_argument("--batch", type=int, default=2, help="Batch size.")
    ap.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--workers", type=int, default=4, help="Num workers.")
    ap.add_argument("--use_tta", action='store_true', help="Enable TTA on test set.")
    ap.add_argument("--rv_weight", type=float, default=1.0, help="RV loss weight.")
    ap.add_argument("--faz_weight", type=float, default=1.0, help="FAZ loss weight.")
    ap.add_argument("--faz_crop", type=int, default=192, help="FAZ crop size.")
    ap.add_argument("--end_to_end", action='store_true', help="Enable end-to-end training.")
    args = ap.parse_args()
    set_seed(args.seed)
    
    FAZ_CROP_SIZE = (args.faz_crop, args.faz_crop) if args.faz_crop > 0 else None
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join("results", f"{args.model}_{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    json_log_path = os.path.join(output_dir, "results.json")
    print(f"Results will be saved to: {output_dir}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    all_datasets = prepareDatasets()
    ds_name = args.dataset
    ld_train = DataLoader(all_datasets[ds_name]["train"], batch_size=args.batch, shuffle=True,  num_workers=args.workers, pin_memory=True)
    ld_val   = DataLoader(all_datasets[ds_name]["val"],   batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    ld_test  = DataLoader(all_datasets[ds_name]["test"],  batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    model_class = MultiTaskOCTFormer if args.model == "MultiTaskOCTFormer" else MultiTaskOCTAMamba
    model = model_class(tasks=[ds_name], use_checkpoint=True, faz_crop_size=args.faz_crop, end_to_end=args.end_to_end).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, total_steps=args.epochs, pct_start=0.1, div_factor=2, final_div_factor=100.0, anneal_strategy='cos')
    
    best_val_metrics = {'dice_avg': -1.0, 'epoch': -1}
    best_rv_dice = -1.0
    best_faz_dice = -1.0
    best_rv_epoch = -1
    best_faz_epoch = -1
    
    def run_test_and_log(tag_name, current_epoch, current_train_metrics, current_val_metrics):
        print(f"      🕒 Running test for '{tag_name}' at epoch {current_epoch}...")
        test_metrics = evaluate_one_epoch(model, ld_test, device, use_tta=args.use_tta)
        
        def fmt(d): return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in d.items()}
        rec = {
            "ts": str(datetime.datetime.now()), "e": current_epoch, "tag": tag_name,
            "tta_used": args.use_tta, "train": fmt(current_train_metrics),
            "val": fmt(current_val_metrics), "test": fmt(test_metrics)
        }
        data = []
        if os.path.exists(json_log_path):
            try:
                with open(json_log_path, 'r') as f: data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {json_log_path}. Starting with a new list.")
                data = []

        data.append(rec)
        with open(json_log_path, "w") as f: json.dump(data, f, indent=4)
        print(f"      ✓ TEST results for '{tag_name}' saved to {json_log_path}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_metrics = train_one_epoch(model, ld_train, device, optimizer, FAZ_CROP_SIZE, args.rv_weight, args.faz_weight)
        scheduler.step()
        val_metrics = evaluate_one_epoch(model, ld_val, device, use_tta=False)
        
        current_dice_avg = val_metrics['dice_avg']
        current_dice_rv = val_metrics['dice_rv']
        current_dice_faz = val_metrics['dice_faz']
        
        print(f"  ├── Train Loss  : {train_metrics['loss']:.4f} (RV: {train_metrics['loss_rv']:.4f}, FAZ: {train_metrics['loss_faz']:.4f}) | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  ├── VAL AVG Dice: {current_dice_avg:.4f} (Best: {best_val_metrics['dice_avg']:.4f} at E{best_val_metrics['epoch']})")
        print(f"  ├── VAL RV  Dice: {current_dice_rv:.4f} (Best: {best_rv_dice:.4f} at E{best_rv_epoch})")
        print(f"  └── VAL FAZ Dice: {current_dice_faz:.4f} (Best: {best_faz_dice:.4f} at E{best_faz_epoch})")

        is_late_training = epoch >= args.epochs * 0.7
        #is_late_training = True
        is_best_avg = current_dice_avg > best_val_metrics['dice_avg']
        if is_best_avg and is_late_training:
            best_val_metrics.update({'dice_avg': current_dice_avg, 'epoch': epoch})
            path = os.path.join(output_dir, "best_model_avg.pth")
            torch.save(model.state_dict(), path)
            print(f"      ✨ New best AVG model saved. Triggering test...")
            run_test_and_log("best_avg", epoch, train_metrics, val_metrics)
            

        if current_dice_rv > best_rv_dice and is_late_training:
            best_rv_dice = current_dice_rv
            best_rv_epoch = epoch
            path = os.path.join(output_dir, "best_model_rv.pth")
            torch.save(model.state_dict(), path)
            print(f"      ✨ New best RV model saved (Dice: {current_dice_rv:.4f}).")
            
        if current_dice_faz > best_faz_dice and is_late_training:
            best_faz_dice = current_dice_faz
            best_faz_epoch = epoch
            path = os.path.join(output_dir, "best_model_faz.pth")
            torch.save(model.state_dict(), path)
            print(f"      ✨ New best FAZ model saved (Dice: {current_dice_faz:.4f}).")

        is_late_training = epoch >= args.epochs * 0.7
        should_periodic_test = is_late_training and (epoch % 5 == 0)
        if should_periodic_test and not is_best_avg: 
            run_test_and_log(f"periodic_e{epoch}", epoch, train_metrics, val_metrics)
            torch.save(model.state_dict(), os.path.join(output_dir, f"periodic_model_e{epoch}.pth"))

        # Early stopping
        early_stop_threshold = 60
        if epoch - best_val_metrics['epoch'] >= early_stop_threshold and best_val_metrics['epoch'] > 0:
            print(f"\nValidation Dice (AVG) didn't improve for {early_stop_threshold} epochs. Stopping.")
            break
            
    print("\n🎉 Training finished.")

    print("\n--- Starting Final Global Evaluation on Test Set ---")
    
    final_log_data = []
    if os.path.exists(json_log_path):
        with open(json_log_path, 'r') as f: final_log_data = json.load(f)

    checkpoints_to_test = [
        ('avg', 'best_model_avg.pth', best_val_metrics['epoch']),
        ('rv', 'best_model_rv.pth', best_rv_epoch),
        ('faz', 'best_model_faz.pth', best_faz_epoch),
    ]
    def fmt(d): return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in d.items()}

    for tag, filename, best_epoch in checkpoints_to_test:
        model_path = os.path.join(output_dir, filename)
        if os.path.exists(model_path):
            print(f"-> Evaluating final best '{tag}' model from epoch {best_epoch}...")
            eval_model = model_class(tasks=[ds_name], faz_crop_size=args.faz_crop, end_to_end=args.end_to_end).to(device)
            eval_model.load_state_dict(torch.load(model_path, map_location=device))
            
            test_metrics = evaluate_one_epoch(eval_model, ld_test, device, use_tta=args.use_tta)

            final_rec = {
                "ts": str(datetime.datetime.now()), "e": best_epoch,
                "tag": f"final_best_{tag}", "tta_used": args.use_tta,
                "test": fmt(test_metrics)
            }
            final_log_data.append(final_rec)
        else:
            print(f"-> Skipping final evaluation for '{tag}': Checkpoint '{filename}' not found.")
    
    with open(json_log_path, 'w') as f:
        json.dump(final_log_data, f, indent=4)

    print("\n✅ Final global evaluation complete. Results appended to log.")
    print(f"Final best model epochs: AVG@{best_val_metrics['epoch']}, RV@{best_rv_epoch}, FAZ@{best_faz_epoch}")

if __name__ == "__main__":
    main()
