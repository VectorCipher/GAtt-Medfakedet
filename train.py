import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json
import random
import time
from glob import glob

from dataset import ManipDensityDataset, collate_keep_boxes
from manip_density_detect import UNetDensity
from train_utils import compute_loss, eval_loader

def build_or_load_fixed_splits(data_dir, img_dir, val_ratio=0.10, test_ratio=0.05):
    split_json = os.path.join(data_dir, "split_fixed_density.json")
    
    def list_ids():
        imgs = sorted(glob(os.path.join(img_dir, "*.png")))
        return [os.path.splitext(os.path.basename(p))[0] for p in imgs]
        
    ids = list_ids()
    if os.path.exists(split_json):
        with open(split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        fixed_val = set(split["val"])
        fixed_test = set(split["test"])
        train = [i for i in ids if i not in (fixed_val | fixed_test)]
        val   = [i for i in ids if i in fixed_val]
        test  = [i for i in ids if i in fixed_test]
        return train, val, test

    ids_shuf = ids.copy()
    random.shuffle(ids_shuf)
    n = len(ids_shuf)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))

    test = ids_shuf[:n_test]
    val  = ids_shuf[n_test:n_test+n_val]
    train = ids_shuf[n_test+n_val:]
    
    os.makedirs(data_dir, exist_ok=True)
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({"val": val, "test": test, "created_at": time.strftime("%F %T")}, f, indent=2)
    return train, val, test

def main():
    parser = argparse.ArgumentParser(description="Train Attention U-Net for MedFakeDet")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory (e.g., ./data/synth_ddpm_manip_2)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./weights", help="Directory to save model weights")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "images")
    den_dir = os.path.join(args.data_dir, "density")
    msk_dir = os.path.join(args.data_dir, "masks")
    lbl_dir = os.path.join(args.data_dir, "labels")

    print(f"Loading data from {args.data_dir}...")
    train_ids, val_ids, test_ids = build_or_load_fixed_splits(args.data_dir, img_dir)
    print(f"Splits: Train {len(train_ids)} | Val {len(val_ids)} | Test {len(test_ids)}")

    train_ds = ManipDensityDataset(train_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=True)
    val_ds   = ManipDensityDataset(val_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False)
    test_ds  = ManipDensityDataset(test_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_keep_boxes)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_keep_boxes)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_keep_boxes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNetDensity().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    THR = 0.28
    MIN_AREA = 20

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = 0.0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        loss_sum = 0.0
        n_batches = 0

        for batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True)
            gt_den = batch["den"].to(device, non_blocking=True)
            gt_msk = batch["msk"].to(device, non_blocking=True)

            pred_den, pred_msk_logit = model(imgs)
            loss, parts = compute_loss(pred_den, pred_msk_logit, gt_den, gt_msk)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=loss_sum/n_batches)

        # Validate
        val_metrics = eval_loader(model, val_loader, device, thr=THR, min_area=MIN_AREA)
        f1 = val_metrics['F1@0.5']
        print(f"\\nEpoch {epoch} Validation: F1@0.5: {f1:.4f}")
        
        # Save Best Model
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(args.save_dir, 'density_region_detector_attention_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path} (F1: {best_f1:.4f})")
            
    print("Training complete.")

if __name__ == "__main__":
    main()
