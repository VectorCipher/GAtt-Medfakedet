import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Import custom modules
from dataset import ManipDensityDataset, collate_keep_boxes
from manip_density_detect import density_to_boxes
from transunet_density import TransUNetDensity

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    A=(a[2]-a[0])*(a[3]-a[1])
    B=(b[2]-b[0])*(b[3]-b[1])
    return inter/(A+B-inter+1e-6)

def evaluate_test_split():
    split_json = r"d:\Projects\split_fixed_density.json"
    data_dir = r"d:\Projects\Dataset_deepfake_detection"
    model_path = r"D:\Projects\GAtt-Medfakedet\weights\ltsrl_transunet_best (1).pth"
    
    # If the user saved it in args.save_dir, it might be in GAtt-Medfakedet directly
    if not os.path.exists(model_path):
        model_path = r"D:\Projects\GAtt-Medfakedet\weights\ltsrl_transunet_best (1).pth"

    img_dir = os.path.join(data_dir, "images")
    den_dir = os.path.join(data_dir, "density")
    msk_dir = os.path.join(data_dir, "masks")
    lbl_dir = os.path.join(data_dir, "labels")

    print("Loading test split from:", split_json)
    with open(split_json, 'r') as f:
        splits = json.load(f)
    
    test_ids = splits.get("test", [])
    if not test_ids:
        print("Error: No 'test' array found in split JSON.")
        return
        
    print(f"Found {len(test_ids)} test images.")

    print("Loading TransUNet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransUNetDensity().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model weights not found at {model_path}. Proceeding with untrained weights for testing the script.")

    model.eval()

    test_ds = ManipDensityDataset(test_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_keep_boxes)

    # Grid Search Parameters
    thresholds = [0.1]
    areas = [20]
    
    # Initialize accumulators
    results = {}
    for t in thresholds:
        for a in areas:
            results[(t, a)] = {
                "y_true": [], "y_pred": [],
                "TP": 0, "FP": 0, "FN": 0,
                "ious": [], "abs_count_err": []
            }

    IOU_THR = 0.5

    print("Starting Evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            imgs = batch["img"].to(device, non_blocking=True)
            gt_boxes_list = batch["boxes"]

            pred_den, _ = model(imgs)
            pd = pred_den.detach().cpu().numpy()  # [B,1,H,W]

            for i in range(pd.shape[0]):
                d = pd[i,0]
                gt_boxes = gt_boxes_list[i]
                
                for t in thresholds:
                    for a in areas:
                        # BUGFIX: Do NOT normalize the density map, use absolute values to prevent False Positives on Real images
                        pred_boxes = density_to_boxes(d, thr=t, min_area=a)
                        
                        # --- Image-level Classification ---
                        is_actual_fake = 1 if len(gt_boxes) > 0 else 0
                        is_pred_fake = 1 if len(pred_boxes) > 0 else 0
                        
                        res = results[(t, a)]
                        res["y_true"].append(is_actual_fake)
                        res["y_pred"].append(is_pred_fake)

                        # --- Box-level Localization ---
                        res["abs_count_err"].append(abs(len(pred_boxes) - len(gt_boxes)))

                        matched = set()
                        for p in pred_boxes:
                            ok = False
                            best_iou = 0.0
                            best_j = -1
                            for j, g in enumerate(gt_boxes):
                                if j in matched:
                                    continue
                                v = iou(p, g)
                                if v > best_iou:
                                    best_iou = v
                                    best_j = j
                            if best_iou >= IOU_THR:
                                res["TP"] += 1
                                matched.add(best_j)
                                ok = True
                                res["ious"].append(best_iou)
                            if not ok:
                                res["FP"] += 1
                        res["FN"] += (len(gt_boxes) - len(matched))

    print("\n" + "="*85)
    print("GRID SEARCH RESULTS (Sorted by Localization F1@0.5)")
    print("="*85)
    print(f"{'Thr':<5} | {'Area':<5} | {'Cls F1':<8} | {'Cls Acc':<8} | {'Loc F1':<8} | {'Loc Prec':<8} | {'Loc Rec':<8} | {'mIoU':<6}")
    print("-" * 85)
    
    summary = []
    for (t, a), res in results.items():
        # Classification
        y_true_cur = res["y_true"]
        y_pred_cur = res["y_pred"]
        cls_acc = accuracy_score(y_true_cur, y_pred_cur)
        cls_f1 = f1_score(y_true_cur, y_pred_cur, zero_division=0)
        
        # Localization
        TP_cur = res["TP"]
        FP_cur = res["FP"]
        FN_cur = res["FN"]
        loc_prec = TP_cur / (TP_cur + FP_cur + 1e-6)
        loc_rec  = TP_cur / (TP_cur + FN_cur + 1e-6)
        loc_f1   = 2 * loc_prec * loc_rec / (loc_prec + loc_rec + 1e-6)
        miou = float(np.mean(res["ious"])) if len(res["ious"]) > 0 else 0.0
        
        summary.append({
            "t": t, "a": a, 
            "cls_f1": cls_f1, "cls_acc": cls_acc,
            "loc_f1": loc_f1, "loc_prec": loc_prec, "loc_rec": loc_rec, "miou": miou
        })
        
    # Sort by Localization F1 descending
    summary.sort(key=lambda x: x["loc_f1"], reverse=True)
    
    for s in summary:
        print(f"{s['t']:<5.2f} | {s['a']:<5d} | {s['cls_f1']:<8.4f} | {s['cls_acc']:<8.4f} | {s['loc_f1']:<8.4f} | {s['loc_prec']:<8.4f} | {s['loc_rec']:<8.4f} | {s['miou']:<6.4f}")
        
    print("="*85)
    best = summary[0]
    print(f"BEST CONFIGURATION: Threshold = {best['t']}, Min Area = {best['a']}")
    print(f"Best Loc F1: {best['loc_f1']:.4f}")
    print(f"Best Cls F1: {best['cls_f1']:.4f}")
    
    # Print detailed metrics for the best (or only) configuration
    best_res = results[(best['t'], best['a'])]
    y_true_cur = best_res["y_true"]
    y_pred_cur = best_res["y_pred"]
    acc = accuracy_score(y_true_cur, y_pred_cur)
    prec = precision_score(y_true_cur, y_pred_cur, zero_division=0)
    rec = recall_score(y_true_cur, y_pred_cur, zero_division=0)
    f1 = f1_score(y_true_cur, y_pred_cur, zero_division=0)
    cm = confusion_matrix(y_true_cur, y_pred_cur)
    
    print("\n\n" + "="*50)
    print(f"DETAILED METRICS FOR THRESHOLD {best['t']} AND MIN_AREA {best['a']}")
    print("="*50)
    print("\n1. IMAGE-LEVEL CLASSIFICATION METRICS (Real vs Fake)")
    print("="*50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                 Predicted Real(0)   Predicted Fake(1)")
    print(f"Actual Real(0)   {cm[0][0]:<19} {cm[0][1] if cm.shape[1]>1 else 0}")
    if len(cm) > 1:
        print(f"Actual Fake(1)   {cm[1][0]:<19} {cm[1][1]}")
    else:
        print(f"Actual Fake(1)   0                   0")

    count_mae = float(np.mean(best_res["abs_count_err"])) if len(best_res["abs_count_err"]) > 0 else 0.0
    print("\n" + "="*50)
    print(f"2. BOX-LEVEL LOCALIZATION METRICS (IoU >= {IOU_THR})")
    print("="*50)
    print(f"Localization Precision : {best['loc_prec']:.4f}")
    print(f"Localization Recall    : {best['loc_rec']:.4f}")
    print(f"Localization F1@0.5    : {best['loc_f1']:.4f}")
    print(f"Mean IoU (Matched)     : {best['miou']:.4f}")
    print(f"Count MAE              : {count_mae:.4f}")
    print(f"TP: {best_res['TP']} | FP: {best_res['FP']} | FN: {best_res['FN']}")
    print("="*50)

if __name__ == "__main__":
    evaluate_test_split()
