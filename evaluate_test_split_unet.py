import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Import custom modules
from dataset import ManipDensityDataset, collate_keep_boxes
from manip_density_detect import UNetDensity, density_to_boxes

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    A=(a[2]-a[0])*(a[3]-a[1])
    B=(b[2]-b[0])*(b[3]-b[1])
    return inter/(A+B-inter+1e-6)

def evaluate_test_split_unet():
    split_json = r"d:\Projects\split_fixed_density.json"
    data_dir = r"d:\Projects\Dataset_deepfake_detection"
    model_path = r"D:\Projects\GAtt-Medfakedet\weights\density_region_detector.pth"
    
    # If the user saved it elsewhere
    if not os.path.exists(model_path):
        model_path = r"D:\Projects\GAtt-Medfakedet\weights\density_region_detector.pth"

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

    print("Loading UNet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDensity().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model weights not found at {model_path}. Proceeding with untrained weights for testing the script.")

    model.eval()

    test_ds = ManipDensityDataset(test_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_keep_boxes)

    # Classification Metrics (Real vs Fake)
    y_true = []
    y_pred = []

    # Localization Metrics (Bounding Box F1 @ IoU 0.5)
    TP = FP = FN = 0
    ious = []
    abs_count_err = []

    THR = 0.28
    MIN_AREA = 20
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
                
                # BUGFIX: Do NOT normalize the density map, use absolute values to prevent False Positives on Real images
                pred_boxes = density_to_boxes(d, thr=THR, min_area=MIN_AREA)
                gt_boxes = gt_boxes_list[i]
                
                # --- Image-level Classification ---
                is_actual_fake = 1 if len(gt_boxes) > 0 else 0
                is_pred_fake = 1 if len(pred_boxes) > 0 else 0
                
                y_true.append(is_actual_fake)
                y_pred.append(is_pred_fake)

                # --- Box-level Localization ---
                abs_count_err.append(abs(len(pred_boxes) - len(gt_boxes)))

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
                        TP += 1
                        matched.add(best_j)
                        ok = True
                        ious.append(best_iou)
                    if not ok:
                        FP += 1
                FN += (len(gt_boxes) - len(matched))

    print("\n" + "="*50)
    print("1. IMAGE-LEVEL CLASSIFICATION METRICS (Real vs Fake)")
    print("="*50)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
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

    print("\n" + "="*50)
    print(f"2. BOX-LEVEL LOCALIZATION METRICS (IoU >= {IOU_THR})")
    print("="*50)
    loc_prec = TP / (TP + FP + 1e-6)
    loc_rec  = TP / (TP + FN + 1e-6)
    loc_f1   = 2 * loc_prec * loc_rec / (loc_prec + loc_rec + 1e-6)
    miou = float(np.mean(ious)) if len(ious)>0 else 0.0
    count_mae = float(np.mean(abs_count_err)) if len(abs_count_err)>0 else 0.0

    print(f"Localization Precision : {loc_prec:.4f}")
    print(f"Localization Recall    : {loc_rec:.4f}")
    print(f"Localization F1@0.5    : {loc_f1:.4f}")
    print(f"Mean IoU (Matched)     : {miou:.4f}")
    print(f"Count MAE              : {count_mae:.4f}")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print("="*50)

if __name__ == "__main__":
    evaluate_test_split_unet()
