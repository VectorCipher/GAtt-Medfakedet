import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Import from the modular package
from transunet_density import TransUNetDensity
from manip_density_detect import preprocess_gray_128, density_to_boxes
_MODEL = None
def load_transunet(model_path, device):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    
    m = TransUNetDensity().to(device)
    sd = torch.load(model_path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    _MODEL = m
    return m
@torch.no_grad()
def detect_transunet(model_path, image_path, device, thr=0.28, min_area=20):
    model = load_transunet(model_path, device)
    x, _ = preprocess_gray_128(image_path, img_size=128)
    x = x.to(device)
    pred_den, _ = model(x)
    den = pred_den[0,0].detach().cpu().numpy().astype(np.float32)  # [H,W] in [0,1]
    
    # Do NOT normalize before thresholding
    boxes = density_to_boxes(den, thr=thr, min_area=min_area)
    
    return {
        "is_manipulated": bool(len(boxes) > 0),
        "density_max": float(den.max()),
        "count": int(len(boxes))
    }
def get_all_images(folder_path):
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    images = []
    for p in patterns:
        images.extend(glob.glob(os.path.join(folder_path, p), recursive=True))
    return images
def main():
    model_path = r"d:\Projects\GAtt-Medfakedet\weights\density_region_detector_transunet_best.pth"
    ct_dir = r"d:\Projects\CT_injection"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"Error: TransUNet Model weights not found at {model_path}")
        print("Make sure you have fully trained it and the path is correct.")
        return
    if not os.path.exists(ct_dir):
        print(f"Error: CT dataset not found at {ct_dir}")
        return
    categories = {
        "TM_jpeg": {"label": 0, "name": "Real (TM)"},
        "FM_CTGAN_Jpeg": {"label": 1, "name": "Fake (CTGAN)"},
        "FM_SD_Jpeg": {"label": 1, "name": "Fake (SD)"}
    }
    y_true = []
    y_pred = []
    results_by_category = {cat: {"y_true": [], "y_pred": []} for cat in categories}
    print("Gathering images...")
    all_tasks = []
    for cat_folder, info in categories.items():
        folder_path = os.path.join(ct_dir, cat_folder)
        if not os.path.exists(folder_path):
            continue
        
        imgs = get_all_images(folder_path)
        print(f"Found {len(imgs)} images in {cat_folder}")
        for img in imgs:
            all_tasks.append({
                "img_path": img,
                "label": info["label"],
                "category": cat_folder
            })
            
    if len(all_tasks) == 0:
        print("No images found to process.")
        return
    print("\nEvaluating TransUNet on CT_injection dataset...")
    for task in tqdm(all_tasks, desc="Processing Images"):
        img_path = task["img_path"]
        label = task["label"]
        cat = task["category"]
        
        try:
            out = detect_transunet(model_path, img_path, device, thr=0.28, min_area=20)
            pred_fake = 1 if out["is_manipulated"] else 0
            
            y_true.append(label)
            y_pred.append(pred_fake)
            
            results_by_category[cat]["y_true"].append(label)
            results_by_category[cat]["y_pred"].append(pred_fake)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    # Compute overall metrics
    print("\n" + "="*40)
    print("OVERALL METRICS (TransUNet - Real vs Fake)")
    print("="*40)
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
    print(f"Actual Real(0)   {cm[0][0]:<19} {cm[0][1]}")
    if len(cm) > 1:
        print(f"Actual Fake(1)   {cm[1][0]:<19} {cm[1][1]}")
    else:
        print(f"Actual Fake(1)   0                   0")
    # Compute per-category accuracy
    print("\n" + "="*40)
    print("PER-CATEGORY ACCURACY")
    print("="*40)
    for cat, data in results_by_category.items():
        if len(data["y_true"]) > 0:
            cat_acc = accuracy_score(data["y_true"], data["y_pred"])
            name = categories[cat]["name"]
            print(f"{name:15s} : {cat_acc:.4f} ({sum(i == j for i, j in zip(data['y_true'], data['y_pred']))}/{len(data['y_true'])})")
if __name__ == "__main__":
    main()