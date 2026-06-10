import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

# Import custom modules
from dataset import ManipDensityDataset, collate_keep_boxes
from manip_density_detect import density_to_boxes
from transunet_density import TransUNetDensity

def visualize_transunet_samples(num_samples=25):
    split_json = r"d:\Projects\split_fixed_density.json"
    data_dir = r"d:\Projects\Dataset_deepfake_detection"
    model_path = r"D:\Projects\GAtt-Medfakedet\weights\TransUnet_Epoch17.pth"
    output_dir = r"D:\Projects\GAtt-Medfakedet\visualizations"
    
    if not os.path.exists(model_path):
        model_path = r"D:\Projects\GAtt-Medfakedet\weights\TransUnet_Epoch17.pth"

    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(data_dir, "images")
    den_dir = os.path.join(data_dir, "density")
    msk_dir = os.path.join(data_dir, "masks")
    lbl_dir = os.path.join(data_dir, "labels")

    print("Loading test split...")
    with open(split_json, 'r') as f:
        splits = json.load(f)
    
    test_ids = splits.get("test", [])
    if not test_ids:
        print("Error: No 'test' array found.")
        return

    # Shuffle to get random samples, or pick a mix of real/fake if we wanted to
    # Here we just shuffle and pick the first `num_samples`
    random.seed(42)  # For reproducibility
    random.shuffle(test_ids)
    sample_ids = test_ids[:num_samples]

    print("Loading TransUNet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransUNetDensity().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Error: Weights not found at {model_path}")
        return

    model.eval()

    test_ds = ManipDensityDataset(sample_ids, img_dir, den_dir, msk_dir, lbl_dir, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_keep_boxes)

    THR = 0.28
    MIN_AREA = 20

    print(f"Generating visualizations for {len(sample_ids)} samples...")
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            sid = batch["id"][0]
            img_tensor = batch["img"].to(device)
            gt_den_tensor = batch["den"]
            gt_boxes = batch["boxes"][0]

            # Inference
            pred_den_tensor, _ = model(img_tensor)
            
            # Convert to numpy for plotting
            img = img_tensor[0, 0].cpu().numpy()            # [128, 128]
            gt_den = gt_den_tensor[0, 0].cpu().numpy()      # [128, 128]
            pred_den = pred_den_tensor[0, 0].cpu().numpy()  # [128, 128]
            
            # Extract predicted boxes using unnormalized density map
            pred_boxes = density_to_boxes(pred_den, thr=THR, min_area=MIN_AREA)

            # Create plot
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f"Sample: {sid}", fontsize=16)

            # 1. Input Image
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title("Input Image")
            axs[0].axis('off')

            # 2. GT Density Map
            axs[1].imshow(gt_den, cmap='jet')
            axs[1].set_title("GT Density Map")
            axs[1].axis('off')

            # 3. Predicted Density Map
            axs[2].imshow(pred_den, cmap='jet')
            axs[2].set_title(f"Predicted Density Map\n(Max: {pred_den.max():.3f})")
            axs[2].axis('off')

            # 4. Extracted Bounding Boxes (on top of input image)
            axs[3].imshow(img, cmap='gray')
            axs[3].set_title(f"Bounding Boxes\nGT: Green | Pred: Red")
            axs[3].axis('off')

            # Draw Ground Truth Boxes
            for (x1, y1, x2, y2) in gt_boxes:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='green', facecolor='none', label='GT')
                axs[3].add_patch(rect)

            # Draw Predicted Boxes
            for (x1, y1, x2, y2) in pred_boxes:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Pred')
                axs[3].add_patch(rect)
                
            # Add simple custom legend
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='green', lw=2),
                            Line2D([0], [0], color='red', lw=2, linestyle='--')]
            axs[3].legend(custom_lines, ['Ground Truth', 'Predicted'], loc='upper right', fontsize='small')

            plt.tight_layout()
            out_file = os.path.join(output_dir, f"sample_{idx+1:02d}_{sid}.png")
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"Successfully saved {num_samples} visualizations to {output_dir}")

if __name__ == "__main__":
    visualize_transunet_samples(num_samples=25)
