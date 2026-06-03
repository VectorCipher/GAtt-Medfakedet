# MFDAttention - Modular Setup for Kaggle / Google Colab

This repository has been refactored into a modular Python package, making it exceptionally easy to train the new **Attention U-Net** on Kaggle, Google Colab, or any cloud GPU provider without scrolling through massive monolithic notebooks.

## How to use on Colab / Kaggle

1. **Clone the repository inside your notebook cell:**
   ```python
   !git clone https://github.com/VectorCipher/MFDAttention.git
   %cd MFDAttention
   ```

2. **Upload/Mount your Dataset**
   - On Colab: Mount Google Drive or upload the zip and extract it.
   - On Kaggle: Add the dataset via the `+ Add Data` button on the right panel.

3. **Train the Attention U-Net**
   We have provided a clean Python script `train.py` that takes command line arguments, completely eliminating the need for monolithic notebooks.

   To train, simply run the script and point it to your data directory:
   ```bash
   python train.py --data_dir ./data/synth_ddpm_manip_2 --epochs 100 --batch_size 16
   ```

## Modular Architecture

Instead of having thousands of lines of code inside a single `.ipynb` file, the codebase is now split logically:
- `manip_density_detect.py`: Contains the `AttentionBlock` and `UNetDensity` architecture.
- `dataset.py`: Contains the `ManipDensityDataset` PyTorch class and data loading utilities.
- `train_utils.py`: Contains loss functions (density regression, mask BCE, Total Variation) and evaluation metrics (IoU, F1).
- `train.py`: A clean orchestration script to run the training loop via command line arguments.
