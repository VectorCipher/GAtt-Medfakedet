import torch
from manip_density_detect import UNetDensity

def test_model():
    model = UNetDensity()
    dummy_input = torch.randn(1, 1, 128, 128)
    out_den, out_msk = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Density output shape: {out_den.shape}")
    print(f"Mask output shape: {out_msk.shape}")
    if out_den.shape == (1, 1, 128, 128) and out_msk.shape == (1, 1, 128, 128):
        print("Test passed! Attention U-Net initialized and output dimensions match expected.")
    else:
        print("Test failed! Output dimensions are incorrect.")

if __name__ == "__main__":
    test_model()
