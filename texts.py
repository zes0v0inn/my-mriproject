from model import build_model
import torch
from torchinfo import summary


def test_models(device="cpu"):
    """
    test_models() runs a simple test to verify that the models can process a random input tensor and produce an output of the expected shape. It iterates over a list of model names, builds each model, and checks the input and output shapes.
    """

    for name in ["basic_unet", "monai_unet"]:
        print(f"\n{'='*60}")
        model = build_model(name, features=(16, 32, 64, 128)).to(device)
        x = torch.randn(1, 4, 64, 64, 64, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"  Input  : {x.shape}")
        print(f"  Output : {y.shape}")
        assert y.shape == (1, 3, 64, 64, 64), f"Shape mismatch! Got {y.shape}"
        summary(model, input_size=(1, 4, 64, 64, 64), device=device)
        print("  ✓ Pass")


if __name__ == "__main__":   
    test_models()