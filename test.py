import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os

from model import get_stl_resnet18, get_student_model


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total


# -----------------------------
# Load model safely
# -----------------------------
def load_model(path, device):
    assert os.path.exists(path), f"❌ File not found: {path}"

    # Decide architecture
    if "model.pth" in path:
        print("📦 Loading KD Student Model")
        model = get_student_model()
    else:
        print("📦 Loading ResNet18 Model")
        model = get_stl_resnet18()

    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--ckpt_dir', default='./checkpoints')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # -----------------------------
    # Dataset
    # -----------------------------
    norm = transforms.Normalize((0.4467, 0.4398, 0.4066),
                                (0.2603, 0.2566, 0.2713))

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    testset = torchvision.datasets.STL10(
        root=args.data_dir,
        split='test',
        transform=test_tf,
        download=False
    )

    testloader = DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4
    )

    print("✅ Dataset Loaded\n")

    # -----------------------------
    # Evaluate models
    # -----------------------------
    paths = {
        "Supervised": os.path.join(args.ckpt_dir, "best_model.pth"),
        "Semi-Supervised": os.path.join(args.ckpt_dir, "best_student.pth"),
        "KD Final": "model.pth"
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Skipping {name} (not found)")
            continue

        print(f"\n🔍 Evaluating {name} Model")
        model = load_model(path, device)
        acc = evaluate(model, testloader, device)

        size_mb = os.path.getsize(path) / (1024 * 1024)

        print(f"📊 Accuracy: {acc:.2f}%")
        print(f"📦 Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()