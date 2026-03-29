import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import random
import numpy as np
import os
import argparse
from model import get_stl_resnet18

def set_seed(seed=42):
    """Rule 2.7.3: Fixing random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PseudoDataset(Dataset):
    def __init__(self, root, indices, labels, transform=None):
        self.stl10_unlabeled = torchvision.datasets.STL10(root=root, split='unlabeled', download=False)
        self.indices = indices
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        img, _ = self.stl10_unlabeled[self.indices[i]]
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

def evaluate(model, loader, device):
    """Helper for real-time monitoring of loss and accuracy."""
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total, running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to STL-10 dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Where to save checkpoints')
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Standard normalization for STL-10
    norm = transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))

    # --- STEP 1: SUPERVISED BASELINE ---
    transform_base = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(), norm,
    ])

    trainset = torchvision.datasets.STL10(root=args.data_dir, split='train', download=True, transform=transform_base)
    testset = torchvision.datasets.STL10(root=args.data_dir, split='test', download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), norm]))

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    teacher = get_stl_resnet18().to(device)
    optimizer = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)

    print("Stage 1: Training Teacher (100 Epochs)...")
    for epoch in range(1, 101):
        teacher.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(teacher(inputs), targets)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            acc, v_loss = evaluate(teacher, testloader, device)
            print(f"[Teacher] Epoch {epoch} | Test Acc: {acc:.2f}% | Loss: {v_loss:.4f}")
            torch.save(teacher.state_dict(), os.path.join(args.save_dir, 'teacher_latest.pth'))

    # --- STEP 4: ROUND 3 PSEUDO-LABEL SCANNING ---
    print("\n🔍 Stage 2: Scanning Unlabeled Data...")
    teacher.eval()
    unlabeled_set = torchvision.datasets.STL10(root=args.data_dir, split='unlabeled',
                                              transform=transforms.Compose([transforms.ToTensor(), norm]))
    un_loader = DataLoader(unlabeled_set, batch_size=128, shuffle=False, num_workers=2)
    indices, labels = [], []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(un_loader):
            outputs = teacher(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            mask = conf > 0.90
            if mask.any():
                idx_batch = torch.where(mask)[0]
                for idx in idx_batch:
                    indices.append(i * 128 + idx.item())
                    labels.append(pred[idx].item())
    print(f"Found {len(labels)} high-confidence images.")

    # --- STEP 5: 85% TARGET SPRINT ---
    print("\n🏃 Stage 3: High-Accuracy Sprint (50 Epochs)...")
    transform_sprint = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), norm,
    ])

    pseudoc_set = PseudoDataset(args.data_dir, indices, labels, transform=transform_sprint)
    combined_loader = DataLoader(ConcatDataset([trainset, pseudoc_set]), batch_size=128, shuffle=True, num_workers=2)

    student = get_stl_resnet18().to(device)
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, 51):
        student.train()
        for inputs, targets in combined_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(student(inputs), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            acc, v_loss = evaluate(student, testloader, device)
            print(f"[Student] Epoch {epoch} | Test Acc: {acc:.2f}% | Loss: {v_loss:.4f}")

    # --- STEP 6: SWA REFINEMENT & FP16 SAVE ---
    print("\n💎 Stage 4: SWA Refinement (10 Epochs)...")
    swa_model = AveragedModel(student)
    swa_sch = SWALR(optimizer, swa_lr=0.001)
    for _ in range(10):
        student.train()
        for inputs, targets in combined_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(student(inputs), targets).backward()
            optimizer.step()
        swa_model.update_parameters(student)
        swa_sch.step()

    update_bn(combined_loader, swa_model)

#F16
    state_dict = swa_model.module.state_dict()
    for key in state_dict: state_dict[key] = state_dict[key].half()

    final_path = 'model.pth'
    torch.save(state_dict, final_path)
    print(f"\nPipeline complete. Final model saved as {final_path}")

    # Final Validation check
    final_acc, _ = evaluate(swa_model, testloader, device)
    print(f" Final SWA Test Accuracy: {final_acc:.2f}%")

if __name__ == '__main__':
    main()