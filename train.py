import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import torch.nn.functional as F

from model import get_stl_resnet18, get_student_model


#SEED
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# KD FUNCTIONS

def topk_logits(logits, k=4):
    return torch.topk(logits, k, dim=1)


def distillation_loss(student_logits, teacher_logits, T=4, k=4):
    t_vals, t_idx = topk_logits(teacher_logits, k)
    s_vals = torch.gather(student_logits, 1, t_idx)

    t_probs = F.softmax(t_vals / T, dim=1)
    s_log = F.log_softmax(s_vals / T, dim=1)

    return F.kl_div(s_log, t_probs, reduction='batchmean') * (T * T)


# EVALUATION FUNCTION

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



# PSEUDO DATASET

class PseudoDataset(Dataset):
    def __init__(self, root, indices, labels, transform):
        self.dataset = torchvision.datasets.STL10(root=root, split='unlabeled')
        self.indices = indices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.dataset[self.indices[i]]
        return self.transform(img), self.labels[i]


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = transforms.Normalize((0.4467,0.4398,0.4066),(0.2603,0.2566,0.2713))

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(96,(0.6,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2,9),
        transforms.ToTensor(),
        norm
    ])

    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    trainset = torchvision.datasets.STL10(args.data_dir, split='train', transform=train_tf)
    testset = torchvision.datasets.STL10(args.data_dir, split='test', transform=test_tf)
    unlabeled = torchvision.datasets.STL10(args.data_dir, split='unlabeled', transform=test_tf)

    trainloader = DataLoader(trainset,128,True,6,pin_memory=True)
    testloader = DataLoader(testset,100,False,6,pin_memory=True)
    unloader = DataLoader(unlabeled,128,False,6,pin_memory=True)

    # ---------------- STAGE 1 ----------------
    model = get_stl_resnet18().to(device)
    opt = optim.SGD(model.parameters(),0.1,0.9,5e-4)
    sched = CosineAnnealingLR(opt,150)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(1,151):
        model.train()
        for x,y in trainloader:
            x,y = x.to(device),y.to(device)
            opt.zero_grad()
            loss = ce(model(x),y)
            loss.backward()
            opt.step()
        sched.step()

        if epoch%5==0:
            acc = evaluate(model,testloader,device)
            if acc>best_acc:
                best_acc=acc
                torch.save(model.state_dict(),os.path.join(args.save_dir,"best_model.pth"))

    # ---------------- STAGE 2 ----------------
    teacher = get_stl_resnet18().to(device)
    teacher.load_state_dict(torch.load(os.path.join(args.save_dir,"best_model.pth")))
    teacher.eval()

    indices, labels = [], []
    with torch.no_grad():
        for i,(x,_) in enumerate(unloader):
            x = x.to(device)
            probs = torch.softmax(teacher(x),1)
            conf,pred = probs.max(1)
            for j in torch.where(conf>0.95)[0]:
                indices.append(i*128+j.item())
                labels.append(pred[j].item())

    pseudo = PseudoDataset(args.data_dir,indices,labels,train_tf)
    comb_loader = DataLoader(ConcatDataset([trainset,pseudo]),128,True,6,pin_memory=True)

    student = get_stl_resnet18().to(device)
    student.load_state_dict(torch.load(os.path.join(args.save_dir,"best_model.pth")))

    opt = optim.SGD(student.parameters(),0.05,0.9,5e-4)
    sched = CosineAnnealingLR(opt,50)

    for epoch in range(1,51):
        student.train()
        for x,y in comb_loader:
            x,y = x.to(device),y.to(device)
            opt.zero_grad()
            loss = ce(student(x),y)
            loss.backward()
            opt.step()
        sched.step()

    torch.save(student.state_dict(),os.path.join(args.save_dir,"best_student.pth"))

    # ---------------- STAGE 3 ----------------
    teacher = student.eval()
    student = get_student_model().to(device)

    opt = optim.SGD(student.parameters(),0.05,0.9,5e-4)
    ce = nn.CrossEntropyLoss()

    best_acc=0
    for epoch in range(1,151):
        student.train()
        for x,y in trainloader:
            x,y = x.to(device),y.to(device)

            with torch.no_grad():
                t = teacher(x)

            s = student(x)
            loss = 0.3*ce(s,y) + 0.7*distillation_loss(s,t)

            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = evaluate(student,testloader,device)
        if acc>best_acc:
            best_acc=acc
            torch.save(student.state_dict(),"model.pth")

    print(f"FINAL ACC: {best_acc:.2f}")


if __name__=="__main__":
    main()